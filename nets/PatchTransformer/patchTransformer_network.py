import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.STNet.affine_STN import AffineSTNNet
from nets.STNet.tps_STN import TpsSTNNet
from utils.tools import transform_template_input, transform_scaled_template_input, transform_blur_template_input
from nets.LightingNet import lighting_net_builder
from nets.ColorNet.PCT_transformation import *
from losses.mask_losses import SIMMMaskLoss, L1MaskLoss, L2MaskLoss, SmoothL1Loss
from nets.PatchTransformer.patch_blurring import PatchBlurringModule

PCT_INFO = {'PCT': PCTTransformation, 'PCTLinear': PCTLinearTransformation, \
            'PCTLinearBias': PCTLinearBiasTransformation, 'PCTNeural': PCTNeuralTransformation}
# LCT_INFO = {'cc': CC_FCN4, 'gen': Generator, 'cc_drn18': CCDRN18}
#LOSS_INFO = {'L1':nn.L1Loss(), 'L2':nn.MSELoss(), 'SSIM':SIMMMaskLoss(val_range=1)}
#LOSS_INFO = {'L1':nn.L1Loss(), 'L2':nn.MSELoss(), 'L1Mask':L1MaskLoss(), 'SIMMMask':SIMMMaskLoss()}
LOSS_INFO = {'L1':nn.L1Loss(), 'L2':nn.MSELoss(), 'L1Mask':L1MaskLoss(), 'L2Mask':L2MaskLoss(), 'SIMMMask':SIMMMaskLoss(), 'SmoothL1Mask':SmoothL1Loss(beta=0.5)}

class PatchTransformerNetwork(nn.Module):
    def __init__(self, config):
        super(PatchTransformerNetwork, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

        if config['STN'] == 'affine':
            self.STN = AffineSTNNet(config)
        elif config['STN'] == 'tps':
            self.STN = TpsSTNNet(config)

        self.learnableSTN = True if config['learnableSTN'] is True else False
        if self.learnableSTN:
            self.stn_loss = LOSS_INFO[self.config['STN_loss']]

        self.PBM = PatchBlurringModule() if self.config['use_PBM'] else None
        self.learnablePBM = True if self.PBM is not None else False

        self.PCT = PCT_INFO[self.config['PrinterCT']](self.config) if self.config['use_PCT'] else None
        #self.learnablePCT = True if self.PCT is not None and self.config['PrinterCT'] != 'PCT' else False
        self.learnablePCT = True if self.PCT is not None else False
        if self.learnablePCT:
            self.pct_loss = LOSS_INFO[self.config['PCT_loss']]

        self.LCT = lighting_net_builder(self.config) if self.config['use_LCT'] else None
        # self.LCT = LCT_INFO[self.config['LightingCT']](self.config) if self.config['use_LCT'] else None
        self.learnableLCT = True if self.LCT is not None else False
        if self.learnableLCT:
            self.lct_loss = LOSS_INFO[self.config['LCT_loss']]

#        self.template_scaling_factor = config['template_scaling_factor']
#        self.scale_template = True if self.template_scaling_factor > 0 else False
        self.template_resize = config['template_resize']

        self.lct_input_size = tuple(config['lct_input_size']) if 'lct_input_size' in config else (256, 256)
        print ('\n-------------------------------------------------')
        print ('Generator input size: (%d %d)' % (config['generator_input_size'][0], config['generator_input_size'][1]))
        print ('LCT input size: (%d %d)\n' % (self.lct_input_size[0], self.lct_input_size[1]))
        if self.template_resize:
            print ('Template resize is ENABLED')

        if self.learnablePBM:
            print ('PBM is learnable')
        if self.learnableSTN:
            print ('STN is learnable') 
        if self.learnablePCT:
            print ('PCT is learnable') 
        if self.learnableLCT:
            print ('LCT is learnable')
        print ('--------------------------------------------------\n')
        
    def learnable(self):
        learnable_params = []

        if self.learnablePBM:
            learnable_params += list(self.PBM.parameters())

        if self.learnableSTN:
            learnable_params += list(self.STN.parameters())

        if self.learnablePCT:
            learnable_params += list(self.PCT.parameters())

        if self.learnableLCT:
            learnable_params += list(self.LCT.parameters())

        return learnable_params

   
    def train(self, mode: bool = True):
        if self.learnableSTN:
            self.STN.train(mode)
        else:
            self.STN.eval()
        
        if self.learnablePBM:
            self.PBM.train(mode)
        elif self.PBM:
            self.PBM.eval()

        if self.learnablePCT:
            self.PCT.train(mode)
        elif self.PCT:
            self.PCT.eval()

        if self.learnableLCT:
            self.LCT.train(mode)
        elif self.LCT:
            self.LCT.eval()
    
    def _pt_loss(self, loss_func, src_img, target_img, bboxes):
        return loss_func(src_img, target_img) if 'MaskLoss' not in type(loss_func).__name__ else \
                loss_func(src_img, target_img, bboxes)

    # Note: 'x' is the cropped region from the input frame and ranges between [-1 1] while template_img and frame_img
    #are within [0 1]. the output is within [0 1]
    def forward(self, x, bboxes, masks, template_img, frame_img, scale_factors):

        results = self.transform_patch(x, bboxes, masks, template_img, frame_img, scale_factors)

        x_stn = results[0]
        x_pct = results[1] if self.PCT is not None else None
        x_lct = results[-1] if self.LCT is not None else None

        ''''
        stn_loss = self.stn_loss(x_stn, frame_img) if self.learnableSTN else torch.zeros(x.shape[0]).cuda()
        pct_loss = self.pct_loss(x_pct, frame_img) if self.learnablePCT else torch.zeros(x.shape[0]).cuda()
        lct_loss = self.lct_loss(x_lct, frame_img) if self.learnableLCT else torch.zeros(x.shape[0]).cuda()
        '''
        stn_loss = self._pt_loss(self.stn_loss, x_stn, frame_img, bboxes) if self.learnableSTN else \
            torch.zeros(x.shape[0]).cuda()
        pct_loss = self._pt_loss(self.pct_loss, x_pct, frame_img, bboxes) if self.learnablePCT else \
            torch.zeros(x.shape[0]).cuda()
        lct_loss = self._pt_loss(self.lct_loss, x_lct, frame_img, bboxes) if self.learnableLCT else \
            torch.zeros(x.shape[0]).cuda()

        '''
        if self.LCT is not None:
            return x_lct, stn_loss, pct_loss, lct_loss

        if self.PCT is not None:
            return x_pct, stn_loss, pct_loss, lct_loss

        return x_stn, stn_loss, pct_loss, lct_loss
        '''
        return {'STN': (x_stn, stn_loss), 'PCT': (x_pct, pct_loss), 'LCT':(x_lct, lct_loss)}

    # return the output of each stage: STN, PCT and LCT
    def transform_patch(self, x, bboxes, masks, template_img, frame_img, scale_factors):
        '''
        y = self.PCT(template_img)
        import torchvision.transforms as transforms
        from utils.utils import visualize_detections
        import os
        import numpy as np
        for i in range(y.shape[0]):
               img = transforms.ToPILImage()(y[i].detach().cpu())
               img.save(os.path.join('tmp', 'template.png'))
        '''

        x_template = self.PBM(template_img) if self.PBM is not None else template_img
        x_stn, _, _ = self.STN(x, x_template)

        # paste the transformed patch to the frame image. To compensate the resolution loss due to distance,
        # we apply blurring on the template as necessary
        output_scales = frame_img.shape[2:]
        
        if self.template_resize:
            x_stn = transform_scaled_template_input(x_stn, bboxes, scale_factors, output_scales)
        else:
            x_stn = transform_template_input(x_stn, bboxes, output_scales)
        x_stn = x_stn * masks + frame_img * (1. - masks)

        #x_stn = torch.clamp(x_stn, 0, 0.999)

        # perform printer color transformation
        x_pct = None
        if self.PCT is not None:
            x_pct = self.PCT(x_stn)
            x_pct = x_pct * masks + frame_img * (1. - masks)
            #x_pct = torch.clamp(x_pct, 0, 0.999)

        '''
        if tuple(output_scales) != self.generator_input_size:
            resized_input_img = F.interpolate(frame_img, self.generator_input_size, mode='bilinear', align_corners=False)
        else:
            resized_input_img = frame_img
        '''

        x_lct = None
        if self.LCT is not None:
            # only compute the adjustment needed for the output
           # x_lct = self.LCT.generate(x_pct, frame_img) if x_pct is not None else \
           #      self.LCT.generate(x_stn, frame_img)
#            lct = self.LCT.forward_template(frame_img)
#            x_lct = x_pct * lct

            x_lct = x_pct if x_pct is not None else x_stn
            if type(self.LCT).__name__ in ['CC_FCN4', 'CCDRN', 'CCGenerator']:
                # Make sure the input to cc is the same as the training size.
                lct_input = F.interpolate(frame_img, self.lct_input_size, mode='bilinear', align_corners=False)
                lct = self.LCT(lct_input) #use frame image as the input
                lct = F.interpolate(lct, output_scales, mode='bilinear', align_corners=False)
                x_lct = x_lct * lct
            elif type(self.LCT).__name__ == 'Generator':
                x_lct = self.LCT(x_lct)
            else:
                raise TypeError('{} not supported!'.format(type(self.LCT).__name__))

            x_lct = x_lct * masks + frame_img * (1. - masks)
            #x_lct = torch.clamp(x_lct, 0, 0.999)

        '''
        import torchvision.transforms as transforms
        from utils.utils import visualize_detections, combine_images
        import os
        import numpy as np
        from PIL import Image
        for i in range(x_pct.shape[0]):
               frm_img = transforms.ToPILImage()(frame_img[i].detach().cpu())
               stn_img = transforms.ToPILImage()(x_stn[i].detach().cpu())
               pct_img = transforms.ToPILImage()(x_pct[i].detach().cpu())
               lct_img = transforms.ToPILImage()(x_lct[i].detach().cpu())
               h, w = frm_img.size
               cut_size = int (w * 0.33)
               crop_bb = (cut_size, 0, w - cut_size, h)
               target_img = combine_images([stn_img.crop(crop_bb), pct_img.crop(crop_bb), lct_img.crop(crop_bb), frm_img.crop(crop_bb)])
               target_img.save(os.path.join('tmp', str(i+1) +'.jpg'))
        '''

        if self.LCT is not None:
            return x_stn, x_pct, x_lct if self.PCT is not None else x_stn, x_lct

        if self.PCT is not None:
            return x_stn, x_pct

        return (x_stn,)


    def transform_and_paste_patch(self, imgs, big_patch_mask, big_patch_bb, small_patch_mask, small_patch_bb, patch_img, person_crop_img, template_img, scale_factor):

        output = self.transform_patch(patch_img, small_patch_bb, small_patch_mask, template_img, person_crop_img)
        transformed_patch = output[-1]

        if scale_factor != 1.0:
            small_patch_bb = small_patch_bb.float() * scale_factor
            small_patch_bb = small_patch_bb.int()

        # print (small_patch_bb)
        transformed_img = self.paste_patch_to_frame(transformed_patch, small_patch_bb, imgs, big_patch_bb, scale_factor)
        transformed_img = transformed_img * big_patch_mask + (1 - big_patch_mask) * imgs

        return transformed_img
    '''
    @staticmethod
    def paste_patch_to_frame(patch, patch_bb, img, img_bb, scale):
    n, c, _, _ = patch.shape
    # create tensor

    resize_patch_bb = patch_bb.float() * scale
    resize_patch_bb = resize_patch_bb.int()

    img_h, img_w = img.shape[2:]
    x = torch.cuda.FloatTensor(n, c, img_h, img_w).fill_(0)
    for i, bbox in enumerate(resize_patch_bb):
        # downsample first 
        resized_patch = F.interpolate(patch[i].unsqueeze(0), scale_factor=scale[i].item(),
                                   mode='bilinear', align_corners=False)

        # now paste
        pb, pl, ph, pw = bbox
        ib, il, ih, iw = img_bb[i]
        resized_tmpl = F.interpolate(resized_patch[:, :, pb:pb + ph, pl:pl + pw], size=(ih, iw),
                                     mode='bilinear', align_corners=False)
        x[i, :, ib:ib + ih, il:il + iw] = resized_tmpl.squeeze()

    return x
    '''
    '''
    # Note that the range of output [0 1] is different from that of input [-1 1] because of the multiplication
    #  of line 91 does not work with the input range. Also 'ground_truth' got changed. Good for now, but it's
    # better to REWRITE in the future for reducing confusion and potential issues.
    def forward(self, x, bboxes, masks, template_img, frame_img):
        # geometric
        new_template_img = self.PCT(template_img)
        x_stn, _ = self.STN(x, new_template_img)

#        x_stn, _ = self.STN(x, template_img)
        # x_stn = F.avg_pool2d(x_stn, 3, stride=1, padding=1)

        # transform the template to be the input image to the generator
        x_stn = transform_template_input(x_stn, bboxes, frame_img.shape[2:])

        # self.vis_tensor(x_stn * masks + frame_img * (1. - masks), 'before_')

        x_stn = self.PCT(x_stn)

        x_pct = x_stn * masks + frame_img * (1. - masks)

        # self.vis_tensor(x_pct, 'after_')

        x_pct = torch.clamp(x_pct, -0.999, 0.999)

        # frame_img_in_lct = transform_frames_input(frame_img, coord_w_set, (256, 256))
        # put the color in range [0 1] !!! critical
        x_pct.add_(1.0).div_(2.0)
        if self.use_LCT:
            frame_img.add_(1.0).div_(2.0)
            lct = self.LCT.forward_template(frame_img)
            lct = F.interpolate(lct, size=frame_img.shape[2:], mode='bilinear', align_corners=False)
            x_lct = x_pct * lct
            x_lct = x_lct * masks + frame_img * (1. - masks)
#            x_lct = torch.clamp(x_lct, -0.999, 0.999)
            x_lct = torch.clamp(x_lct, 0, 0.999)

        if self.use_LCT:
            return x_lct, lct
        else:
            return x_pct
    '''

    '''
    def load_from_file(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            #for key, _ in checkpoint.items():
            #    print (key)
            self.STN.load_state_dict(fix_checkpoint_key((checkpoint['stn'])), strict=True)
            print ('--- Finished loading STN module .....')

            if self.PCT is not None:
                if not self.learnablePCT:
                    self.PCT.load_state_dict(fix_checkpoint_key((checkpoint['color_transformer'])), strict=True)
                    print (checkpoint['color_transformer'])
                    print ('--- Finished loading Printer Color Transformation (%s) module' % (self.config['PrinterCT']))
                else:
                    print ('--- Printer Color Transformation (%s) module loaded from somewhere else' % (self.config['PrinterCT']))

            if self.LCT is not None:
                self.LCT.load_state_dict(fix_checkpoint_key((checkpoint['generator'])), strict=True)
                print ('--- Finished loading Lighting Color Transformation (%s) module' % (self.config['LightingCT']))

        except Exception as e:
            print (e)
            raise IOError('Warning ---mostly this is because the model was trained using different names for its submodules. Please double check if the right model is used.')
    '''
'''
def fix_checkpoint_key(checkpoint):
    new_dict = {}
    for k, v in checkpoint.items():
        # TODO: a better approach:
        new_dict[k.replace("module.", "")] = v
    return new_dict
'''
