from nets.STNet.affine_STN import AffineSTNNet
from nets.STNet.tps_STN import TpsSTNNet
from utils.tools import transform_template_input
from nets.ColorNet.cc_f4 import CC_Alex_FCN4
from nets.ColorNet.PCT_transformation import *

PCT_INFO = {'PCT':PCTTransformation, 'PCTLinear': PCTLinearTransformation, 'PCTNeural': PCTNeuralTransformation}
LCT_INFO = {'cc_fcn4':CC_Alex_FCN4}

class PatchTransformerNet(nn.Module):
    def __init__(self, config):
        super(PatchTransformerNet, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

        if config['STN'] == 'affine':
            self.STN = AffineSTNNet(config)
        elif config['STN'] == 'tps':
            self.STN = TpsSTNNet(config)

        # printer color transformation (PCT)
        self.predefined_PCT = True  if self.config['PrinterCT'] == 'PCT' else False
        self.PCT = PCT_INFO[self.config['PrinterCT']](self.config) if self.config['PrinterCT'] != 'None' else None
        # only applied PCT once. This demonstrates better performance
        self.apply_PCT_twice = config['use_double_PCT']
        assert self.apply_PCT_twice == False

        # Lighting color transformation (LCT)
        self.use_LCT = self.config['use_LightingCT']
        if self.use_LCT:
            assert self.PCT is not None
            self.LCT = LCT_INFO[self.config['LightingCT']]()

    # Note: 'x' is within [-1 1] while template_img and frame_img are within [0 1].
    # the output is within [0 1]
    def forward(self, x, bboxes, masks, template_img, frame_img):
        if self.apply_PCT_twice:
            new_template_img = self.PCT(template_img)
            x_stn, _ = self.STN(x, new_template_img)
        else:
            x_stn, _ = self.STN(x, template_img)

        # paste the transformed patch to the frame image
        x_stn = transform_template_input(x_stn, bboxes, frame_img.shape[2:])

        # perform printer color transformation
        x_pct = self.PCT(x_stn) if self.PCT is not None else x_stn
        x_pct = x_pct * masks + frame_img * (1. - masks)
        x_pct = torch.clamp(x_pct, 0, 0.999)

        if self.use_LCT:
            lct = self.LCT.forward_template(frame_img)
           # print (lct)
            x_lct = x_pct * lct
            x_lct = x_lct * masks + frame_img * (1. - masks)
            x_lct = torch.clamp(x_lct, 0, 0.999)

        if self.use_LCT:
            return x_lct, x_pct, lct
        else:
            return x_pct

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

    def load_from_file(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            #for key, _ in checkpoint.items():
            #    print (key)
            self.STN.load_state_dict(fix_checkpoint_key((checkpoint['stn'])), strict=True)
            print ('--- Finished loading STN module .....')

            if self.PCT is not None:
                if not self.predefined_PCT:
                    self.PCT.load_state_dict(fix_checkpoint_key((checkpoint['color_transformer'])), strict=True)
                    print (checkpoint['color_transformer'])
                    print ('--- Finished loading Printer Color Transformation (%s) module' % (self.config['PrinterCT']))
                else:
                    print ('--- Printer Color Transformation (%s) module loaded from somewhere else' % (self.config['PrinterCT']))

            if self.use_LCT:
                self.LCT.load_state_dict(fix_checkpoint_key((checkpoint['generator'])), strict=True)
                print ('--- Finished loading Lighting Color Transformation (%s) module' % (self.config['LightingCT']))

        except Exception as e:
            print (e)
            raise IOError('Warning ---mostly this is because the model was trained using different names for its submodules. Please double check if the right model is used.')


def fix_checkpoint_key(checkpoint):
    new_dict = {}
    for k, v in checkpoint.items():
        # TODO: a better approach:
        new_dict[k.replace("module.", "")] = v
    return new_dict
