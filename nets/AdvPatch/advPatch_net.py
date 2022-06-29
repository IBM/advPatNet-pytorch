"""
Training code for Adversarial patch training

"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed

#from nets.PatchTransformer.patch_transformer_net import PatchTransformerNet
from nets.PatchTransformer.patchTransformer_network import PatchTransformerNetwork
from nets.PatchTransformer.patchTransformer_model_builder import build_patchTransformer_from_checkpoint
from nets.EOTTransformer.EOT_transformer import EOTTransformer
from detector.build_object_detector import build_object_detector
from nets.AdvPatch.advPatch import create_advPatch_model
from nets.AdvPatch.advPatch_util import get_max_detection_score, get_totalVariation, paste_patch_to_frame


class AdvPatchNet(nn.Module):
    def __init__(self, config):
        super(AdvPatchNet, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

        #self.total_variation = TotalVariation()

        # patch transformer (geometry + printer + illumination + ...)
#        self.patch_transformer = PatchTransformerNet(config)
        pt_model = PatchTransformerNetwork(config)
        pt_model, _, _, _ ,_ = build_patchTransformer_from_checkpoint(pt_model, config['patch_transformer_path'])
        self.patch_transformer = pt_model

        # Exepected of Transformation (can be used as data augmentation as well)
        self.EOTTransformer = EOTTransformer() if config['use_EOT'] else None

        # adversarial patch
        self.adv_patch_model = create_advPatch_model(config)
        self.no_patch_resize = True if tuple(config['template_shape']) == self.adv_patch_model.patch_size else False
        self.resize_shape = config['template_shape'][:2]

        # victim detector
        self.detector = build_object_detector(config)

        self.obj_loss_type = config['obj_loss_type']

        self.target_obj_id = self.detector.target_object_id
        self.train_conf_thresh = config['train_conf_thresh']
        self.train_nms_thresh = config['train_nms_thresh']
        #self.val_conf_thresh = config['val_conf_thresh']
        #self.val_nms_thresh = config['val_nms_thresh']

        if self.use_cuda and not torch.distributed.is_initialized():
            self.adv_patch_model.cuda(self.device_ids[0])
            self.patch_transformer.cuda(self.device_ids[0])
            self.detector.cuda(self.device_ids[0])

    def set_train_mode(self):

        self.train()

        # do not expect changes of these models
        self.patch_transformer.eval()
        self.detector.eval()

    def set_eval_mode(self):
        self.eval()

    '''
    @staticmethod
    def paste_patch_to_frame(patch, patch_bb, img, img_bb):
        n, c, _, _ = patch.shape
        # create tensor
        img_h, img_w = img.shape[2:]
        x = torch.cuda.FloatTensor(n, c, img_h, img_w).fill_(0)
        for i, bbox in enumerate(patch_bb):
            pb, pl, ph, pw = bbox
            ib, il, ih, iw = img_bb[i]
            resized_tmpl = F.interpolate(patch[i,:,pb:pb+ph, pl:pl+pw].unsqueeze(0), size=(ih, iw), mode='bilinear', align_corners=False)
            x[i, :, ib:ib+ih, il:il+iw] = resized_tmpl.squeeze()

        return x
    '''

    #@torchsnooper.snoop()
    def forward(self, imgs, big_patch_mask, big_patch_bb, person_bb, padded_person_crop, small_patch_mask, small_patch_bb, patch_img, scale_factor):
        '''
        param imgs: images of original size
        param patch_mask: adversarial patch mask for 'imgs'
        param patch_bb:   bounding boxes of adversarial patches for 'imgs'
        param person_bb:   bounding boxes of the person for 'imgs'
        param small_patch_crop: adversarial patches cropped out from 'img's and resized to [224 224]
        param padded_person_crop: the person wearing advPatch cropped out from 'img's and resized to [256 256]
        param smal_patch_mask: advPatch mask for 'padded_person_crop'
        param small_patch_bb: bounding boxes of advPatch for 'padded_person_crop'
        '''

        # resize to print size if needed
        adv_patch = self.adv_patch_model() if self.no_patch_resize else   \
            F.interpolate(self.adv_patch_model().unsqueeze(0), self.resize_shape, mode='bilinear').squeeze()
        template_patch = adv_patch.unsqueeze(0).repeat(imgs.shape[0], 1, 1, 1)

        # transform the patch
        #output = self.patch_transformer(patch_img, small_patch_bb, small_patch_mask, template_patch, padded_person_crop, scale_factor)
        #transformed_patch = output[0] if type(output) == tuple else output
        template_resize_scale = 0.6 / (1.0 + torch.exp(-2.0 * scale_factor.view(-1, 1)))
        template_resize_scale = torch.clamp(template_resize_scale, max=1.0)
        output = self.patch_transformer.transform_patch(patch_img, small_patch_bb, small_patch_mask, template_patch, \
                 padded_person_crop, template_resize_scale)
        transformed_patch = output[-1]

        # paste the patch to the frame
        adv_batch_t = paste_patch_to_frame(transformed_patch, small_patch_bb, imgs, big_patch_bb)
        #half_big_patch_mask = self.mask_half_patch(big_patch_mask, big_patch_bb)
        adv_batch_t = adv_batch_t * big_patch_mask + (1 - big_patch_mask) * imgs
#        adv_batch_t = adv_batch_t * half_big_patch_mask + (1 - half_big_patch_mask) * imgs

        if self.training and self.EOTTransformer is not None:
            adv_batch_t = self.EOTTransformer(adv_batch_t)

        if self.training:
            detection_results = self.detector.detect_train(adv_batch_t, conf_thresh=self.train_conf_thresh, nms_thresh=self.train_nms_thresh)
            max_detection_score = get_max_detection_score(detection_results, person_bb, self.target_obj_id, min_detection_score=self.train_conf_thresh, loss_type=self.obj_loss_type)
        else:
            detection_results = self.detector.detect(adv_batch_t, conf_thresh=self.train_conf_thresh, nms_thresh=self.train_nms_thresh)
            max_detection_score = get_max_detection_score(detection_results, person_bb, self.target_obj_id, min_detection_score=self.train_conf_thresh)

         # focal point loss (not working)
#        if self.training:
#           max_detection_score = -(((max_detection_score -self.train_conf_thresh) / (1.0 - self.train_conf_thresh)) ** 2) * torch.log((1.0-max_detection_score))

        '''
        import torchvision.transforms as transforms
        from utils.utils import visualize_detections
        import os
        import numpy as np
        for i in range(imgs.shape[0]):
               print (detection_results[i])
            #if max_detection_score[i] > 0.3:
               train_img = transforms.ToPILImage()(adv_batch_t[i].detach().cpu())
               #train_img.save(os.path.join('tmp', name[i]+'.jpg'))
               visualize_detections(train_img, detection_results[i], \
                                os.path.join('tmp', '%d.jpg' % (int(100*np.random.rand()))), self.detector.class_names)
        '''

        tv =  get_totalVariation(self.adv_patch_model.adv_patch)
        #print ([ len(item) for item in detection_results])
        return adv_batch_t, max_detection_score, tv

    def mask_half_patch(self, mask_img, patch_bb):
        for i, bbox in enumerate(patch_bb):
            pb, pl, ph, pw = bbox
            yc = pb + ph // 2
            mask_img[i, :, yc:, :] = 0

        return mask_img

    '''
    def get_max_detection_score(self, output, obj_bbox, target_obj_id=0, min_detection_score=0.3):
        # output a list of (x1,y1,x2,y2, object_conf, class_pred)
        # obj_bbox: a list of (x1, y1, x2, y2)
        assert len(output) == obj_bbox.shape[0]

        # minimum prob. is set to 0.3
        max_prob = torch.zeros((obj_bbox.shape[0], 1)).cuda()
  #      print ('max_prob_0', max_prob)
        for k in range(len(output)):
            detection = output[k]
            if isinstance(detection, list) and detection[0] is None:
                continue

            person_detection = detection[detection[:, -1] == target_obj_id]
            if person_detection.shape[0] == 0:
                continue

            bbox = obj_bbox[k]
            xc = (person_detection[:, 0] + person_detection[:,2]) / 2.0
            yc = (person_detection[:, 1] + person_detection[:,3]) / 2.0

            x_inside = (xc > bbox[0]) & (bbox[2] > xc)
            y_inside = (yc > bbox[1]) & (bbox[3] > yc)

            xy_inside = x_inside & y_inside
            #assert any(xy_inside>0), (xy_inside, xc, yc, detection, bbox, x_inside, y_inside, xy_inside)
            if any(xy_inside > 0):
                max_prob[k] = torch.max(person_detection[xy_inside, 4])
        max_prob = torch.clamp(max_prob, min=min_detection_score)
        return max_prob

    # total variation of the patch
    def get_totalVariation(self, adv_patch):
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)
'''
