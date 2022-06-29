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
from nets.AdvPatch.collaborative_advPatch import create_collaborative_advPatch_model
from nets.AdvPatch.hybrid_advPatch import create_hybrid_advPatch_model
from detector.build_object_detector import build_object_detector
from nets.AdvPatch.advPatch_util import get_max_detection_score, get_totalVariation, paste_patch_to_frame
from losses.smooth_l1_loss import SmoothL1Loss
from losses.mask_losses import SmoothL1MaskLoss

class CollaborativeAdvPatchNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

        # patch transformer (geometry + printer + illumination + ...)
#        self.patch_transformer = PatchTransformerNetwork(config)
#        self.patch_transformer.load_from_file(config['patch_transformer_path'])
        pt_model = PatchTransformerNetwork(config)
        pt_model, _, _, _ ,_ = build_patchTransformer_from_checkpoint(pt_model, config['patch_transformer_path'])
        self.patch_transformer = pt_model

        # Exepected of Transformation (can be used as data augmentation as well)
        self.EOTTransformer = EOTTransformer() if config['use_EOT'] else None

        # adversarial patch
        #self.adv_patch_model = create_collaborative_advPatch_model(config)
        self.adv_patch_model = create_hybrid_advPatch_model(config)
        self.no_patch_resize = True if tuple(config['template_shape']) == self.adv_patch_model.patch_size else False
        self.resize_shape = config['template_shape'][:2]
        self.collaborative_learning = self.adv_patch_model.collaborative_learning

        # victim detector
        self.detector = build_object_detector(config)

        self.obj_loss_type = config['obj_loss_type']

        self.kd_type = config['kd_type']
        if self.kd_type == 'margin':
            self.kd_loss = nn.TripletMarginLoss(margin=0.1, p=config['kd_norm'], reduction='none')
        elif self.kd_type == 'MSE':
            self.kd_loss = nn.MSELoss(reduction='none')
        elif self.kd_type == 'L1':
            self.kd_loss = nn.L1Loss(reduction='none')
        elif self.kd_type == 'SmoothL1':
            self.kd_loss = SmoothL1Loss(reduction='none', beta=0.5)
        elif self.kd_type == 'mutual':
            self.kd_loss = nn.KLDivLoss(reduction='none')
        elif self.kd_type == 'one':
            self.kd_loss = nn.KLDivLoss(reduction='none')
        else:
            raise ValueError('kd type %s is not supported' % (self.kd_type))

        self.sim_loss = SmoothL1MaskLoss(beta=0.5)

        self.half_patches = config['half_patches']

        self.patch_size_median = config['patch_size_median']

        self.target_obj_id = self.detector.target_object_id
        self.train_conf_thresh = config['train_conf_thresh']
        self.train_nms_thresh = config['train_nms_thresh']
#        self.val_conf_thresh = config['val_conf_thresh']
#        self.val_nms_thresh = config['val_nms_thresh']

        print ('------- Collaborative Learning --------')
        if self.half_patches:
            print ('half patches: Yes')
        else:
            print ('half patches: No')
        print ('KD type: %s' % (self.kd_type))
        print ('detection confidence thresh: %4.2f' % (self.train_conf_thresh))
        print ('detection nms thresh: %4.2f' % (self.train_nms_thresh))
         

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

    def forward(self, imgs, big_patch_mask, big_patch_bb, person_bb, padded_person_crop, small_patch_mask,
                small_patch_bb, patch_img, patch_size_info):
        '''
        param imgs: images of original size
        param patch_mask: adversarial patch mask for 'imgs'
        param patch_bb:   bounding boxes of adversarial patches for 'imgs'
        param person_bb:   bounding boxes of the person for 'imgs'
        param small_patch_crop: adversarial patches cropped out from 'img's and resized to [224 224]
        param padded_person_crop: the person wearing advPatch cropped out from 'imgs' and resized/padded to [256 256]
        param smal_patch_mask: advPatch mask for 'padded_person_crop'
        param small_patch_bb: bounding boxes of advPatch for 'padded_person_crop'
        param patch_img: patches cropped out from 'imgs'
        param patch_size_info: small or large patch
        '''

        if self.half_patches:
            return self.forward_half(imgs, big_patch_mask, big_patch_bb, person_bb, padded_person_crop, small_patch_mask,
                small_patch_bb, patch_img, patch_size_info)

        if self.kd_type == 'mutual' or self.kd_type == 'one':
            return self.forward_kl(imgs, big_patch_mask, big_patch_bb, person_bb, padded_person_crop, small_patch_mask,
                small_patch_bb, patch_img, patch_size_info)

        #if self.kd_type == 'margin':
        return self.forward_margin(imgs, big_patch_mask, big_patch_bb, person_bb, padded_person_crop, small_patch_mask,
            small_patch_bb, patch_img, patch_size_info)

    def forward_half(self, imgs, big_patch_mask, big_patch_bb, person_bb, padded_person_crop, small_patch_mask,
                           small_patch_bb, patch_img, patch_size_info):
            '''
            param imgs: images of original size
            param patch_mask: adversarial patch mask for 'imgs'
            param patch_bb:   bounding boxes of adversarial patches for 'imgs'
            param person_bb:   bounding boxes of the person for 'imgs'
            param small_patch_crop: adversarial patches cropped out from 'img's and resized to [224 224]
            param padded_person_crop: the person wearing advPatch cropped out from 'imgs' and resized/padded to [256 256]
            param smal_patch_mask: advPatch mask for 'padded_person_crop'
            param small_patch_bb: bounding boxes of advPatch for 'padded_person_crop'
            param patch_img: patches cropped out from 'imgs'
            param patch_size_info: small or large patch
            '''

            if not self.training:
                return self.inference(imgs, big_patch_mask, big_patch_bb, person_bb,
                                      padded_person_crop, small_patch_mask,
                                      small_patch_bb, patch_img, patch_size_info)

            adv_patch, adv_patch_near, adv_patch_far = self.adv_patch_model()

            adv_images, max_prob = self.forward_single(adv_patch,
                                                       imgs,
                                                       big_patch_mask,
                                                       big_patch_bb,
                                                       person_bb,
                                                       padded_person_crop,
                                                       small_patch_mask,
                                                       small_patch_bb,
                                                       patch_img,
                                                       patch_size_info)

            near_half_mask = self.mask_half_patch(big_patch_mask, big_patch_bb, top=True)
            #near_half_mask = big_patch_mask
            _, max_prob_near = self.get_max_prob(adv_images, imgs, near_half_mask, person_bb)
            far_half_mask = self.mask_half_patch(big_patch_mask, big_patch_bb, top=False)
            #far_half_mask = big_patch_mask
            _, max_prob_far = self.get_max_prob(adv_images, imgs, far_half_mask, person_bb)

            near_index = (patch_size_info > self.patch_size_median).nonzero()
            far_index = (patch_size_info <= self.patch_size_median).nonzero()
            near_weight = torch.zeros(patch_size_info.shape).cuda(patch_size_info.device)
            far_weight = torch.zeros(patch_size_info.shape).cuda(patch_size_info.device)
            near_weight[near_index] = 1.0
            far_weight[far_index] = 1.0

            if self.kd_type == 'margin':
                kd_loss = near_weight * self.kd_loss(max_prob, max_prob_near, max_prob_far)
                kd_loss += far_weight * self.kd_loss(max_prob, max_prob_far, max_prob_near)
            else:
                kd_loss = near_weight * self.kd_loss(max_prob, max_prob_near)
                kd_loss += far_weight * self.kd_loss(max_prob, max_prob_far)

            adv_loss = max_prob.squeeze()
            if self.collaborative_learning:
                adv_loss += (near_weight * max_prob_near.squeeze())
                adv_loss += (far_weight * max_prob_far.squeeze())

            tv_loss = get_totalVariation(adv_patch)

            return adv_images, adv_loss, tv_loss, kd_loss

    def mask_half_patch(self, mask_img, patch_bb, top=True):
        new_mask_img = mask_img.clone()
        for i, bbox in enumerate(patch_bb):
            pb, pl, ph, pw = bbox
            yc = pb + ph // 2
            if top:
                new_mask_img[i, :, yc:, :] = 0
            else:
                new_mask_img[i, :, :yc, :] = 0

        return new_mask_img

    def forward_margin(self, imgs, big_patch_mask, big_patch_bb, person_bb, padded_person_crop, small_patch_mask,
                small_patch_bb, patch_img, patch_size_info):
        '''
        param imgs: images of original size
        param patch_mask: adversarial patch mask for 'imgs'
        param patch_bb:   bounding boxes of adversarial patches for 'imgs'
        param person_bb:   bounding boxes of the person for 'imgs'
        param small_patch_crop: adversarial patches cropped out from 'img's and resized to [224 224]
        param padded_person_crop: the person wearing advPatch cropped out from 'imgs' and resized/padded to [256 256]
        param smal_patch_mask: advPatch mask for 'padded_person_crop'
        param small_patch_bb: bounding boxes of advPatch for 'padded_person_crop'
        param patch_img: patches cropped out from 'imgs'
        param patch_size_info: small or large patch
        '''

        if not self.training:
            return self.inference(imgs, big_patch_mask, big_patch_bb, person_bb,
                             padded_person_crop, small_patch_mask,
                             small_patch_bb, patch_img, patch_size_info)

        adv_patch, adv_patch_near, adv_patch_far = self.adv_patch_model()

        adv_images, max_prob = self.forward_single(adv_patch,
                                                imgs,
                                                big_patch_mask,
                                                big_patch_bb,
                                                person_bb,
                                                padded_person_crop,
                                                small_patch_mask,
                                                small_patch_bb,
                                                patch_img,
                                                patch_size_info)

        adv_images_near, max_prob_near = self.forward_single(adv_patch_near,
                                                imgs,
                                                big_patch_mask,
                                                big_patch_bb,
                                                person_bb,
                                                padded_person_crop,
                                                small_patch_mask,
                                                small_patch_bb,
                                                patch_img,
                                                patch_size_info) 

        adv_images_far, max_prob_far = self.forward_single(adv_patch_far,
                                                imgs,
                                                big_patch_mask,
                                                big_patch_bb,
                                                person_bb,
                                                padded_person_crop,
                                                small_patch_mask,
                                                small_patch_bb,
                                                patch_img,
                                                patch_size_info)

        kd_loss = torch.zeros(patch_size_info.shape).cuda()
        if self.adv_patch_model.collaborative_weight:  # learnable weighting
            near_weight = self.adv_patch_model.collaborative_weight(patch_size_info.unsqueeze(-1))  # nx1
            far_weight = 1.0 - near_weight
            #print(patch_size_info, near_weight, self.adv_patch_model.collaborative_weight[0].weight, self.adv_patch_model.collaborative_weight[0].bias)
            #print(self.adv_patch_model.collaborative_weight[0].weight, self.adv_patch_model.collaborative_weight[0].bias)
            if self.kd_type == 'margin':
                kd_loss += near_weight[:, 0] * self.kd_loss(max_prob, max_prob_near, max_prob_far).squeeze()
                kd_loss += far_weight[:, 0] * self.kd_loss(max_prob, max_prob_far, max_prob_near).squeeze()
            else:
                kd_loss +=  near_weight[:,0] * self.kd_loss(max_prob, max_prob_near).squeeze()
                kd_loss += far_weight[:,0] * self.kd_loss(max_prob, max_prob_far).squeeze()

            adv_loss = max_prob.squeeze()
            if self.collaborative_learning:
                adv_loss += (near_weight[0,:] * max_prob_near.squeeze())
                adv_loss += (far_weight[0,:] * max_prob_far.squeeze())

        else:  # heuristic weighting
            # index
            near_index = (patch_size_info > self.patch_size_median).nonzero()
            far_index = (patch_size_info <= self.patch_size_median).nonzero()
            
            # sample size weight
#            near_samples = float(torch.numel(near_index))
#            far_samples = float(torch.numel(far_index))
#            total_samples = near_samples + far_samples

            near_weight = torch.zeros(patch_size_info.shape).cuda(patch_size_info.device)
            far_weight = torch.zeros(patch_size_info.shape).cuda(patch_size_info.device)
            near_weight[near_index] = 1.0
            far_weight[far_index] = 1.0
            #near_weight[near_index] = total_samples / (2.0 * near_samples) if near_samples > 0 else 0.0
            #far_weight[far_index] = total_samples / (2.0 * far_samples) if far_samples > 0 else 0.0

            if self.kd_type == 'margin':
                kd_loss +=  near_weight * (self.kd_loss(max_prob, max_prob_near, max_prob_far)).squeeze()
                kd_loss += far_weight * (self.kd_loss(max_prob, max_prob_far, max_prob_near)).squeeze()
            else:
                kd_loss +=  near_weight * (self.kd_loss(max_prob, max_prob_near)).squeeze()
                kd_loss += far_weight * (self.kd_loss(max_prob, max_prob_far)).squeeze()
           
            #sim_loss = near_weight * (self.sim_loss(adv_images, adv_images_near, big_patch_bb)).squeeze()
            sim_loss = far_weight * (self.sim_loss(adv_images, adv_images_far, big_patch_bb)).squeeze()

            kd_loss += sim_loss

            adv_loss = max_prob.squeeze()
            #adv_loss = torch.zeros(patch_size_info.shape).cuda()
            if self.collaborative_learning:
                adv_loss += (near_weight * max_prob_near.squeeze())
                adv_loss += (far_weight * max_prob_far.squeeze())

        tv_loss = get_totalVariation(adv_patch)
        if self.collaborative_learning:
            tv_loss += get_totalVariation(adv_patch_near)
            tv_loss += get_totalVariation(adv_patch_far)
            tv_loss /= 3.0

        return adv_images, adv_loss, tv_loss, kd_loss

    def forward_kl(self, imgs, big_patch_mask, big_patch_bb, person_bb, padded_person_crop, small_patch_mask,
                small_patch_bb, patch_img, scale_factor):
        """
        param imgs: images of original size
        param patch_mask: adversarial patch mask for 'imgs'
        param patch_bb:   bounding boxes of adversarial patches for 'imgs'
        param person_bb:   bounding boxes of the person for 'imgs'
        param small_patch_crop: adversarial patches cropped out from 'img's and resized to [224 224]
        param padded_person_crop: the person wearing advPatch cropped out from 'imgs' and resized/padded to [256 256]
        param smal_patch_mask: advPatch mask for 'padded_person_crop'
        param small_patch_bb: bounding boxes of advPatch for 'padded_person_crop'
        param patch_img: patches cropped out from 'imgs'
        param patch_size_info: small or large patch
        """
        adv_patch = self.adv_patch_model.adv_patch
        adv_patch_near = self.adv_patch_model.adv_patch_near
        adv_patch_far = self.adv_patch_model.adv_patch_far

        adv_images, max_prob = self.forward_single(adv_patch,
                                                   imgs,
                                                   big_patch_mask,
                                                   big_patch_bb,
                                                   person_bb,
                                                   padded_person_crop,
                                                   small_patch_mask,
                                                   small_patch_bb,
                                                   patch_img,
                                                   scale_factor)

        _, max_prob_near = self.forward_single(adv_patch_near,
                                               imgs,
                                               big_patch_mask,
                                               big_patch_bb,
                                               person_bb,
                                               padded_person_crop,
                                               small_patch_mask,
                                               small_patch_bb,
                                               patch_img,
                                               scale_factor)

        _, max_prob_far = self.forward_single(adv_patch_far,
                                              imgs,
                                              big_patch_mask,
                                              big_patch_bb,
                                              person_bb,
                                              padded_person_crop,
                                              small_patch_mask,
                                              small_patch_bb,
                                              patch_img,
                                              scale_factor)

        # ensemble all models
        ensemble_max_prob = (max_prob_near + max_prob_far + max_prob ) / 3.0
        if not self.training:
            return adv_images, ensemble_max_prob, get_totalVariation(adv_patch)

        if self.kd_type == 'mutual':
            adv_loss = max_prob
            adv_loss += max_prob_near
            adv_loss += max_prob_far

            kd_loss = self.kd_loss(max_prob, max_prob_near)
            kd_loss += self.kd_loss(max_prob, max_prob_far)
            kd_loss += self.kd_loss(max_prob_near, max_prob)
            kd_loss += self.kd_loss(max_prob_near, max_prob_far)
            kd_loss += self.kd_loss(max_prob_far, max_prob)
            kd_loss += self.kd_loss(max_prob_far, max_prob_near)
            kd_loss *= 0.5  # normalized by (1/(K-1) according to the paper)
        else:
            adv_loss = ensemble_max_prob
            adv_loss += max_prob
            adv_loss += max_prob_near
            adv_loss += max_prob_far

            kd_loss = self.kd_loss(max_prob, ensemble_max_prob)
            kd_loss += self.kd_loss(max_prob_near, ensemble_max_prob)
            kd_loss += self.kd_loss(max_prob_far, ensemble_max_prob)

        tv_loss = get_totalVariation(adv_patch)
        tv_loss += get_totalVariation(adv_patch_near)
        tv_loss += get_totalVariation(adv_patch_far)
        tv_loss /= 3.0

        return adv_images, adv_loss, tv_loss, kd_loss

    @torch.no_grad()
    def inference(self, imgs, big_patch_mask, big_patch_bb, person_bb, padded_person_crop, small_patch_mask,
                small_patch_bb, patch_img, scale_factor):

        adv_patch = self.adv_patch_model()
        adv_images, max_prob = self.forward_single(adv_patch,
                                                         imgs,
                                                         big_patch_mask,
                                                         big_patch_bb,
                                                         person_bb,
                                                         padded_person_crop,
                                                         small_patch_mask,
                                                         small_patch_bb,
                                                         patch_img,
                                                         scale_factor)
        tv = get_totalVariation(adv_patch)
        return adv_images, max_prob, tv

    def forward_single(self, adv_patch, imgs, big_patch_mask, big_patch_bb, person_bb, padded_person_crop, small_patch_mask,
                small_patch_bb, patch_img, scale_factor):
        # While it doesn't seem neccessary to have a large patch size (416, 416), it yields better performance.
        # resize to print size if needed
        adv_patch_resize = adv_patch if self.no_patch_resize else \
            F.interpolate(adv_patch.unsqueeze(0), self.resize_shape, mode='bilinear').squeeze()
        template_patch = adv_patch_resize.unsqueeze(0).repeat(imgs.shape[0], 1, 1, 1)

        # transform the patch
        template_resize_scale = 0.6 / (1.0 + torch.exp(-2.0 * scale_factor.view(-1, 1)))
        template_resize_scale = torch.clamp(template_resize_scale, max=1.0)

        output = self.patch_transformer.transform_patch(patch_img, small_patch_bb, small_patch_mask, template_patch, padded_person_crop, scale_factors=template_resize_scale)
        transformed_patch = output[-1]

        # paste the patch to the frame
        adv_batch_t = paste_patch_to_frame(transformed_patch, small_patch_bb, imgs, big_patch_bb)

        adv_batch_t = adv_batch_t * big_patch_mask + (1 - big_patch_mask) * imgs

        if self.training and self.EOTTransformer is not None:
            adv_batch_t = self.EOTTransformer(adv_batch_t)

        if self.training:
            detection_results = self.detector.detect_train(adv_batch_t, conf_thresh=self.train_conf_thresh,
                                                           nms_thresh=self.train_nms_thresh)
        else:
            detection_results = self.detector.detect(adv_batch_t, conf_thresh=self.train_conf_thresh,
                                                     nms_thresh=self.train_nms_thresh)
        max_prob = get_max_detection_score(detection_results, person_bb, self.target_obj_id, min_detection_score=self.train_conf_thresh, loss_type=self.obj_loss_type)
        return adv_batch_t, max_prob

    def get_max_prob(self, adv_batch_t, imgs, half_mask, person_bb):
        adv_batch_t_new = adv_batch_t * half_mask + (1 - half_mask) * imgs

        if self.training and self.EOTTransformer is not None:
            adv_batch_t_new = self.EOTTransformer(adv_batch_t_new)

        if self.training:
            detection_results = self.detector.detect_train(adv_batch_t_new, conf_thresh=self.train_conf_thresh,
                                                           nms_thresh=self.train_nms_thresh)
        else:
            detection_results = self.detector.detect(adv_batch_t_new, conf_thresh=self.train_conf_thresh,
                                                     nms_thresh=self.train_nms_thresh)
        max_prob = get_max_detection_score(detection_results, person_bb, self.target_obj_id, min_detection_score=self.train_conf_thresh, loss_type=self.obj_loss_type)
        return adv_batch_t, max_prob
