import torch
import torch.nn.functional as F
from .object_detector import ObjectDetector
from .faster_rcnn.lib.model.utils.config import cfg, cfg_from_file
from .faster_rcnn.lib.model.faster_rcnn.vgg16 import vgg16
from .faster_rcnn.lib.model.faster_rcnn.resnet import resnet
from .faster_rcnn.lib.model.roi_layers import nms
from .faster_rcnn.lib.model.rpn.bbox_transform import bbox_transform_inv
from .faster_rcnn.lib.model.rpn.bbox_transform import clip_boxes
import numpy as np

class Faster_RCNN_Detector(ObjectDetector):
    def __init__(self, model_name, cfg_path, model_path, class_names,  input_size=(-1, -1), test_size=(-1, -1), target_object_id=-1):
        # load SSD
        super().__init__(model_name, cfg_path, model_path, class_names, input_size, test_size, target_object_id)

        self.mean = cfg.PIXEL_MEANS[0][0].tolist()
#        self.test_size = cfg.TEST.SCALES
        self.cfg = cfg

    def load_model(self, cfg_path, model_path, class_names):
        cfg_from_file(cfg_path)
        # fixed
        cfg.POOLING_MODE = 'align'
        cfg.class_agnostic = False

        obj_classes = np.asarray(class_names)
        # initilize the network here.
        if cfg.EXP_DIR == 'vgg16':
            fasterRCNN = vgg16(obj_classes, pretrained=False, class_agnostic=cfg.class_agnostic, anchor_scales=cfg.ANCHOR_SCALES, anchor_ratios=cfg.ANCHOR_RATIOS)
        elif cfg.EXP_DIR == 'res50':
            fasterRCNN = resnet(obj_classes, 50, pretrained=False, class_agnostic=cfg.class_agnostic, anchor_scales=cfg.ANCHOR_SCALES, anchor_ratios=cfg.ANCHOR_RATIOS)
        elif cfg.EXP_DIR == 'res101':
            fasterRCNN = resnet(obj_classes, 101, pretrained=False, class_agnostic=cfg.class_agnostic, anchor_scales=cfg.ANCHOR_SCALES, anchor_ratios=cfg.ANCHOR_RATIOS)
        elif cfg.EXP_DIR == 'res152':
            fasterRCNN = resnet(obj_classes, 152, pretrained=False, class_agnostic=cfg.class_agnostic, anchor_scales=cfg.ANCHOR_SCALES, anchor_ratios=cfg.ANCHOR_RATIOS)
        else:
            raise NameError("network %s is not defined" % (cfg.EXP_DIR) )

        fasterRCNN.create_architecture()

        checkpoint = torch.load(model_path, map_location=(lambda storage, loc: storage))
        fasterRCNN.load_state_dict(checkpoint['model'])

        fasterRCNN.eval()

        return fasterRCNN

    def detect(self, images, conf_thresh=0.2, nms_thresh=0.0):
        
        input_imgs, im_scale = self.preprocess(images)

        batch_size = input_imgs.size(0)
        with torch.no_grad():
            im_info = np.array([[input_imgs.shape[2], input_imgs.shape[3], im_scale[0]]], dtype=np.float32)
            im_info = np.repeat(im_info, batch_size, axis=0)
            im_info = torch.from_numpy(im_info).cuda(device=images.device)
            num_boxes = torch.zeros(batch_size).cuda(device=images.device)
            gt_boxes = torch.zeros(batch_size, 1, 5).cuda(device=images.device)

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = self.model(input_imgs, im_info, gt_boxes, num_boxes)

        scores = cls_prob
        boxes = rois[:, :, 1:5]
        results = self.post_process(im_info, bbox_pred, scores, boxes, im_scale, conf_thresh, nms_thresh)
        return results

    def preprocess(self, images):
        batch_size, _, h, w = images.shape
        im_size_min = min(h,w)
        im_size_max = max(h,w)
        im_scale = float(self.test_size[0]) / im_size_min
        if np.round(im_scale * im_size_max) > self.cfg.TEST.MAX_SIZE:
            im_scale = float(self.cfg.TEST.MAX_SIZE) / float(im_size_max)

        # scale the image
        test_size = (round(h*im_scale), round(w*im_scale))
        scaled_imgs = F.interpolate(images, size=test_size,  mode='bilinear', align_corners=False)
        scaled_imgs *= 255.0
 
        '''
        import torchvision.transforms as transforms
        from utils.utils import visualize_detections
        import os
        for i in range(scaled_imgs.shape[0]):
               train_img = transforms.ToPILImage()(images[i].detach().cpu())
               train_img.save(os.path.join('tmp', '%d.jpg' % (int(100*np.random.rand()))))
        '''
        # normalize the image
        mean = torch.tensor(self.mean).view(1, len(self.mean), 1, 1).cuda(device=images.device)
        input_imgs = scaled_imgs - mean

        return input_imgs, (im_scale, im_scale)

    def do_nms(self, scores, pred_boxes, conf_thresh, nms_thresh):
        results = list()
        for j in range(1, len(self.class_names)):
            inds = torch.nonzero(scores[:, j] > conf_thresh).view(-1)
            #print (inds)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.cfg.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                keep = nms(cls_boxes[order, :], cls_scores[order], nms_thresh)
                cls_dets = cls_dets[keep.view(-1).long()]
                label_ids = j * torch.ones(cls_dets.size(0), 1).cuda(device=cls_dets.device)
                #print (cls_dets.shape, label_ids.shape)
                results.append(torch.cat((cls_dets, label_ids), 1))
        #print (results) 
        return torch.cat(results, dim = 0) if len(results) > 0 else [None]

    def post_process(self, im_info, bbox_pred, scores, boxes, im_scale, conf_thresh, nms_thresh):
        batch_size = bbox_pred.size(0)
        if self.cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred
            if self.cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if self.cfg.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(batch_size, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(batch_size, -1, 4 * len(self.class_names))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, batch_size)
            pred_boxes = clip_boxes(pred_boxes, im_info, batch_size)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scale[0]
    #    scores = scores.squeeze()
    #    pred_boxes = pred_boxes.squeeze()
    
        results = [self.do_nms(scores[k], pred_boxes[k], conf_thresh, nms_thresh) for k in range(batch_size)]

        return results
