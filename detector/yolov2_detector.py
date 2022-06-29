import os

import torch
import torch.nn.functional as F

from detector.yolov2.darknet import Darknet
from detector.yolov2.utils import do_detect_1
from detector.object_detector import ObjectDetector
from detector.yolo_util import nms, xywh2xyxy, wrap_detection_results


class YOLOV2_Detector(ObjectDetector):
    def __init__(self, model_name,  cfg_path, model_path, class_names, input_size=(-1, -1), test_size=(-1, -1), target_object_id=-1):
        # load darknet
#        model, class_names = self._load_model(cfg_path, model_path)
        super().__init__(model_name, cfg_path, model_path, class_names, input_size, test_size, target_object_id)
        
        # skip background i.e. 0
        self.class_names = [name for k, name in enumerate(class_names) if k > 0]

    def load_model(self, cfg_path, model_path, class_names=None):
        darknet_model = Darknet(cfg_path)
        darknet_model.load_weights(model_path)
        darknet_model = darknet_model.eval()

        return darknet_model

    '''
    def detect(self, images, conf_thresh=0.2, nms_thresh=0.0):
        scaled_images = F.interpolate(images, size=self.test_size, mode='bilinear', align_corners=False)
        outputs = self.model(scaled_images)
        boxes = get_region_boxes(outputs, conf_thresh, self.model.num_classes, self.model.anchors, self.model.num_anchors)
        if nms_thresh > 0:
            boxes = [nms(box, nms_thresh) for box in boxes]

        # convert it to coordinates with regards to the orginal sizes
        outputs = []
        height, width = self.input_size
        for b in boxes:
            if len(b) == 0:
                #outputs += [torch.FloatTensor([[0.0, 0.0, 0.0, 0.0, 0.0, -1]]).cuda()]
                outputs += [None]
            else:
                t = torch.stack(b).cuda()
                t_new = t.clone()
                t_new[:,0] = (t[:,0] - t[:,2] / 2.0) * width
                t_new[:,1] = (t[:,1] - t[:,3] / 2.0) * height
                t_new[:,2] = (t[:,0] + t[:,2] / 2.0) * width
                t_new[:,3] = (t[:,1] + t[:,3] / 2.0) * height
        #        print ('t_new', t_new)
                # skip classification score
                outputs += [t_new[:, [0,1,2,3,4,6]]]
        return outputs
    '''

    def detect(self, images, conf_thresh=0.2, nms_thresh=0.0):
        _, h, w, _ = images.shape
        if self.test_size[0] == w and self.test_size[1] == h:
            scaled_images = images
        else:
            scaled_images = F.interpolate(images, size=self.test_size, mode='bilinear', align_corners=False)

        outputs = self.model(scaled_images)
        # print (outputs.shape)
        outputs = post_processing(outputs, self.model.num_classes, self.model.anchors, self.model.num_anchors, self.test_size)
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        outputs[..., :4] = xywh2xyxy(outputs[..., :4])

        outputs = nms(outputs, conf_thres=conf_thresh, nms_thres=nms_thresh)
        results = wrap_detection_results(outputs, self.test_size[0], self.input_size)

        return results

    # The 'detect' method is implemented differently from the one in the original implmentation of yolov2.
    # 'detector_detect' attempts to keep the same implementation as the original one.
    def detector_detect(self, img, conf_thresh, nms_thresh):
        batch, h, w, _ = img.shape
        if self.test_size[0] == w and self.test_size[1] == h:
            scaled_img = img
        else:
            scaled_img = F.interpolate(img, size=self.test_size, mode='bilinear', align_corners=False)

        outputs = do_detect_1(self.model, scaled_img, conf_thresh, nms_thresh)
        if not outputs:
            return [[None]] * batch

        for item in outputs:
            item[:4] = xywh2xyxy(item[:4])
            item[:4] *= self.test_size[0]
        outputs = [torch.stack(outputs, dim=0)]
        results = wrap_detection_results(outputs, self.test_size[0], self.input_size)
        # resize
        return results

def post_processing(output, num_classes, anchors, num_anchors, test_size):
    # anchor_step = len(anchors)/num_anchors
    FloatTensor = torch.cuda.FloatTensor if output.is_cuda else torch.FloatTensor

    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)

    batch, _,  h, w = output.shape
    assert (output.size(1) == (5 + num_classes) * num_anchors)

    #print(output.size())
    output = output.view(batch * num_anchors, 5 + num_classes, h * w)
    #print(output.size())
    output = output.transpose(1, 2).contiguous()
    #print(output.size())
    output = output.view(batch * num_anchors * h * w, 5 + num_classes)
    #print(output.size())

    # Get outputs
    x = torch.sigmoid(output[..., 0])  # Center x
    y = torch.sigmoid(output[..., 1])  # Center y
    pred_conf = torch.sigmoid(output[..., 4])  # Conf
    pred_cls = torch.sigmoid(output[..., 5:])  # Cls pred.

    # print(output.size())
    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).cuda()
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).cuda()
    xs = x + grid_x
    ys = y + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).cuda()
    ws = torch.exp(output[..., 2]) * anchor_w
    hs = torch.exp(output[..., 3]) * anchor_h

    iw, ih = test_size
    output = torch.cat(
        (
            xs.view(batch, -1, 1) / w * iw,
            ys.view(batch, -1, 1) / h * ih,
            ws.view(batch, -1, 1) / w * iw,
            hs.view(batch, -1, 1) / h * ih,
            pred_conf.view(batch, -1, 1),
            pred_cls.view(batch, -1, num_classes),
        ),
        dim=2,
    )
    #print (output.shape)
    return output
