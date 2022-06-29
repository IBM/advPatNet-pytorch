import os

import torch.nn.functional as F
import torch

from detector.yolov3.models import Darknet
from detector.yolov3.utils.utils import non_max_suppression
from detector.object_detector import ObjectDetector
from detector.yolo_util import nms, wrap_detection_results


class YOLOV3_Detector(ObjectDetector):
    def __init__(self, model_name, cfg_path, model_path, class_names, input_size=(-1, -1), test_size=(-1, -1), target_object_id=-1):
        # load darknet
        super().__init__(model_name, cfg_path, model_path, class_names, input_size, test_size, target_object_id)
        
        # skip background i.e. 0
        self.class_names = [name for k, name in enumerate(class_names) if k > 0]

    def load_model(self, cfg_path, model_path, class_names=None):
        # Initiate model
        model = Darknet(cfg_path)
        model.load_darknet_weights(model_path)
        model = model.eval()
        return model

    '''
    def detect(self, images, conf_thresh=0.2, nms_thresh=0.0):
        scaled_images = F.interpolate(images, size=self.test_size, mode='bilinear', align_corners=False)
        outputs = self.model(scaled_images)
        outputs = test_size_suppression(outputs, conf_thres=conf_thresh, nms_thres=nms_thresh)

        #print ([item.shape for item in outputs])
        new_outputs = [None for _ in outputs] 
        for k in range(len(outputs)):
            if outputs[k] is not None:
                new_outputs[k]  = resize_boxes(outputs[k], self.test_size[0], self.input_size)
                new_outputs[k] = new_outputs[k][:, [0,1,2,3,4,6]]
            else:
                #new_outputs[k]  = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0, 0.0, -1]]).cuda()
                new_outputs[k]  = [None]
                #print ('------', k, new_outputs[k].shape, new_outputs[k])
            #print (k, outputs[k], new_outputs[k])

        #print (new_outputs)
        return new_outputs
    '''

    def detect(self, images, conf_thresh=0.2, nms_thresh=0.0):
        _, h, w, _ = images.shape
        if self.test_size[0] == w and self.test_size[1] == h:
            scaled_images = images
        else:
            scaled_images = F.interpolate(images, size=self.test_size, mode='bilinear', align_corners=False)

        outputs = self.model(scaled_images)
        outputs = non_max_suppression(outputs, conf_thres=conf_thresh, nms_thres=nms_thresh)
        results = wrap_detection_results(outputs, self.test_size[0], self.input_size)
        return results

    def detector_detect(self, img, conf_thresh, nms_thresh):
        with torch.no_grad():
            output = self.detect(img, conf_thresh, nms_thresh)

        return output

'''
def resize_boxes(detection, test_size, input_size):
    h, w = input_size
    rh, rw = float(h)/test_size, float(w)/test_size 
    detection[:,0] *= rw
    detection[:,1] *= rh
    detection[:,2] *= rw
    detection[:,3] *= rh

    return detection
'''
