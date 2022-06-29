import torch
from torch import nn

class ObjectDetector(nn.Module):
    def __init__(self, detector_name, cfg_path, model_path, class_names=None, input_size=(-1, -1), test_size=(-1, -1), target_object_id=-1):
        super().__init__()
        self.name = detector_name
        self.model = self.load_model(cfg_path, model_path, class_names)
        self.class_names = class_names
        self.input_size = input_size
        self.test_size = test_size
        self.target_object_id = target_object_id

    def load_model(self, cfg_path, model_path, class_names=None):
        raise NotImplementedError('base class not implemented')

    def forward(self, x, *args, **kwargs):
        return self.detect(x, *args, **kwargs)

    # used for training/val
    def detect(self, images, conf_thresh=0.1, nms_thresh=0):
        raise NotImplementedError('base class not implemented')

    # used for test. Most times it is the same as 'detect', but in some cases such as YOLO_V2, it might be implemented differently from 'detect'.
    def detector_detect(self, images, conf_thresh=0.1, nms_thresh=0):
        raise NotImplementedError('base class not implemented')

    def detect_train(self, images, conf_thresh=0.1, nms_thresh=0):
        """Only use for FasterRCNN type of model since we attack of FasterRCNN at RPN not final output."""
        return self.detect(images, conf_thresh, nms_thresh)

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        return self._model

    @property
    def class_names(self):
        return self._class_names

    @property
    def input_size(self):
        return self._input_size
    
    @property
    def test_size(self):
        return self._test_size

    @property
    def target_object_id(self):
        return self._target_object_id

    @name.setter
    def name(self,val):
        self._name = val

    @model.setter
    def model(self,val):
        self._model = val

    @class_names.setter
    def class_names(self, val):
        self._class_names = val

    @input_size.setter
    def input_size(self, val):
        self._input_size = val
    
    @test_size.setter
    def test_size(self, val):
        self._test_size = val

    @target_object_id.setter
    def target_object_id(self, val):
        self._target_object_id = val

    def cuda(self, device):
        self.model.cuda(device)

    def eval(self):
        self.model.eval()

    def training(self):
        self.model.train()
