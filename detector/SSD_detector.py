from detector.SSD.ssd.modeling.detector import build_detection_model
from detector.SSD.ssd.config import cfg
from detector.SSD.ssd.utils.checkpoint import CheckPointer
from detector.yolo_util import wrap_detection_results, nms

from .object_detector import ObjectDetector
import torch
import torch.nn.functional as F

class SSD_Detector(ObjectDetector):
    def __init__(self, model_name, cfg_path, model_path, class_names, input_size=(-1, -1), test_size=(-1, -1), target_object_id=-1):
        # load SSD
        super().__init__(model_name, cfg_path, model_path, class_names, input_size, test_size, target_object_id)
        data_mean = cfg.INPUT.PIXEL_MEAN
        data_mean[0], data_mean[1], data_mean[2] = data_mean[2], data_mean[0], data_mean[1]
        self.mean = data_mean

        if test_size[0] != cfg.INPUT.IMAGE_SIZE or test_size[1] != cfg.INPUT.IMAGE_SIZE:
            raise Warning('Scale size (%d, %d) is different from the default (%d %d)!' \
                          % (test_size[0], test_size[1], cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE))

        # skip background i.e. 0
        self.class_names = [name for k, name in enumerate(class_names) if k > 0]

    def load_model(self, cfg_path, model_path, class_names=None):
        cfg.merge_from_file(cfg_path)
        cfg.freeze()

        ssd_model = build_detection_model(cfg)
        checkpointer = CheckPointer(ssd_model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.load(model_path, use_latest=False)
        ssd_model.eval()

        return ssd_model

    '''
    def _gpu_normalize(self, x_batch):
        x_batch *= 255.0
        mean = torch.tensor(self.mean).view(1, len(self.mean), 1, 1).cuda()
        std = torch.tensor(self.std).view(1, len(self.std), 1, 1).cuda()
        return (x_batch - mean) / std
    '''

    def detect(self, images, conf_thresh=0.2, nms_thresh=0.0):
        _, h, w, _ = images.shape
        if self.test_size[0] == w and self.test_size[1] == h:
            scaled_images = images
        else:
            scaled_images = F.interpolate(images, size=self.test_size, mode='bilinear', align_corners=False)

        scaled_images *= 255.0
        mean = torch.tensor(self.mean).view(1, len(self.mean), 1, 1).cuda(device=images.device)
        inputs = scaled_images - mean
        outputs = self.model(inputs)
        outputs = [torch.cat((o['boxes'], o['scores'].unsqueeze_(-1), o['labels'].unsqueeze_(-1).float()-1.0), dim=-1) for o in outputs]
        outputs = nms(outputs, conf_thres=conf_thresh, nms_thres=nms_thresh)

        results = wrap_detection_results(outputs, self.test_size[0], self.input_size, skip=False)
        return results

    def detector_detect(self, img, conf_thresh, nms_thresh):
        with torch.no_grad():
            output = self.detect(img, conf_thresh, nms_thresh)

        return output