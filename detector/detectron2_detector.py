import typing

import torch
import torch.nn.functional as F

from .object_detector import ObjectDetector
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling.meta_arch import GeneralizedRCNN, RetinaNet
from detectron2.structures import Boxes, Instances


class Detectron2Detector(ObjectDetector):
    def __init__(self, model_name: str, cfg_path: str, model_path: str, input_size: typing.Tuple[int, int] = (-1, -1),
                 test_size: typing.Tuple[int, int] = (-1, -1), target_object_id=-1, class_names=None):
        if 'COCO' in model_name.split("_")[-1]:
            class_names = MetadataCatalog.get('coco_2017_train').thing_classes
        else:
            class_names = MetadataCatalog.get('voc_2012_trainval').thing_classes

        self.color_order = 'RGB'
        super().__init__(model_name, cfg_path, model_path, class_names, input_size, test_size, target_object_id)
        self.train_size = self.test_size

    def load_model(self, cfg_path, model_path, class_names=None) -> torch.nn.Module:
        cfg = get_cfg()
        model_cfg_path = cfg_path
        cfg.merge_from_file(model_cfg_path)
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('/'.join(model_cfg_path.split("/")[-2:]))
        cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.BACKBONE.FREEZE_AT = 6
        cfg.freeze()

        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)  # load a file, usually from cfg.MODEL.WEIGHTS
        model.eval()

        self.color_order = cfg.INPUT.FORMAT

        # TODO: hard-coded the size for training
        if isinstance(model, GeneralizedRCNN):
            self.train_size = [250, 500]
        
        return model

    def _preprocessing(self, images: torch.Tensor, image_size: typing.List[int], conf_thresh: float = 0.2, nms_thresh: float = 0.0):
        """Setting the threshold and converting the input format"""
        if isinstance(self.model, GeneralizedRCNN):
            self.model.roi_heads.box_predictor.test_score_thresh = conf_thresh
            self.model.roi_heads.box_predictor.test_nms_thresh = nms_thresh
        elif isinstance(self.model, RetinaNet):
            self.model.score_threshold = conf_thresh
            self.model.nms_threshold = nms_thresh
        else:
            raise ValueError(f"Unknown detector type in Detectron2. {type(self.model)}")

        # compute the resize size
        original_width = images.size(-1)
        original_height = images.size(-2)
        # resize
        if original_width > original_height:
            if (image_size[0] / original_height) * original_width > image_size[1]:
                # exceed the max_size
                new_width = image_size[1]
                new_height = new_width / original_width * original_height
            else:
                new_height = image_size[0]
                new_width = new_height / original_height * original_width
        else:
            if (image_size[0] / original_width) * original_height > image_size[1]:
                # exceed the max_size
                new_height = image_size[1]
                new_width = new_height / original_height * original_width
            else:
                new_width = image_size[0]
                new_height = new_width / original_width * original_height
        images = F.interpolate(images, size=(int(new_height + 0.5), int(new_width + 0.5)), mode='bilinear', align_corners=False)
        images = images * 255.
        if self.color_order == 'BGR':
            images = images[:, [2, 1, 0], :, :]

        inputs = [{'image': images[i, ...], 'height': original_height, 'width': original_width} for i in range(images.size(0))]

        return inputs

    def _postprocessing(self, outputs):
        """Reorganize outputs to desired format."""
        new_outputs = []
        for o in outputs:
            instance = o['instances']
            tmp = [None] if len(instance) == 0 else \
                torch.cat((instance.pred_boxes.tensor, instance.scores.unsqueeze_(-1), instance.pred_classes.unsqueeze_(-1).float()), dim=-1)
            new_outputs.append(tmp)
        return new_outputs

    def detect(self, images: torch.Tensor, conf_thresh: float = 0.2, nms_thresh: float = 0.0) -> typing.List[torch.Tensor]:
        """

        :param images: NxCxHxW, in the original size, e.g. H: 1080, W: 1920
        :param conf_thresh:
        :param nms_thresh:
        :return:
            [Nx6], a list of tensors, each element in the list is for one image, and each image could have N detetion results,
            each result has 6-dimention (x1, y1, x2, y2, detection score, object_id).
        """
        inputs = self._preprocessing(images, self.test_size, conf_thresh, nms_thresh)
        outputs = self.model(inputs)
        new_outputs = self._postprocessing(outputs)
        return new_outputs

    def detect_train(self, images: torch.Tensor, conf_thresh: float = 0.2, nms_thresh: float = 0.0) -> typing.List[torch.Tensor]:
        """

        :param images: NxCxHxW, in the original size, e.g. H: 1080, W: 1920
        :param conf_thresh:
        :param nms_thresh:
        :return:
            Nx6, each vector contains (x1, y1, x2, y2, detection score, object_id)
        """

        return self._detect_normal(images, conf_thresh, nms_thresh)
        #return self._detect_train_targeted(images, conf_thresh, nms_thresh)


    def _detect_normal(self, images: torch.Tensor, conf_thresh: float = 0.2, nms_thresh: float = 0.0) -> typing.List[torch.Tensor]:
        """

        :param images: NxCxHxW, in the original size, e.g. H: 1080, W: 1920
        :param conf_thresh:
        :param nms_thresh:
        :return:
            Nx6, each vector contains (x1, y1, x2, y2, detection score, object_id)
        """
        return self.detect(images, conf_thresh, nms_thresh)


    def _detect_train_rpn(self, images: torch.Tensor, conf_thresh: float = 0.2, nms_thresh: float = 0.0) -> typing.List[torch.Tensor]:
        """

        :param images: NxCxHxW, in the original size, e.g. H: 1080, W: 1920
        :param conf_thresh:
        :param nms_thresh:
        :return:
            Nx6, each vector contains (x1, y1, x2, y2, detection score, object_id)
        """
        inputs = self._preprocessing(images, self.train_size, conf_thresh, nms_thresh)
        if isinstance(self.model, GeneralizedRCNN):
            # attacking proposals
            images = self.model.preprocess_image(inputs)

            features = self.model.backbone(images.tensor)
            proposals, _ = self.model.proposal_generator(images, features)
            for p in proposals:
                p.pred_boxes = p.proposal_boxes
                p.scores = p.objectness_logits.clamp(0)
                p.pred_classes = torch.ones_like(p.objectness_logits) * self.target_object_id
            outputs = GeneralizedRCNN._postprocess(proposals, inputs, images.image_sizes)
        else:
            outputs = self.model(inputs)
        new_outputs = self._postprocessing(outputs)
        return new_outputs

    def _detect_train_targeted(self, images: torch.Tensor, conf_thresh: float = 0.2, nms_thresh: float = 0.0) -> typing.List[torch.Tensor]:
        """

        :param images: NxCxHxW, in the original size, e.g. H: 1080, W: 1920
        :param conf_thresh:
        :param nms_thresh:
        :return:
            Nx6, each vector contains (x1, y1, x2, y2, detection score, object_id)
        """
        #inputs = self._preprocessing(images, self.train_size, conf_thresh, nms_thresh)
        inputs = self._preprocessing(images, self.train_size, 0.0, 1.0)
        if isinstance(self.model, GeneralizedRCNN):
            # attacking proposals
            images = self.model.preprocess_image(inputs)
            features = self.model.backbone(images.tensor)
            with torch.no_grad():
                proposals, _ = self.model.proposal_generator(images, features)
            
            predictions, _ = self.model.roi_heads(images, features, proposals, logits=True)
            boxes = self.model.roi_heads.box_predictor.predict_boxes(predictions, proposals)
            scores = self.model.roi_heads.box_predictor.predict_probs(predictions, proposals)
            image_shapes = [x.image_size for x in proposals]
            instances = []
            for box, score, image_shape in zip(boxes, scores, image_shapes):
                num_bbox_reg_classes = box.shape[1] // 4
                # Convert to Boxes to use the `clip` function ...
                box = Boxes(box.reshape(-1, 4))
                box.clip(image_shape)
                box = box.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
                pred_classes = torch.argmax(score, dim=-1)
                person = torch.where(pred_classes == self.target_object_id)[0]
                score = score[person]
                box = box[person, self.target_object_id, ...]
                result = Instances(image_shape)
                result.pred_boxes = Boxes(box)
                result.scores = score
                result.pred_classes = pred_classes[person]
                instances.append(result)
            outputs = GeneralizedRCNN._postprocess(instances, inputs, images.image_sizes)
            new_outputs = []
            for o in outputs:
                instance = o['instances']
                tmp = [None] if len(instance) == 0 else \
                      torch.cat((instance.pred_boxes.tensor, instance.scores, instance.pred_classes.unsqueeze_(-1).float()), dim=-1)
                new_outputs.append(tmp)

            return new_outputs
        else:
            outputs = self.model(inputs)
        new_outputs = self._postprocessing(outputs)
        #print(len(new_outputs))
        #for o in new_outputs:
        #    print(o.shape, flush=True)
        return new_outputs

    def detector_detect(self, images: torch.Tensor, conf_thresh: float = 0.2, nms_thresh: float = 0.0) -> typing.List[torch.Tensor]:
        with torch.no_grad():
            output = self.detect(images, conf_thresh, nms_thresh)
        return output
