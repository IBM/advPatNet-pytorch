import torch
from detector.yolov2_detector import YOLOV2_Detector
from detector.yolov3_detector import YOLOV3_Detector
    
_HAS_FASTER_RCNN_DETECTOR = False
_HAS_SSD_DETECTOR = False
_HAS_DETECTRON2 = False

'''
if torch.__version__ > '1.1.0':
    from detector.SSD_detector import SSD_Detector
    from detector.detectron2_detector import Detectron2Detector
    _HAS_SSD_DETECTOR = True
    _HAS_DETECTRON2 = True
else:
    from detector.faster_rcnn_detector import Faster_RCNN_Detector
    _HAS_FASTER_RCNN_DETECTOR = True
'''
#from detector.faster_rcnn_detector import Faster_RCNN_Detector
#_HAS_FASTER_RCNN_DETECTOR = True

OBJECT_CLASS_NAMES = {'VOC': ['__background__',
                                 'aeroplane', 'bicycle', 'bird', 'boat',
                                 'bottle', 'bus', 'car', 'cat', 'chair',
                                 'cow', 'diningtable', 'dog', 'horse',
                                 'motorbike', 'person', 'pottedplant',
                                 'sheep', 'sofa', 'train', 'tvmonitor'],
                      'COCO': ['__background__',
                               'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                               'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                               'kite', 'baseball bat', 'baseball glove', 'skateboard',
                               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                               'refrigerator', 'book', 'clock', 'vase', 'scissors',
                               'teddy bear', 'hair drier', 'toothbrush']
                      }

DETECTOR_INFO = {
    'YOLO': {
        'YOLO_V2_COCO': {
            'detector': YOLOV2_Detector,
            'cfg_path': './detector/yolov2/yolo.cfg',
            'model_path': './detector/yolov2/yolo.weights',
            'test_size': [416, 416],
            'target_object_id': 0
        },
        'YOLO_V3_COCO': {
            'detector': YOLOV3_Detector,
            'cfg_path': './detector/yolov3/config/yolov3.cfg',
            'model_path': './detector/yolov3/weights/yolov3.weights',
            'test_size': [416, 416],
            'target_object_id': 0
        },
    }
}

if _HAS_FASTER_RCNN_DETECTOR:
    DETECTOR_INFO['Faster_RCNN'] = {
        'Faster_RCNN_VGG16_VOC': {
            'detector': Faster_RCNN_Detector,
            'cfg_path': './detector/faster_rcnn/cfgs/vgg16.yml',
            'model_path': './detector/faster_rcnn/outputs/faster_rcnn_1_6_10021.pth',
            'test_size': [600, -1],
            'target_object_id': 15
        },
        'Faster_RCNN_R101_VOC': {
            'detector': Faster_RCNN_Detector,
            'cfg_path': './detector/faster_rcnn/cfgs/res101_voc.yml',
            'model_path': './detector/faster_rcnn/outputs/faster_rcnn_1_10_625.pth',
            'test_size': [600, -1],
            'target_object_id': 15
        },
        'Faster_RCNN_R101_COCO': {
            'detector': Faster_RCNN_Detector,
            'cfg_path': './detector/faster_rcnn/cfgs/res101_coco.yml',
            'model_path': './detector/faster_rcnn/outputs/faster_rcnn_1_10_9771.pth',
            'test_size': [600, -1],
            'target_object_id': 1
        },
    }

if _HAS_SSD_DETECTOR:
    DETECTOR_INFO['SSD'] = {
        'SSD300_VGG16_COCO': {
            'detector': SSD_Detector,
            'cfg_path': './detector/SSD/configs/vgg_ssd300_coco_trainval35k.yaml',
            'model_path': './detector/SSD/outputs/vgg_ssd300_coco_trainval35k.pth',
            'test_size': [300, 300],
            'target_object_id': 0
        },
        'SSD512_VGG16_COCO': {
            'detector': SSD_Detector,
            'cfg_path': './detector/SSD/configs/vgg_ssd512_coco_trainval35k.yaml',
            'model_path': './detector/SSD/outputs/vgg_ssd512_coco_trainval35k.pth',
            'test_size': [512, 512],
            'target_object_id': 0
        }
    }

if _HAS_DETECTRON2:
    DETECTOR_INFO['Detectron2'] = {
        'DFaster_RCNN_R50_VOC': {
            'detector': Detectron2Detector,
            'cfg_path': './detector/detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml',
            'model_path': '',
            'test_size': [800, 1333],
            'target_object_id': 14
        },
        'DFaster_RCNN_R50_COCO': {
            'detector': Detectron2Detector,
            'cfg_path': './detector/detectron2/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml',
            'model_path': '',
            'test_size': [800, 1333],
            'target_object_id': 0
        },
        'DFaster_RCNN_R50_3x_COCO': {
            'detector': Detectron2Detector,
            'cfg_path': './detector/detectron2/configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml',
            'model_path': '',
            'test_size': [800, 1333],
            'target_object_id': 0
        },
        'DFaster_RCNN_R101_COCO': {
            'detector': Detectron2Detector,
            'cfg_path': './detector/detectron2/configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml',
            'model_path': '',
            'test_size': [800, 1333],
            'target_object_id': 0
        },
        'DRCNN_FPN_R50_COCO': {
            'detector': Detectron2Detector,
            'cfg_path': './detector/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml',
            'model_path': '',
            'test_size': [800, 1333],
            'target_object_id': 0
        },
        'DRCNN_FPN_R50_3x_COCO': {
            'detector': Detectron2Detector,
            'cfg_path': './detector/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
            'model_path': '',
            'test_size': [800, 1333],
            'target_object_id': 0
        },
        'DRCNN_FPN_R101_COCO': {
            'detector': Detectron2Detector,
            'cfg_path': './detector/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
            'model_path': '',
            'test_size': [800, 1333],
            'target_object_id': 0
        },
        'DRetinaNet_R50_COCO': {
            'detector': Detectron2Detector,
            'cfg_path': './detector/detectron2/configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml',
            'model_path': '',
            'test_size': [800, 1333],
            'target_object_id': 0
        },
        'DRetinaNet_R50_3x_COCO': {
            'detector': Detectron2Detector,
            'cfg_path': './detector/detectron2/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml',
            'model_path': '',
            'test_size': [800, 1333],
            'target_object_id': 0
        },
        'DRetinaNet_R101_COCO': {
            'detector': Detectron2Detector,
            'cfg_path': './detector/detectron2/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml',
            'model_path': '',
            'test_size': [800, 1333],
            'target_object_id': 0
        }
    }

'''
DETECTOR_BUILDER = { 'YOLO': {
                        'YOLO_V2': YOLOV2_Detector,
                        'YOLO_V3': YOLOV3_Detector
                        },
                       #'Detectron2': {
                       # 'DFaster_RCNN_R50': Detectron2Detector,
                       # 'DFaster_RCNN_R101': Detectron2Detector,
                       # 'DRCNN_FPN_R50': Detectron2Detector,
                       # 'DRCNN_FPN_R101': Detectron2Detector,
                       # 'DRetinaNet_R50': Detectron2Detector,
                       # 'DRetinaNet_R101': Detectron2Detector},
                       # 'SSD': {
                       #  'SSD300_VGG16': SSD_Detector,
                       #  'SSD512_VGG16': SSD_Detector},
#                        'Faster_RCNN': {
#                          'Faster_RCNN_VGG16': Faster_RCNN_Detector,
#                          'Faster_RCNN_R50': Faster_RCNN_Detector,
#                          'Faster_RCNN_R101': Faster_RCNN_Detector,
#                          'Faster_RCNN_R152': Faster_RCNN_Detector}
                          
}

DETECTOR_CFG_PATH = { 'YOLO': {
                        'YOLO_V2': './detector/yolov2/yolo.cfg',
                        'YOLO_V3': './detector/yolov3/config/yolov3.cfg',
                        },
                       'Detectron2': {
                        'DFaster_RCNN_R50': './detector/detectron2/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml',
                        'DFaster_RCNN_R101': './detector/detectron2/configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml',
                        'DRCNN_FPN_R50': './detector/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml',
                        'DRCNN_FPN_R101': './detector/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
                        'DRetinaNet_R50': './detector/detectron2/configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml',
                        'DRetinaNet_R101': './detector/detectron2/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml'},
                      'SSD': {
                         'SSD300_VGG16': './detector/SSD/configs/vgg_ssd300_coco_trainval35k.yaml',
                         'SSD512_VGG16': './detector/SSD/configs/vgg_ssd512_coco_trainval35k.yaml'},
                      'Faster_RCNN': {
                         'Faster_RCNN_VGG16': './detector/faster_rcnn/cfgs/vgg16.yml',
                         'Faster_RCNN_R101': './detector/SSD/configs/vgg_ssd512_coco_trainval35k.yaml'}
}
'''
