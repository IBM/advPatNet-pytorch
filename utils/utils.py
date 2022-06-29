# reference: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/utils.py
import math
import os
import typing

import torch
import torch.nn as nn
import torch.distributed
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

def set_gradient_false(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

def fix_checkpoint_key(checkpoint):
    new_dict = {}
    for k, v in checkpoint.items():
        # TODO: a better approach:
        new_dict[k.replace("module.", "")] = v
    return new_dict

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def single_bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def combine_images(img_list):
    widths, heights = zip(*(i.size for i in img_list))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in img_list:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


def visualize_detections(img, detections, savename=None, class_names=None):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    draw = ImageDraw.Draw(img)
    fnt = ImageFont.truetype('utils/arial.ttf', 30)
    for det in detections:
        if det is None:
            continue
        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]

        cls_conf = det[4]
        cls_id = int(det[5])
        rgb = (255, 0, 0)
        if class_names and cls_id >= 0:
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)

            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw_text = '%s (%.2f)' % (class_names[cls_id], cls_conf)
            #draw.rectangle([x1, y1 - 13, x1 + 38, y1], fill=rgb)
            draw.text((x1 + 2, y1 - 13), draw_text, fill='yellow', font=fnt)
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=6)
    if savename:
        img.save(savename)
    return img

def visualize_detections_1(img, det, idx, savename=None, class_names=None):
        colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

        def get_color(c, x, max_val):
          ratio = float(x) / max_val * 5
          i = int(math.floor(ratio))
          j = int(math.ceil(ratio))
          ratio = ratio - i
          r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
          return int(r * 255)

        draw = ImageDraw.Draw(img)
        fnt = ImageFont.truetype('utils/arial.ttf', 30)
        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]

        cls_conf = 1.0
        cls_id = idx
        rgb = (255, 0, 0)
        if class_names and cls_id >= 0:
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)

            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw_text = '%s (%.2f)' % (class_names[cls_id], cls_conf)
            #draw.rectangle([x1, y1 - 13, x1 + 38, y1], fill=rgb)
            draw.text((x1 + 2, y1 - 13), draw_text, fill='yellow', font=fnt)
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=6)
        if savename:
            img.save(savename)
        return img


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor.contiguous(), async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def detection_accuracy(detector: nn.Module, images: torch.Tensor, names: typing.List[str],
                       person_bb: torch.Tensor, big_patch_bb: torch.Tensor, nms_thresh: float,
                       conf_thresh: float, iou_threshold: float, target_obj_id: int,
                       output_dir: str = None, histogram: bool = False) -> torch.Tensor:
    """

    :param detector: the detector model
    :param images: images, NxCxHxW
    :param names: file names
    :param person_bb: the ground truth of the bounding box of the targeted person
    :param nms_thresh: the threshold of NMS
    :param conf_thresh: the threshold of detection confidence (score)
    :param iou_threshold:
    :param target_obj_id: the target class id
    :param output_dir: the folder for saving detection results, if None, do not save.
    :return: detection success tensor
    """

    dsr = []
    if torch.distributed.is_initialized():
        detection_results = detector.detect(images, conf_thresh, nms_thresh)
    else:
        # do it one by one because it won't start DataParallel, i.e. using one gpu only
        detection_results = []
        for k in range(images.shape[0]):
            adv_img = images[k].detach().clone()
#            tmp = detector.detect(adv_img.unsqueeze(0), conf_thresh, nms_thresh)
            tmp = detector.detector_detect(adv_img.unsqueeze(0), conf_thresh, nms_thresh)
            detection_results.extend(tmp)

    for idx, detection in enumerate(detection_results):
        det_correct = 0
        det_score = 0
        for det in detection:
            if det is None:
                continue
            if det[-1] == target_obj_id:  # only count the person
                det_score = det[-2] #assume there is only one dection (not always TRUE)
                if single_bbox_iou(det[:4], person_bb[idx]) > iou_threshold:
                    det_correct = 1
                    break
        #if det_score == 0:
        #   print (names[idx], det_score)
        if not histogram:
            if det_correct == 1:
                dsr.append(1.0)
            else:
                dsr.append(0.0)
        else:
            #height = person_bb[idx][3] - person_bb[idx][1]
            height = big_patch_bb[idx][2]
            dsr.append([float(det_correct), det_score, height])

        if output_dir:
            img = transforms.ToPILImage()(images[idx].cpu())
            output_path = os.path.join(output_dir, f"{names[idx]}_predictions_{det_correct}.jpg")
            visualize_detections(img, detection, output_path, detector.class_names)
            #img.save(os.path.join(output_dir, names[idx]+".jpg"))

    dsr = torch.tensor(dsr, device=images.device)
    if torch.distributed.is_initialized():
        dsr = concat_all_gather(dsr)
    return dsr

