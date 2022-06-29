import torch
import torch.nn.functional as F

def generate_patch(type, size=(416, 416)):
    """
    Generate a random patch as a starting point for optimization.

    :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
    :return:
    """

    if type == 'gray':
        adv_patch = torch.full((3, size[0], size[1]), 0.5)
    elif type == 'random':
        adv_patch = torch.rand((3, size[0], size[1]))

    return adv_patch

def generate_border_mask(patch_size, border_size):
    h = patch_size[0]
    w = patch_size[1]
    border_mask = torch.full((3, h, w), 0)
    bottom = border_size
    top = h - bottom
    border_mask[:, bottom:top, :] = 1.0

    return border_mask

def paste_patch_to_frame(patch, patch_bb, img, img_bb):
    n, c, _, _ = patch.shape
    # create tensor
    img_h, img_w = img.shape[2:]
    x = torch.cuda.FloatTensor(n, c, img_h, img_w).fill_(0)
    for i, bbox in enumerate(patch_bb):
        pb, pl, ph, pw = bbox
        ib, il, ih, iw = img_bb[i]
        resized_tmpl = F.interpolate(patch[i, :, pb:pb + ph, pl:pl + pw].unsqueeze(0), size=(ih, iw),
                                     mode='bilinear', align_corners=False)
        x[i, :, ib:ib + ih, il:il + iw] = resized_tmpl.squeeze()

    return x

def get_max_detection_score(output, obj_bbox, target_obj_id=0, min_detection_score=0.3, loss_type = 'max'):
    # output a list of (x1,y1,x2,y2, object_conf, class_pred)
    # obj_bbox: a list of (x1, y1, x2, y2)
    assert len(output) == obj_bbox.shape[0]

    # minimum prob. is set to 0.3
    max_prob = torch.zeros((obj_bbox.shape[0], 1)).cuda()
    #print ('max_prob_0', max_prob)
    for k in range(len(output)):
        detection = output[k]
        if isinstance(detection, list) and detection[0] is None:
            continue

        person_detection = detection[detection[:, -1] == target_obj_id]
        if person_detection.shape[0] == 0:
            continue

        bbox = obj_bbox[k]
        xc = (person_detection[:, 0] + person_detection[:, 2]) / 2.0
        yc = (person_detection[:, 1] + person_detection[:, 3]) / 2.0

        x_inside = (xc > bbox[0]) & (bbox[2] > xc)
        y_inside = (yc > bbox[1]) & (bbox[3] > yc)

        xy_inside = x_inside & y_inside
        # assert any(xy_inside>0), (xy_inside, xc, yc, detection, bbox, x_inside, y_inside, xy_inside)
        if loss_type == 'ce':
            prob = person_detection[xy_inside, 4:-1]
            if len(prob) > 0:
                max_prob[k] = torch.nn.functional.nll_loss(prob.log(), (prob.shape[-1] - 1) * torch.ones(len(prob), dtype=torch.long, device=prob.device))
            min_detection_score = 0.0
        else:
            if any(xy_inside > 0):
                if loss_type == 'avg':
                    max_prob[k] = torch.mean(person_detection[xy_inside, 4])
                    min_detection_score = 0.0
                else:
                    max_prob[k] = torch.max(person_detection[xy_inside, 4])
    max_prob = torch.clamp(max_prob, min=min_detection_score)
    return max_prob


# total variation of the patch
def get_totalVariation(adv_patch):
    tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
    tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
    tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
    tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
    tv = tvcomp1 + tvcomp2
    return tv / torch.numel(adv_patch)

# adversarial loss
def advsarial_loss(max_detection_score, loss_type):
    if loss_type == '':
        return None

    if loss_type == '':
        return None

    if loss_type == '':
        return None

    return max_detection_score