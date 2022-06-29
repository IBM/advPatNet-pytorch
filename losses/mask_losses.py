import torch
from torch import nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from utils.tools import tensor_to_grey
from .smooth_l1_loss import SmoothL1Loss

class MaskLoss(nn.Module):
    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func

    '''
    def forward(self, pred, label, mask_bb):
    # pred: reconstructed or transformed 2d images
    # label: ground-truth images
        l1_losses = []
        for i, bbox in enumerate(mask_bb):
            b, l, h, w = bbox
            crop_pred = pred[i, :, b:b+h, l:l+w]
            crop_label = label[i, :, b:b+h, l:l+w]
            #crop_label = Variable(crop_label.data.cuda(),requires_grad=False)
            l1_losses.append(F.l1_loss(crop_pred, crop_label))
        return torch.mean(torch.stack(l1_losses, dim=0))
    '''

    def forward(self, pred, label, mask_bb):
    # pred: reconstructed or transformed 2d images
    # label: ground-truth images
        n, _, h, w = pred.shape
        l1_losses = self.loss_func(pred, label)
        l1_losses = torch.mean(l1_losses.view(n, -1), dim=1)
       # mask_size = mask_bb[:,2:4].clone(device=mask_bb.device).detach()
        w_loss = torch.prod(mask_bb[:,2:4].float().detach(), 1) / (h * w)
        l1_losses /= w_loss
        l1_losses = torch.mean(l1_losses)
        return l1_losses

class L1MaskLoss(MaskLoss):
    def __init__(self):
        super().__init__(nn.L1Loss(reduction='none'))

class L2MaskLoss(MaskLoss):
    def __init__(self):
        super().__init__(nn.MSELoss(reduction='none'))

class SmoothL1MaskLoss(MaskLoss):
    def __init__(self, beta=0.5):
        super().__init__(SmoothL1Loss(reduction='none', beta=beta))

class SIMMMaskLoss(nn.Module):
    def __init__(self, val_range=None):
        super(SIMMMaskLoss, self).__init__()
        self.val_range = val_range

    def forward(self, pred, label, mask_bb):
    # pred: reconstructed or transformed 2d images
    # label: ground-truth images
        ssim_losses = []
        for i, bbox in enumerate(mask_bb):
            b, l, h, w = bbox
            #pred_grey = tensor_to_grey(pred[i, :, b:b+h, l:l+w]).view(1,1,h,w)
            #label_grey = tensor_to_grey(label[i, :, b:b+h, l:l+w]).view(1,1,h,w)
            pred_grey = pred[i, :, b:b+h, l:l+w]
            label_grey = label[i, :, b:b+h, l:l+w]
            pred_grey = pred_grey.view(1,-1,h,w)
            label_grey = label_grey.view(1,-1,h,w)
#            label_grey = Variable(label_grey.data.cuda(),requires_grad=False)

            #print (pred_grey.shape, label_grey.shape)
            ssim_val = 1.0 - ssim(pred_grey, label_grey, val_range=self.val_range)
            ssim_losses.append(ssim_val)
        #print (ssim_losses)
        return torch.mean(torch.stack(ssim_losses, dim=0))
