import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchBlurringModule(nn.Module):
    def __init__(self):
        super(PatchBlurringModule, self).__init__()
        self.blurring_factor = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        blurring_factor = torch.clamp(self.blurring_factor, min=0.1, max=1.0)
        print (self.blurring_factor, blurring_factor)
        return F.interpolate(x, scale_factor= blurring_factor.item(), mode='bilinear', align_corners=False)
