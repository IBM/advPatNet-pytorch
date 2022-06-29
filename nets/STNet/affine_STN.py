import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from nets.STNet.STNLocalizer import AffineLocalizer

def init_module(module):
    for m in module():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class AffineSTNNet(nn.Module):
    def __init__(self, config):
        super(AffineSTNNet, self).__init__()
        self.localizer = AffineLocalizer(backbone=config['loc_backbone'],
                                         downsample_dim=config['loc_downsample_dim'],
                                         fc_dim=config['loc_fc_dim'],
                                         predict_dimension=config['adjust_patch_size'])

    # transform the template
    def forward(self, x, template):
        # transform the input
        theta, output_scale = self.localizer(x)
        grid = F.affine_grid(theta, template.size())
        y = F.grid_sample(template, grid)

        return y, theta, output_scale
