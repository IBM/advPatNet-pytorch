import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import itertools
from nets.STNet.STNLocalizer import BoundedTPSLocalizer, UnBoundedTPSLocalizer
from nets.STNet.tps_grid_gen import TPSGridGen
from nets.STNet.grid_sample import grid_sample

class TpsSTNNet(nn.Module):
    def __init__(self, config):
        super(TpsSTNNet, self).__init__()

        r1, r2 = config['TPS_range'] # height and width
        grid_height, grid_width = config['TPS_grid']
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_width - 1)),
        )))

        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)

        GridLocNet = {
            'unbounded_stn': UnBoundedTPSLocalizer,
            'bounded_stn': BoundedTPSLocalizer,
        }[config['TPS_localizer']]

        backbone = config['loc_backbone']
#        img_height, img_width, _ = config['image_shape']
        #img_height, img_width, _ = config['image_shape'] if config['template_resize'] else config['template_shape']
        img_height, img_width, _ = config['template_shape']
        downsample_dim = config['loc_downsample_dim']
        fc_dim = config['loc_fc_dim']
        adjust_patchDim = config['adjust_patch_size']
        self.loc_net = GridLocNet(backbone, downsample_dim, fc_dim, grid_height, grid_width, target_control_points, predict_dimension=adjust_patchDim)
        self.tps = TPSGridGen(img_height, img_width, target_control_points)

    # transform the template
    def forward(self, x, template):
        # transform the input
        batch_size = x.size(0)
        source_control_points, output_scale = self.loc_net(x)
        source_coordinate = self.tps(source_control_points)
        _, _, H, W = template.size()
        grid = source_coordinate.view(batch_size, H, W, 2)
        y = grid_sample(template, grid)

        return y, source_control_points, output_scale
