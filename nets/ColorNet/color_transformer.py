import torch
import torch.nn.functional as F
from torch import nn
from nets.backbone.backbone_config import get_backbone, get_last_conv_dim

class PatternColorTransformer(nn.Module):
    def __init__(self, use_cuda, device_ids):
        super(PatternColorTransformer, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.color_mapping = torch.nn.Parameter(torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])) # 3x3

    # transform the color
    def forward(self, x):
        # transform the input
        n, c, h, w = x.shape
        y = torch.matmul(self.color_mapping, x.view(n, c, -1))
        y = torch.clamp(y, -1., 1.)
        return y.view(n, c, h, w)

class ColorMapEstimator(nn.Module):
    def __init__(self, backbone, fc_dim=256, num_output=9):
        super(ColorMapEstimator, self).__init__()

        resnet_model = get_backbone(backbone)(num_classes=10)
        last_conv_dim = get_last_conv_dim(backbone)

        self.backbone = nn.Sequential(*list(resnet_model.children())[0:-2])
        self.fc_dim = fc_dim
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(last_conv_dim, self.fc_dim),
            nn.ReLU(True),
            nn.Linear(self.fc_dim, num_output)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        theta = self.fc_loc(x.squeeze())
        return theta


class ColorMapNet(nn.Module):
    def __init__(self, backbone, downsample_dim, fc_dim):
        super(ColorMapNet, self).__init__()
        self.color_map = ColorMapEstimator(backbone, fc_dim=fc_dim, num_output=9)

        # initialization
        self.color_map.fc_loc[-1].weight.data.zero_()
        self.color_map.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float))

    # localization
    def forward(self, x):
        x = self.color_map(x)
        x = F.relu(x)
        return x.view(-1, 3, 3)


class LightingColorTransformer(nn.Module):
    def __init__(self, config):
        super(LightingColorTransformer, self).__init__()
        self.color_map = ColorMapNet(backbone=config['loc_backbone'],
                                         downsample_dim=config['loc_downsample_dim'],
                                         fc_dim=config['loc_fc_dim'])

    # transform the template
    def forward(self, x, template):
        # transform the input
        c_map = self.color_map(x)
        n, c, h, w = template.shape
        y = torch.matmul(c_map, template.view(n, c, -1))
        y = torch.clamp(y, -1., 1.)
        y = y.view(n, c, h, w)
        return y, c_map

    def forward_template(self, x, template):
        y, _= self.forward(x, template)
        return y
