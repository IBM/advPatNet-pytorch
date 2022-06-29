import torch
from torch import nn
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
from nets.backbone.backbone_config import get_backbone, get_last_conv_dim

#backbone_info = { 'resnet18': {'model': models.resnet18, 'last_conv_dim':512},
#                  'resnet50': {'model': models.resnet50, 'last_conv_dim':2048},
#                  'resnet101': {'model': models.resnet101, 'last_conv_dim': 2048}
#                  }

class BasicLocalizer(nn.Module):
    def __init__(self, backbone, downsample_dim=128, fc_dim=256, num_output=6):
        super(BasicLocalizer, self).__init__()

#        resnet_model = backbone_info[backbone]['model'](num_classes=10)
#        last_conv_dim = backbone_info[backbone]['last_conv_dim']

#        self.backbone = nn.Sequential(*list(resnet_model.children())[0:-2])
        self.backbone = get_backbone(backbone)
        last_conv_dim = get_last_conv_dim(backbone)


        self.downsample_dim = downsample_dim
        self.down_sampler = nn.Sequential(
            nn.Conv2d(last_conv_dim, self.downsample_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Regressor for the 3 * 2 affine matrix
        self.fc_dim = fc_dim
        self.last_spatial_dim = 4
        self.fc_loc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.downsample_dim * self.last_spatial_dim * self.last_spatial_dim, self.fc_dim),
            nn.ReLU(True),
            nn.Linear(self.fc_dim, num_output)
        )

    # localization
    def forward(self, x):
        xs = self.backbone(x)
     #   print (xs.shape)
        xs = self.down_sampler(xs)
     #   print (xs.shape)
        xs = xs.view(-1, self.downsample_dim * self.last_spatial_dim * self.last_spatial_dim)
        theta = self.fc_loc(xs)
#        theta = theta.view(-1, 2, 3)
        return theta

class AffineLocalizer(nn.Module):
    def __init__(self, backbone, downsample_dim, fc_dim, predict_dimension=False):
        super(AffineLocalizer, self).__init__()
        self.predict_dimension=predict_dimension
        num_output = 6
        if self.predict_dimension:
            num_output += 1

        self.localizer = BasicLocalizer(backbone, downsample_dim=downsample_dim, fc_dim=fc_dim, num_output=num_output)

        # initialization
        self.localizer.fc_loc[-1].weight.data.zero_()
        if self.predict_dimension:
            self.localizer.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 1], dtype=torch.float))
        else:
            self.localizer.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # localization
    def forward(self, x):
        x = self.localizer(x)
        if self.predict_dimension:
            return x[:,:6].view(-1, 2, 3), x[:,-1]
        else:
            return x.view(-1, 2, 3), None

# based on https://github.com/WarBean/tps_stn_pytorch/blob/master/tps_grid_gen.py
class BoundedTPSLocalizer(nn.Module):

    def __init__(self, backbone, downsample_dim, fc_dim, grid_height, grid_width, target_control_points, predict_dimension=False):
        super(BoundedTPSLocalizer, self).__init__()
        self.precit_dimension = predict_dimension
        num_output = grid_height * grid_width * 2
        if self.precit_dimension:
            num_output += 1
        self.cnn = BasicLocalizer(backbone, downsample_dim=downsample_dim, fc_dim=fc_dim, num_output=num_output)

        #bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        #bias = bias.view(-1)
        bias = torch.zeros(num_output)
        if self.precit_dimension:
            bias[:-1] = torch.from_numpy(np.arctanh(target_control_points.numpy())).view(-1)
            bias[-1] = 1.0
        else:
            bias = torch.from_numpy(np.arctanh(target_control_points.numpy())).view(-1)

        self.cnn.fc_loc[-1].bias.data.copy_(bias)
        self.cnn.fc_loc[-1].weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        if self.precit_dimension:
            return torch.tanh(points[:,:-1]).view(batch_size, -1, 2), points[:,-1]
        else:
            return torch.tanh(points).view(batch_size, -1, 2), None

# based on https://github.com/WarBean/tps_stn_pytorch/blob/master/tps_grid_gen.py
class UnBoundedTPSLocalizer(nn.Module):

    def __init__(self, backbone, downsample_dim, fc_dim, grid_height, grid_width, target_control_points, predict_dimension=False):
        super(UnBoundedTPSLocalizer, self).__init__()

        self.precit_dimension = predict_dimension
        num_output = grid_height * grid_width * 2
        if self.precit_dimension:
            num_output += 1

        self.cnn = BasicLocalizer(backbone,  downsample_dim=downsample_dim, fc_dim=fc_dim, num_output=num_output)

#        bias = target_control_points.view(-1)
        bias = torch.zeros(num_output)
        if self.precit_dimension:
            bias[:-1] = target_control_points.view(-1)
            bias[-1] = 1.0
        else:
            bias = target_control_points.view(-1)

        self.cnn.fc_loc[-1].bias.data.copy_(bias)
        self.cnn.fc_loc[-1].weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        if self.precit_dimension:
            return points[:, :-1].view(batch_size, -1, 2), points[:,-1]
        else:
            return points.view(batch_size, -1, 2), None
