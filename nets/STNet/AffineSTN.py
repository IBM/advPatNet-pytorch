import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


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


class AffineSTNNet_depreciated(nn.Module):
    def __init__(self, backbone):
        super(AffineSTNNet_depreciated, self).__init__()
        if backbone == 'resnet18':
            resnet_model = models.resnet18(num_classes=10)
        elif backbone == 'resnet50':
            resnet_model = models.resnet50(num_classes=10)
        self.localizer = nn.Sequential(*list(resnet_model.children())[0:8])

        self.last_conv_dim = 128
        self.down_sampler = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 3 * 2)
        )

        # weight initialization
       # init_module(self.down_sampler)
       # init_module(self.fc_loc)

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # localization
    def localization(self, x):
        xs = self.localizer(x)
     #   print (xs.shape)
        xs = self.down_sampler(xs)
     #   print (xs.shape)
        xs = xs.view(-1, 128 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta

    # transform the template
    def forward(self, x, template):
        # transform the input
        theta = self.localization(x)
        grid = F.affine_grid(theta, template.size(), align_corners=False)
        try:
           y = F.grid_sample(template, grid, align_corners=False)
        except:
           y = F.grid_sample(template, grid)

        return y, theta
