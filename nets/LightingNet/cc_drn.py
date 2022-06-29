
import torch.nn.functional as F
import torch.nn as nn
from nets.backbone.backbone_config import get_backbone, get_last_conv_dim

from . import LIGHTINGNET_REGISTRY


@LIGHTINGNET_REGISTRY.register()
class CCDRN(nn.Module):

    def __init__(self, config=None):
        super().__init__()

        backbone_name = config['lct_backbone']
        self.backbone = get_backbone(backbone_name)
        last_conv_dim = get_last_conv_dim(backbone_name)
        ori_output_size = 256 // 4

        self.projection = nn.Sequential(
            nn.Conv2d(last_conv_dim, 64, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=1, padding=0, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.Conv2d(3, 3, kernel_size=9, stride=ori_output_size // 8, bias=True)

    # transform the template
    def forward(self, x):
       return self.forward_template(x)

    # the normalized output is NOT required as we need to learn the lighting condition
    # changes in the environment
    def forward_template(self, x):
        y = self.backbone(x)
        _, _, h, w = y.shape

        rgb = y[:, :3, :, :]
        rgb = F.normalize(rgb, p=2, dim=1)
        confidence = y[:, -1, :, :].view(-1, h * w)
        confidence = F.softmax(confidence, dim=1)
        confidence = confidence.view(-1, 1, h, w)

        rgb = rgb * confidence
        rgb = F.relu(self.pool(rgb))
        return rgb

    def generate(self, src_img, frame_img):
        rgb = self.forward_template(frame_img)
        return src_img * rgb
