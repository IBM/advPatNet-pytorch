
import torch.nn.functional as F
import torch.nn as nn

from . import LIGHTINGNET_REGISTRY
from .fine_generator import FineGenerator

@LIGHTINGNET_REGISTRY.register()
class CCGenerator(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ngf']
        self.use_cuda = config['cuda']
        self.device_ids = config['gpu_ids']

        self.backbone = FineGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids, output_dim=4)

        # self.pool = nn.Conv2d(3, 3, kernel_size=9, stride=8, bias=True)
        self.pool = nn.Conv2d(3, 3, kernel_size=11, stride=10, bias=True)

    # transform the template
    def forward(self, x):
        return self.forward_template(x)

    # the normalized output is NOT required as we need to learn the lighting condition
    # changes in the environment
    def forward_template(self, x):
        y = self.backbone(x, add_input_back=False)
        _, _, h, w = y.shape

        rgb = y[:, :3, :, :]
        rgb = F.normalize(rgb, p=2, dim=1)
        confidence = y[:, -1, :, :].view(-1, h * w)
        confidence = F.softmax(confidence, dim=1)
        confidence = confidence.view(-1, 1, h, w)

        rgb = rgb * confidence
        #rgb = F.relu(rgb)
        rgb = F.relu(self.pool(rgb))

        return rgb

    def generate(self, src_img, frame_img):
        rgb = self.forward_template(frame_img)
        return src_img * rgb
