"""
Training code for Adversarial patch training

"""
from torchvision import transforms
import torch
from torch import nn
from PIL import Image
from .advPatch_util import generate_patch, generate_border_mask

class AdvPatch(nn.Module):
    def __init__(self, config):
        super(AdvPatch, self).__init__()
        self.adv_patch_size = tuple(config['adv_patch_size'])
        self.apply_border_mask = config['apply_border_mask']
        print(' ===== AdvPatch size: (%d %d %d) =======' % (self.adv_patch_size))

        if self.apply_border_mask:
            self.border_value = config['border_value']
            border_size = int(self.adv_patch_size[0] * config['border_mask_ratio'] + 0.5)
            print(' ===== Border mask size: %d Value: %d =======' % (border_size, self.border_value))
            self.border_mask = nn.Parameter(generate_border_mask(self.adv_patch_size, border_size))

        self.adv_patch = nn.Parameter(generate_patch("gray", size=self.adv_patch_size[:2]))

    @property
    def patch_size(self):
        return self.adv_patch_size

    @property
    def border_size(self):
        return self.border_size if self.apply_border_mask else 0

    def learnable(self):
        return [self.adv_patch]

    def clip(self):
        self.adv_patch.data.clamp_(0, 1)  # keep patch in image range

    def forward(self):
        if self.apply_border_mask:
            # note that nn.parameter cannot be assigned directly, so an internal change is needed
            self.adv_patch.data *= self.border_mask.data
            self.adv_patch.data +=  (1 - self.border_mask.data) * self.border_value

        return self.adv_patch

    def save_patch(self, patch_path):
        adv_patch = self.adv_patch.detach().cpu()
        im = transforms.ToPILImage('RGB')(adv_patch)
        im.save(patch_path)

    def load_patch(self, patch_path):
        patch_img = Image.open(patch_path).convert('RGB')
        w, h = patch_img.size
        adv_h, adv_w = self.adv_patch_size[:2]
        if w !=  adv_w or h != adv_h:
            patch_img = transforms.Resize((adv_h, adv_w), Image.BILINEAR)(patch_img)

        self.adv_patch = torch.nn.Parameter(transforms.ToTensor()(patch_img))

def create_advPatch_model(config):
    return AdvPatch(config)
