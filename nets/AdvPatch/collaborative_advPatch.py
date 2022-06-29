import torch
from torch import nn
from torchvision import transforms
from .advPatch import AdvPatch
from PIL import Image
from .advPatch_util import generate_patch, generate_border_mask
import os

class CollaborativeAdvPatch(nn.Module):
    def __init__(self, config):
        super(CollaborativeAdvPatch, self).__init__()
        self.adv_patch_size = tuple(config['adv_patch_size'])
        self.apply_border_mask = config['apply_border_mask']
        print(' ===== AdvPatch size: (%d %d %d) =======' % (self.adv_patch_size))

        if self.apply_border_mask:
            self.border_value = config['border_value']
            border_size = int(self.adv_patch_size[0] * config['border_mask_ratio'] + 0.5)
            print(' ===== Border mask size: %d Value: %d =======' % (border_size, self.border_value))
            self.border_mask = nn.Parameter(generate_border_mask(self.adv_patch_size, border_size))

        self.collaborative_learning = not config['CL_pretrained']
        #self.adv_patch = nn.Parameter(generate_patch("gray", size=self.adv_patch_size[:2]))
        #self.adv_patch_near = nn.Parameter(generate_patch("gray", size=self.adv_patch_size[:2]))
        #self.adv_patch_far = nn.Parameter(generate_patch("gray", size=self.adv_patch_size[:2]))
        self.adv_patch = nn.Parameter(generate_patch("random", size=self.adv_patch_size[:2]))
        self.adv_patch_near = nn.Parameter(generate_patch("random", size=self.adv_patch_size[:2]))
        self.adv_patch_far = nn.Parameter(generate_patch("random", size=self.adv_patch_size[:2]))

        # learnable weights
        if config.get('collaborative_weights', False):
            self.collaborative_weight = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
            nn.init.constant_(self.collaborative_weight[0].weight, 10.0)
            nn.init.constant_(self.collaborative_weight[0].bias, -2.5)
        else:
            self.collaborative_weight = None

    @property
    def patch_size(self):
        return self.adv_patch_size

    @property
    def border_size(self):
        return self.border_size if self.apply_border_mask else 0

    def learnable(self):
        out = [self.adv_patch] if not self.collaborative_learning else \
            [self.adv_patch, self.adv_patch_near, self.adv_patch_far]
        if self.collaborative_weight:
            out += [self.collaborative_weight[0].weight, self.collaborative_weight[0].bias]
        return out

    def clip(self):
        self.adv_patch.data.clamp_(0, 1)  # keep patch in image range
        if self.collaborative_learning:
            self.adv_patch_near.data.clamp_(0, 1)
            self.adv_patch_far.data.clamp_(0, 1)

        if self.collaborative_weight:
            self.collaborative_weight[0].weight.data.clamp_(9.0, 11.0)
            self.collaborative_weight[0].bias.data.clamp_(-3.0, -2.0)

    def forward(self):
        if self.apply_border_mask:
            # note that nn.parameter cannot be assigned directly, so an internal change is needed
            self.adv_patch.data *= self.border_mask.data
            self.adv_patch.data +=  (1 - self.border_mask.data) * self.border_value

        if self.training:
            return self.adv_patch, self.adv_patch_near, self.adv_patch_far

        return self.adv_patch

    def save_patch(self, patch_path):
        adv_patch = self.adv_patch.detach().cpu()
        im = transforms.ToPILImage('RGB')(adv_patch)
        im.save(patch_path)

        if self.collaborative_learning:
            base_path, adv_file = os.path.split(patch_path)
            base_file, ext = adv_file.split('.')

            adv_patch_near = self.adv_patch_near.detach().cpu()
            im_near = transforms.ToPILImage('RGB')(adv_patch_near)
            im_near.save(os.path.join(base_path, base_file + '_near.' + ext))

            adv_patch_far = self.adv_patch_far.detach().cpu()
            im_far = transforms.ToPILImage('RGB')(adv_patch_far)
            im_far.save(os.path.join(base_path, base_file + '_far.' + ext))

    def _load_patch_image(self, patch_path):
        patch_img = Image.open(patch_path).convert('RGB')
        w, h = patch_img.size
        # first dim is height
        adv_h, adv_w = self.adv_patch_size[:2]
        if w !=  adv_w or h != adv_h:
            patch_img = transforms.Resize((adv_h, adv_w), Image.BILINEAR)(patch_img)
        return patch_img

    def load_patch(self, patch_path):
        patch_img = self._load_patch_image(patch_path)
        self.adv_patch = torch.nn.Parameter(transforms.ToTensor()(patch_img))

        if self.collaborative_learning:
            base_path, adv_file = os.path.split(patch_path)
            base_file, ext = adv_file.split('.')
            adv_near_file = os.path.join(base_path, base_file+'_near.'+ext)
            if os.path.isfile(adv_near_file):
                near_patch_img = self._load_patch_image(adv_near_file)
                self.adv_patch_near = torch.nn.Parameter(transforms.ToTensor()(near_patch_img))

            adv_far_file = os.path.join(base_path, base_file+'_far.'+ext)
            if os.path.isfile(adv_far_file):
                far_patch_img = self._load_patch_image(adv_far_file)
                self.adv_patch_far = torch.nn.Parameter(transforms.ToTensor()(far_patch_img))

    def load_pretrained_patch(self, near_patch_path, far_patch_path):
        assert not self.collaborative_learning
        if near_patch_path is not None:
            near_patch_img = self._load_patch_image(near_patch_path)
            self.adv_patch_near = torch.nn.Parameter(transforms.ToTensor()(near_patch_img))
            print ('Loading near model from %s' % (near_patch_path))

        if far_patch_path is not None:
            far_patch_img = self._load_patch_image(far_patch_path)
            self.adv_patch_far = torch.nn.Parameter(transforms.ToTensor()(far_patch_img))
            print ('Loading far model from %s' % (far_patch_path))

def create_collaborative_advPatch_model(config):
    return CollaborativeAdvPatch(config)
