import torch
from torch import nn
from torchvision import transforms
from .advPatch import AdvPatch
from PIL import Image
from .advPatch_util import generate_patch, generate_border_mask
import os
from utils.gaussian_blur import gaussian_blur
import cv2
import numpy as np

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    '''
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    '''

    filter = cv2.getGaussianKernel(kernel_size, sigma=sigma)
    gaussian_kernel = np.dot(filter, filter.T)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).float()
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size // 2)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class PatchBlurringModule(nn.Module):
    def __init__(self, kernel_size=3):
        super(PatchBlurringModule, self).__init__()
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(torch.ones(3, 1, self.kernel_size, self.kernel_size) / (self.kernel_size * self.kernel_size))

    def forward(self, x):

        weights = torch.sigmoid(self.weights)
        normalized_w = torch.cat([item/item.sum() for item in weights])
        normalized_w.unsqueeze_(1)
        #print (normalized_w)
        return torch.conv2d(x, normalized_w, bias=None, padding=self.kernel_size//2, groups=3) 

class HybridAdvPatch(nn.Module):
    def __init__(self, config):
        super(HybridAdvPatch, self).__init__()
        self.adv_patch_size = tuple(config['adv_patch_size'])
        self.apply_border_mask = config['apply_border_mask']
        print(' ===== AdvPatch size: (%d %d %d) =======' % (self.adv_patch_size))

        if self.apply_border_mask:
            self.border_value = config['border_value']
            border_size = int(self.adv_patch_size[0] * config['border_mask_ratio'] + 0.5)
            print(' ===== Border mask size: %d Value: %d =======' % (border_size, self.border_value))
            self.border_mask = nn.Parameter(generate_border_mask(self.adv_patch_size, border_size))

        #self.adv_patch = nn.Parameter(generate_patch("random", size=self.adv_patch_size[:2]))
        self.adv_patch_near = nn.Parameter(generate_patch("random", size=self.adv_patch_size[:2]))
        self.adv_patch_far = nn.Parameter(generate_patch("random", size=self.adv_patch_size[:2]))

    #    self.high_filter_size = nn.Parameter(torch.tensor(5.0))
    #    self.low_filter_size =  nn.Parameter(torch.tensor(5.0))
    #    self.high_filter_sigma = nn.Parameter(torch.tensor(0.5))
    #    self.low_filter_sigma =  nn.Parameter(torch.tensor(0.5))

 #       self.lfilter = get_gaussian_kernel(kernel_size=5, sigma=2, channels=3)
 #       self.hfilter = get_gaussian_kernel(kernel_size=5, sigma=2, channels=3)
        #self.lfilter = PatchBlurringModule(kernel_size=5)
        #self.hfilter = PatchBlurringModule(kernel_size=5)
        self.blending = nn.Parameter(torch.ones(self.adv_patch_size[0], self.adv_patch_size[1]) * 0.5)

        self.collaborative_learning = not config['CL_pretrained']
        self.collaborative_weight = None

    @property
    def patch_size(self):
        return self.adv_patch_size

    @property
    def border_size(self):
        return self.border_size if self.apply_border_mask else 0

    def learnable(self):
        #return [self.adv_patch_near, self.adv_patch_far, self.adv_patch]
        #return [self.adv_patch_near, self.adv_patch_far] + list(self.lfilter.parameters()) + \
        #    list(self.hfilter.parameters())
        return [self.adv_patch_near, self.adv_patch_far, self.blending]

    def clip(self):
        self.adv_patch.data.clamp_(0, 1)
        self.adv_patch_near.data.clamp_(0, 1)
        self.adv_patch_far.data.clamp_(0, 1)
#        self.high_filter_size.data.clamp_(3.0, 7.0)
#        self.low_filter_size.data.clamp_(3.0, 7.0)
#        self.high_filter_sigma.data.clamp_(0.3, 0.8)
#        self.low_filter_sigma.data.clamp_(0.3, 0.8)
        #self.high_filter_size = torch.round(self.high_filter_size)
        #self.low_filter_size = torch.round(self.low_filter_size)
        #print (self.low_filter_size, self.high_filter_size)

    def forward(self):
        '''
        lf_advT_patch = self.lfilter(self.adv_patch_far.unsqueeze(0))
        lf_advT_patch = lf_advT_patch.squeeze(0)
        hf_advT_patch = self.hfilter(self.adv_patch_near.unsqueeze(0))
        hf_advT_patch = hf_advT_patch.squeeze(0)
        hf_advT_patch = self.adv_patch_near - hf_advT_patch

        advT_patch = lf_advT_patch + hf_advT_patch
        advT_patch.data.clamp_(0,1)
        '''
        blending = torch.sigmoid(self.blending)
        self.adv_patch = self.adv_patch_far * blending + self.adv_patch_near * ( 1.0 - blending)

        if self.training:
            return self.adv_patch, self.adv_patch_near, self.adv_patch_far

        return self.adv_patch

    def save_patch(self, patch_path):
        adv_patch = self.adv_patch.detach().cpu()
        im = transforms.ToPILImage('RGB')(adv_patch)
        im.save(patch_path)

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

def create_hybrid_advPatch_model(config):
    return HybridAdvPatch(config)
