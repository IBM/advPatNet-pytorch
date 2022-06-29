import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class EOTTransformer(nn.Module):
    def __init__(self, contrast=(0.9, 1.1), brightness=(-0.1, 0.1), rotation=8.0, scale=(0.85, 1.15)):
    #def __init__(self, contrast=(0.8, 1.2), brightness=(-0.2, 0.2), rotation=8.0, scale=(1.0, 1.0)):
        super(EOTTransformer, self).__init__()
        self.contrast_min, self.contrast_max = contrast
        self.brightness_min, self.brightness_max = brightness
        self.rotation_min, self.rotation_max = -rotation, rotation
        self.scale_min, self.scale_max = scale
        #self.theta = nn.Parameter(torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float))
        #self.theta.cuda()

        #self.compose = transforms.Compose([transforms.ColorJitter(brightness, contrast),
        #                                   transforms.RandomAffine(rotation, scale=scale, fillcolor=0.0)])

    # x in range [0 1]
    def forward(self, x, do_rotate=True):
        num_batch= x.shape[0]
        contrast = torch.FloatTensor(num_batch, 1, 1, 1).uniform_(self.contrast_min, self.contrast_max).cuda()
        brightness = torch.FloatTensor(num_batch, 1, 1, 1).uniform_(-self.brightness_min, self.brightness_max).cuda()
        y = torch.clamp(x * contrast + brightness, 0, 1)

        # do affine transformation
        a = np.random.uniform(self.rotation_min, self.rotation_max, num_batch) / 180 * np.pi
        s = np.random.uniform(self.scale_min, self.scale_max, num_batch)

        t = np.stack ((np.cos(a)*s, -np.sin(a)*s, np.zeros(num_batch), np.sin(a)*s, np.cos(a)*s, np.zeros(num_batch)), axis=1)
        t = t.reshape(num_batch, 2, 3)
 #       t = np.array([[np.cos(angle), -1.0*np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0]], dtype=np.float)*scale
 #       self.theta.data.copy_(torch.from_numpy(t))
        #print (angle/np.pi*180, scale, t, self.theta)
        t = torch.tensor(t, dtype=torch.float).cuda()
        grid = F.affine_grid(t, y.size(), align_corners=False)
        y = F.grid_sample(y, grid)
        y = torch.clamp(y, 0, 1)

        return y
