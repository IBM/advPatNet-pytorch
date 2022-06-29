import torch
import torch.nn.functional as F
from torch import nn
from nets.backbone.backbone_config import get_backbone, get_last_conv_dim

from . import LIGHTINGNET_REGISTRY

class FCN32s(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = get_backbone(model_name)
        last_dim = get_last_conv_dim(model_name)

        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(last_dim, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x) # size = (N, 512, x.H/32, x.W/32)

        score = self.bn1(self.relu(self.deconv1(output)))     # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

# class CC_FCN4(nn.Module):
#     def __init__(self, model_name):
#         super().__init__()
#         self.fcn = FCN32s(model_name, 4)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         score = self.fcn(x)
# #        score = self.relu(score)
#         rgb = self.relu(score[:,:3,:,:])
#         #rgb = F.normalize(rgb, p=2, dim=1)
#         _, _, h, w = score.shape
#         confidence = score[:,3:4,:,:].view(-1, h*w)
#         confidence = F.softmax(confidence, dim=1)
#         rgb = rgb * confidence.view(-1, 1, h, w)
#         # average pool
# #        rgb = F.normalize(rgb, p=2, dim=1)
#         return rgb

class CC_Alex_FCN4(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        FC1_SIZE = 64
        FC1_KERNEL_SIZE = 6
        self.backbone = get_backbone('alexnet')
        last_conv_dim = get_last_conv_dim('alexnet')
        self.fc1 = nn.Conv2d(last_conv_dim, FC1_SIZE, kernel_size=FC1_KERNEL_SIZE, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Conv2d(FC1_SIZE, 4,  kernel_size=1, stride=1, bias=True)
        #self.fc_pool = nn.Conv2d(3, 3, kernel_size=8, padding=3, bias=False)
        self.fc_pool = nn.Conv2d(3, 3, kernel_size=8, bias=True)

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
        '''

    # transform the template
    def forward(self, x):
       return self.forward_template(x)

    # the normalized output is NOT required as we need to learn the lighting condition
    # changes in the environment
    def forward_template(self, x):
        y = self.backbone(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.relu(y)
        _, _, h, w = y.shape

        rgb = y[:, :3, :, :]
        rgb = F.normalize(rgb, p=2, dim=1)
        confidence = y[:,3:4,:,:].view(-1, h*w)
        confidence = F.softmax(confidence, dim=1)
        confidence = confidence.view(-1, 1, h, w)
#        rgb = F.adaptive_avg_pool2d(rgb, (1, 1))
 #       rgb = F.normalize(rgb, p=2, dim=1)
        rgb *= confidence
        rgb = self.relu(self.fc_pool(rgb))
        print (rgb)
        #rgb = F.normalize(rgb, p=2, dim=1)
        rgb = F.interpolate(rgb, x.size()[2:], mode='bilinear', align_corners=False)
        return rgb #, confidence

    def generate(self, src_img, frame_img):
        rgb = self.forward_template(frame_img)
        return src_img * rgb


@LIGHTINGNET_REGISTRY.register()
class CC_FCN4(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        FC1_OUTPUT_SIZE = 64
        FC1_KERNEL_SIZE = 6
        FC2_OUTPUT_SIZE = 4
        POOL_SIZE = 8
        backbone_name =config['lct_backbone']
        self.backbone = get_backbone(backbone_name)
        last_conv_dim = get_last_conv_dim(backbone_name)

        if backbone_name == 'resnet18':
            POOL_SIZE = 9

        self.fc1 = nn.Conv2d(last_conv_dim, FC1_OUTPUT_SIZE, kernel_size=FC1_KERNEL_SIZE, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Conv2d(FC1_OUTPUT_SIZE, FC2_OUTPUT_SIZE,  kernel_size=1, stride=1, bias=True)
        #self.fc_pool = nn.Conv2d(3, 3, kernel_size=8, padding=3, bias=False)
        self.fc_pool = nn.Conv2d(3, 3, kernel_size=POOL_SIZE, bias=True)

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
        '''

    # transform the template
    def forward(self, x):
       return self.forward_template(x)

    # the normalized output is NOT required as we need to learn the lighting condition
    # changes in the environment
    def forward_template(self, x):
        y = self.backbone(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.relu(y)
        _, _, h, w = y.shape

        rgb = y[:, :3, :, :]
        rgb = F.normalize(rgb, p=2, dim=1)
        confidence = y[:,3:4,:,:].view(-1, h*w)
        confidence = F.softmax(confidence, dim=1)
        confidence = confidence.view(-1, 1, h, w)
#        rgb = F.adaptive_avg_pool2d(rgb, (1, 1))
 #       rgb = F.normalize(rgb, p=2, dim=1)
        #print ('-----', x.shape, y.shape, confidence.shape, rgb.shape)
        rgb *= confidence
        rgb = self.relu(self.fc_pool(rgb))
        #rgb = F.normalize(rgb, p=2, dim=1)
        #print (rgb)
#        rgb = F.interpolate(rgb, x.size()[2:], mode='bilinear', align_corners=False)
        return rgb #, confidence

    def generate(self, src_img, frame_img):
        rgb = self.forward_template(frame_img)
        return src_img * rgb
