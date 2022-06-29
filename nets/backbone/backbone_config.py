from torch import nn
import torchvision.models as models
from functools import partial

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def dilated_resnet(depth, num_classes=1000, pretrained: bool = False):
    block = models.resnet.BasicBlock if depth < 50 else models.resnet.Bottleneck
    layers = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]}[depth]

    model = models.ResNet(block, layers, num_classes=num_classes, replace_stride_with_dilation=[True, True, True])
    if pretrained:
        state_dict = load_state_dict_from_url(models.resnet.model_urls[f'resnet{depth}'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


backbone_info = {
    'resnet18': {'model': models.resnet18, 'last_conv_dim': 512},
    'resnet50': {'model': models.resnet50, 'last_conv_dim': 2048},
    'resnet101': {'model': models.resnet101, 'last_conv_dim': 2048},
    'alexnet': {'model': models.alexnet, 'last_conv_dim': 256},
    'vgg11_bn': {'model': models.vgg11_bn, 'last_conv_dim': 512},
    'vgg11': {'model': models.vgg11, 'last_conv_dim': 512},
    'vgg19_bn': {'model': models.vgg19_bn, 'last_conv_dim': 512},
    'vgg19': {'model': models.vgg19, 'last_conv_dim': 512},
    'dresnet18': {'model': partial(dilated_resnet, depth=18), 'last_conv_dim': 512},
    'dresnet50': {'model': partial(dilated_resnet, depth=50), 'last_conv_dim': 512}
}


def get_backbone(name: str, pretrained: bool = True) -> nn.Module:
    model = backbone_info[name]['model'](num_classes=1000, pretrained=pretrained)
    return nn.Sequential(*list(model.children())[0:-2])


def get_last_conv_dim(name: str) -> int:
    return backbone_info[name]['last_conv_dim']
