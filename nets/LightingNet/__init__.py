from typing import Dict, Any

import torch.nn as nn
from fvcore.common.registry import Registry

LIGHTINGNET_REGISTRY = Registry('LIGHTINGNET')
LIGHTINGNET_REGISTRY.__doc__ = """
Registry for lightning model.
The registered object will be called with `obj(config)`.
The call should return a `torch.nn.Module` object.
"""


def lighting_net_builder(config: Dict[str, Any]) -> nn.Module:

    return LIGHTINGNET_REGISTRY.get(config['LightingCT'])(config)


from .cc_drn import CCDRN
from .cc_f4 import CC_FCN4
from .fine_generator import Generator
from .cc_gen import CCGenerator


# TODO: support deprecated name
LIGHTINGNET_REGISTRY._do_register('cc', CC_FCN4)
LIGHTINGNET_REGISTRY._do_register('gen', Generator)
