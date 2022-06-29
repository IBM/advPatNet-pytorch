import torch
from torchvision import transforms
from PIL import Image
from nets.AdvPatch.advPatch_net import AdvPatchNet
from nets.AdvPatch.collaborative_advPatch_net import CollaborativeAdvPatchNet
from nets.AdvPatch.hybrid_advPatch import HybridAdvPatch
from utils.tools import normalize
import os

def get_adv_model_path(config):
    model_name = config['detector_name']
    model_name += '_' + config['object_dataset']
    if config['collaborative_learning'] is True:
        model_name += '_CL'
        if config['half_patches'] is True:
            model_name += 'half'
        if config['collaborative_weight']:
            model_name += '_weighted'
        if config['CL_pretrained'] is True:
            model_name += '_pretrained'
    model_name += '_adv%d' % (config['adv_patch_size'][0])

    model_name += '_' + config['dataset']
#    if config['use_augmentation']:
#        model_name += '_aug'
    if config['use_ohem']:
        model_name += '_ohem%d' % (int(config['ohem_ratio']*100))
     #   model_name += '_ohem'
    if config['apply_border_mask']:
        model_name += '_border'

    #model_name += '_STN' if config['learnableSTN'] else '_fixedSTN'

#    model_name += '_p%d' % (config['person_crop_size'][0])
    #if config['template_scaling_factor'] > 0:
    #    model_name += '_blur%d' % (config['template_scaling_factor'])
    #model_name += '_' + config['loc_backbone']
    #model_name += '_ds%d'% (config['loc_downsample_dim'])
    #model_name += '_fc%d'% (config['loc_fc_dim'])
    model_name += '_' + config['STN']
    #if config['STN'] == 'tps':
    #    if config['TPS_localizer'] == 'bounded_stn':
    #        model_name += '_bounded'
    #    else:
    #        model_name += '_unbounded'
    #    model_name += '%dx%d'% (config['TPS_grid'][0], config['TPS_grid'][1])

    if config['use_PCT']:
        model_name += '_' + config['PrinterCT']

    if config['use_LCT']:
        model_name += '_' + config['LightingCT']
        if config['LightingCT'] == 'cc':
            model_name += '_' + config['lct_backbone']
        model_name += '_p%d' % (config['lct_input_size'][0])

    if config['use_LCT']:
        model_name += '_' + config['LCT_loss']
    elif config['use_PCT']:
        model_name += '_' + config['PCT_loss']

    if config['use_EOT']:
        model_name += '_EOT'

#    model_name += '_%s_loss' % (config['MaxProbExtractor_loss'])
    model_name += '_tv%d' % (int(config['tv_loss_weight']*10))
    model_name += '_bs%d' % (config['batch_size'])
    model_name += '_e%d' % (config['epochs'])

    return model_name

def build_advPatch_model(config):
    return AdvPatchNet(config) if not config['collaborative_learning'] else \
       CollaborativeAdvPatchNet(config)

def build_advPatch_model_from_checkpoint_file(model, model_path):
    print ('loading advPatch model: %s' % (model_path))
    checkpoint = torch.load(model_path, map_location='cpu')
    y1 = torch.sigmoid(checkpoint['adv_patch']['blending'])
    import torchvision.transforms as transforms
    img = transforms.ToPILImage()(y1)
    img.save('blend_mask.png')
    print (y1)

    model.adv_patch_model.load_state_dict(checkpoint['adv_patch'])
    return model, checkpoint['epoch'], checkpoint['iteration'], checkpoint['lr'], checkpoint['best_error']

def build_advPatch_model_from_image_file(model, model_path):
    model.adv_patch_model.load_patch(model_path)
    return model

#def set_gradient_false(model):
#    for p in model.parameters():
#        p.requires_grad = False
