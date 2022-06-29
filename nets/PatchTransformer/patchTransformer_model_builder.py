import torch
from nets.PatchTransformer.patchTransformer_network import PatchTransformerNetwork
from utils.utils import fix_checkpoint_key

def get_patchTransformer_model_path(config):
    model_name = 'PT'
    model_name += '_%s' % (config['dataset'])

    if config['use_PBM']:
        model_name += '_PBM'

    model_name += '_STN' if config['learnableSTN'] else '_fixedSTN'
    #if config['adjust_patch_size']:
    #    model_name += '_APS'
    #if config['template_scaling_factor'] > 0:
    #    model_name += '_blur%d' % (config['template_scaling_factor'])
    model_name += '_' + config['loc_backbone']
    model_name += '_ds%d'% (config['loc_downsample_dim'])
    model_name += '_fc%d'% (config['loc_fc_dim'])
    model_name += '_' + config['STN']
    if config['STN'] == 'tps':
        if config['TPS_localizer'] == 'bounded_stn':
            model_name += '_bounded'
        else:
            model_name += '_unbounded'
        model_name += '%dx%d'% (config['TPS_grid'][0], config['TPS_grid'][1])

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

    model_name += '_bs%d' % (config['batch_size'])
    model_name += '_e%d' % (config['epochs'])

    return model_name

def build_patchTransformer_model(config):
    return PatchTransformerNetwork(config)

def build_patchTransformer_from_checkpoint(model, model_path):
    checkpoint = torch.load(model_path, map_location='cpu')

    pt_modules = {'PBM':model.PBM, 'STN':model.STN, 'PCT':model.PCT, 'LCT':model.LCT}

    print ('Loading weights from %s' % (model_path))
    for module_name in ['PBM', 'STN', 'PCT', 'LCT']:
        module = pt_modules[module_name]
        if module is not None:
            if checkpoint[module_name] is not None:
                try:
                    module.load_state_dict(fix_checkpoint_key((checkpoint[module_name])), strict=True)
                    print ("===== Finished loading %s module  =====" % (module_name))
                except Exception as e:
                    print (e)
                    print ("==== Woops, no %s module loaded ====" % (module_name))
            else:
                print ('==== %s module is not available ====' % (module_name))

    #print ("\nGenerator input size: {size}".format(size=model.generator_input_size))
    #if 'generator_input_size' in checkpoint and model.generator_input_size != checkpoint['generator_input_size']:
        #print ("Change generator input size from {size1} to {size2}".format(size1=model.generator_input_size, size2=checkpoint['generator_input_size']))
        #model.generator_input_size = checkpoint['generator_input_size']
     #   print ("Warnging: generator input size {size1} is different from the size {size2} in the loaded model.".format(size1=model.generator_input_size, size2=checkpoint['generator_input_size']))

    '''
    pct_key = 'PCT'
    if model.PCT is not None:
    try:
        if model.PCT is not None:
            model.PCT.load_state_dict(fix_checkpoint_key((checkpoint[pct_key])), strict=True)
            print ("===== Finished loading PCT module with key: %s =====" % (pct_key))
    except Exception as e:
        print (e)
        print ("==== Woops, no PCT module loaded ====")

    lct_key = 'LCT'
    try:
        model.LCT.load_state_dict(fix_checkpoint_key((checkpoint[lct_key])), strict=True)
        print("===== Finished loading LCT module with key:%s =====" % (lct_key))
    except Exception as e:
        print (e)
        print("==== Woops, no LCT module loaded ====")
    print ('\n')
    '''
    # now load optimizer
    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    iteration = checkpoint['iteration'] if 'iteration' in checkpoint else 0
    lr = checkpoint['lr'] if 'lr' in checkpoint else 0.0

    best_error = checkpoint['best_error'] if 'best_error' in checkpoint else 9999.0

    return model, epoch, iteration, lr, best_error
