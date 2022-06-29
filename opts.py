import argparse

from nets.LightingNet import LIGHTINGNET_REGISTRY

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action recognition Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
    parser.add_argument('--seed', type=int, help='manual seed')
    parser.add_argument('--dataset', help='path to dataset file list')
    parser.add_argument('--datadir', metavar='DIR', help='path to dataset file list')
    parser.add_argument('--logdir', dest='logdir', help='where to save the model')
    parser.add_argument('--train_list_file', type=str, help='training file')
    parser.add_argument('--val_list_file', type=str, help='validation file')
    parser.add_argument('--no_flip', dest='no_flip', action='store_true', help='do not flip data')
    parser.add_argument('--template_resize', dest='template_resize', action='store_true', help='resize template')
    parser.add_argument('--mask_loss', dest='mask_loss', action='store_true', help='use L1 masked loss')
    
    parser.add_argument('--loc_backbone', dest='loc_backbone', choices=['resnet18', 'resnet50', 'resnet101'],  help='which backbone to use')

    parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to the model for resuming')
    parser.add_argument('--auto_resume', action='store_true', help='use the last checkpoint in the logdir for resume')
    parser.add_argument('--pretrained', dest='pretrained', type=str, metavar='PATH', help='use pre-trained model')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='use this flag to validate without training')
    parser.add_argument('--batch_size', type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--disable_cudnn_benchmark', dest='cudnn_benchmark', action='store_false',
                        help='Disable cudnn to search the best mode (avoid OOM)')
    parser.add_argument('--optimizer', type=str, help='optimizer (Default: Adam)', choices=['Adam', 'SGD'], default='Adam')
    parser.add_argument('--scheduler', type=str, help='Learning Rate scheduler (Default: ReduceLROnPlateau)', choices=['ReduceLROnPlateau', 'Cosine'], default='ReduceLROnPlateau')
    parser.add_argument('--use_val_loss', action='store_true', help='When using ReduceLROnPlateau, use val loss to change learning rate')
    parser.add_argument('--name_suffix', type=str, help='suffix of model name, used for creating log folder', default='')
    parser.add_argument('--gpu', dest='gpu_ids', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--compute_dsr', action='store_true', help='Compute detection successful rate in validation, it will be automatically turned on if evaluate flag is existed.')
    parser.add_argument('--obj_loss_type', type=str, default='max', choices=['max', 'avg', 'ce'], help='different way to compute obj loss')
    parser.add_argument('--show_dsr_hist', action='store_true', help='Show the histogram of detection acc w.r.t. the height of person.')

    # data-related
    parser.add_argument('-j', '--num_workers', type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', type=float, metavar='N', help='learning rate')

    parser.add_argument('--use_PBM', dest='use_PBM', action='store_true', help='use patch blurring')

    parser.add_argument('--STN', type=str, help='affine or tps')
    parser.add_argument('--learnableSTN', dest='learnableSTN', action='store_true', help='learn STN')
    parser.add_argument('--TPS_localizer', dest='TPS_localizer', type=str, help='tps localizer')

    parser.add_argument('--use_PCT', dest='use_PCT', action='store_true', help='use lighting color transformation')
    parser.add_argument('--PrinterCT', type=str, help='PCT or LinearPCT')

    parser.add_argument('--use_LCT', dest='use_LCT', action='store_true', help='use lighting color transformation')
    parser.add_argument('--LightingCT', type=str, help='cc (color constancy) or gen (image generator)', choices=LIGHTINGNET_REGISTRY._obj_map.keys())
    parser.add_argument('--lct_backbone', type=str, help='set the backbone of lightning net', default=None)

    parser.add_argument('--target_patch_path', dest='target_patch_path', type=str, help='target patch to be transformed')

    parser.add_argument('--patch_transformer_path', dest='patch_transformer_path', type=str, help='stn model')

    parser.add_argument('--tv_loss_weight', type=float, metavar='N', help='tv_loss_weight range[0,10]')

    #parser.add_argument('--use_augmentation', dest='use_augmentation', action='store_true', help='use augmentation')
    parser.add_argument('--use_ohem', dest='use_ohem', action='store_true', help='use ohem')
    parser.add_argument('--ohem_ratio', type=float, metavar='N', help='ohem ratio [0.1-1.0]')
    parser.add_argument('--use_EOT', dest='use_EOT', action='store_true', help='use augmentation')

#    parser.add_argument('--MaxProbExtractor_loss', dest='MaxProbExtractor_loss', type=str, help='type of max prob extractor')
    
    parser.add_argument('--visualize', dest='visualize', action='store_true', help='store adversarial images')
    parser.add_argument('--test_dir', dest='test_dir',  type=str, help='test directory with images')
    parser.add_argument('--detection_output_dir', dest='detection_output_dir',  type=str, help='output directory')
    
    parser.add_argument('--detector_impl', dest='detector_impl', type=str, help='implementation')
    parser.add_argument('--detector_name', dest='detector_name', type=str, help='detector name')
    parser.add_argument('--object_dataset', dest='object_dataset', type=str, help='object dataset: COCO or PASCAL')

    parser.add_argument('--collaborative_learning', '--CL',  action='store_true', help='collaborative learning')
    parser.add_argument('--CL_pretrained', '--CLPretrain',  action='store_true', help='use pretrained models collaborative learning')
    parser.add_argument('--collaborative_weights', '--CW', action='store_true', help='using learnable weights in collaborative learning ')
    parser.add_argument('--kd_norm', type=float, metavar='N', help='margin loss norm')
    parser.add_argument('--kd_type', type=str, metavar='N', help='loss type: margin (proposed) | mutual | one')

    # for distributed learning
    parser.add_argument('--sync-bn', action='store_true',
                        help='sync BN across GPUs')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--hostfile', default='', type=str,
                        help='hostfile distributed learning')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', '--ddp', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    return parser

def merge_args(args, config):
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    return config
