import sys
import random
import shutil
import builtins
import os
import platform

import torch.backends.cudnn as cudnn
from torch import optim
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch.distributed as dist

from trainer.collaborative_advPatch_trainer import *
from nets.AdvPatch.advPatch_model_builder import *
from dataset.dataset_util import get_advPatch_data_loader
from utils.tools import get_config
from utils.logger import get_logger
from utils.utils import set_gradient_false
from opts import arg_parser, merge_args

#import multiprocessing

'''
def load_filter_patch(patch_path):
    patch_img = Image.open(patch_path).convert('RGB')
    return torch.nn.Parameter(transforms.ToTensor()(patch_img), requires_grad=False)
'''

if os.environ.get('WSC', None):
    os.system("taskset -p 0xfffffffffffffffffffffffffffffffffffffffffff %d > /dev/null 2>&1" % os.getpid())


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()

    config = get_config(args.config)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in config['gpu_ids']])

    if args.hostfile != '':
        curr_node_name = platform.node().split(".")[0]
        with open(args.hostfile) as f:
            nodes = [x.strip() for x in f.readlines() if x.strip() != '']
            master_node = nodes[0].split(" ")[0]
        for idx, node in enumerate(nodes):
            if curr_node_name in node:
                args.rank = idx
                break
        args.world_size = len(nodes)
        args.dist_url = f"tcp://{master_node}:10598"
    
    torch.autograd.set_detect_anomaly(True)
    '''    
    config = get_config(args.config)
    config = merge_args(args, config)
    val_list_file = config['val_list_file'] if 'val_list_file' in config else None
    val_dataset = get_advPatch_data_loader(
                data_dir=config['datadir'],
                dataset=config['dataset'], 
                image_size=(1080, 1920),
                data_type='val',
                data_list_file=val_list_file,
                use_augmentation=True,
                use_loc_net= config['use_loc_net'])
#    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                                 batch_size=config['batch_size'],
#                                                 shuffle=True,
#                                                 num_workers=config['num_workers'],
#                                                 sampler=None)
#    for i, (batch_data) in enumerate(val_loader):
#            name, img_batch, big_mask_batch, big_patch_bb, person_bb, small_human_img, \
#                small_patch_mask, small_patch_bb, patch_img = batch_data
#            print (name, img_batch.shape, big_mask_batch.shape)
    sys.exit(0)
    '''
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(None, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        print(f"Using GPU {args.gpu}")
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    config = get_config(args.config)
    config = merge_args(args, config)
    if args.distributed:
        config['gpu_ids'] = [gpu]


    # cv2 (warpPerspective) has multithreading issues. This is needed to get the
    # the dataloader work.
#    if config['dataset'] == 'color_data':
#        multiprocessing.set_start_method('spawn', force=True)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if args.distributed:
        cudnn.benchmark = config['cudnn_benchmark']
    else:
        if cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
            device_ids = list(range(len(device_ids)))
            config['gpu_ids'] = device_ids
            cudnn.benchmark = config['cudnn_benchmark']

    model_name = get_adv_model_path(config)
    checkpoint_path = os.path.join(args.logdir, model_name + args.name_suffix)

    if not os.path.exists(checkpoint_path) and args.rank == 0:
        os.makedirs(checkpoint_path)
    if args.rank == 0:
        shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
#    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    try:  # for unexpected error logging
        start_epoch = 0
        start_iteration = 0
        model_best_results = 999999.0

        # Define the model
        adv_model = build_advPatch_model(config)

        # use pretrained model
        if 'pretrained' in config:
            print("=> using pre-trained model '{}'".format(config['pretrained']))
            adv_model = build_advPatch_model_from_image_file(adv_model, config['pretrained'])

        if config['auto_resume']:
            if not os.path.isfile(config['resume']):  # do not specify the path for resume
                config['resume'] = os.path.join(checkpoint_path, config['model_checkpoint'])

        # preparation for processing
        if args.evaluate: # evaluation
            if 'pretrained' not in config:
                best_model_path = os.path.join(checkpoint_path, config['model_best'])
                print ("==> using saved best model '{}".format((best_model_path)))
                adv_model, epoch, _, _, best_det_loss = build_advPatch_model_from_checkpoint_file(adv_model, best_model_path)
                print ('Best detection loss %.4f at epoch %d' % (best_det_loss, epoch))
        else: # first training
            # load pretrained patches if needed
            if config['CL_pretrained']:
                adv_model.adv_patch_model.load_pretrained_patch(config['near_patch_path'], config['far_patch_path'])

            # for safety, freeze the patch_transformer and darknet model
            #if config['use_loc_net']:
            set_gradient_false(adv_model.patch_transformer)
            set_gradient_false(adv_model.detector.model)           

            if args.rank == 0:
                logger_mode = 'a' if os.path.isfile(config['resume']) else 'w'
                logger = get_logger(checkpoint_path, 'log.log', logger_mode)  # get logger and configure it at the first call
                command = " ".join(sys.argv)
                logger.info("\n{}".format(command))
                logger.info("{}\n".format('-----------------------'))
                logger.info("Random seed: {}".format(args.seed))
                # log the argument
                logger.info("Arguments: {}".format(args))
                # Log the configuration
                logger.info("Configuration: {}".format(config))


        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                adv_model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                # the batch size should be divided by number of nodes as well
                config['batch_size'] = int(config['batch_size'] / args.world_size)
                config['num_workers'] = int(config['num_workers'] / ngpus_per_node)

                if args.sync_bn:
                    process_group = torch.distributed.new_group(list(range(args.world_size)))
                    adv_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(adv_model, process_group)

                adv_model = torch.nn.parallel.DistributedDataParallel(adv_model, device_ids=device_ids, find_unused_parameters=True)
                #adv_model = torch.nn.parallel.DistributedDataParallel(adv_model, device_ids=device_ids)
            else:
                adv_model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                adv_model = torch.nn.parallel.DistributedDataParallel(adv_model)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            adv_model = adv_model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            # assign rank to 0
            if cuda:
                print('Move model %s to GPU...' % model_name)
                adv_model = torch.nn.DataParallel(adv_model.cuda(), device_ids=device_ids).cuda()

        model_module = adv_model.module if cuda else adv_model

        ###### freeze part of the model

        # adam scheduler
        if config['optimizer'] == 'SGD':
            optimizer = optim.SGD(model_module.adv_patch_model.learnable(), lr=config['lr'], momentum=0.9, weight_decay=0.0001)
        elif config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model_module.adv_patch_model.learnable(), lr=config['lr'], amsgrad=True)
        else:
            optimizer = optim.Adam(model_module.adv_patch_model.learnable(), lr=config['lr'], amsgrad=True)

        if config['scheduler'] == 'Cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], 0)
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config['scheduler_patience'], factor=config['scheduler_factor'])
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config['scheduler_patience'], factor=config['scheduler_factor'])

        # create trainer
        collaborative_adv_trainer = CollaborativeAdvPatchTrainer(adv_model, optimizer, scheduler, use_gpu=cuda, use_ohem=config['use_ohem'], ohem_ratio=config['ohem_ratio'])

        if not config['evaluate']:
            print('loading training data...')
#            train_dataset = Dataset_adv(config['data_path'], config['ori_imgs_path'], config['mask_path'], config['bbox_path'],
#                                    config['grid_path'], config['tps_path'], config['use_loc_net'])
            train_list_file = config['train_list_file'] if 'train_list_file' in config else None
            train_dataset = get_advPatch_data_loader(
                data_dir=config['datadir'],
                dataset=config['dataset'], 
                data_type='train',
                image_size=tuple(config['detector_input_size']),
                person_crop_size=tuple(config['generator_input_size']),
                patch_size_range=config['patch_size_range'],
                data_list_file=train_list_file,
                use_augmentation= config['use_augmentation'])
                #use_loc_net= config['use_loc_net'])

            sampler = torch.utils.data.DistributedSampler(train_dataset) if args.distributed else None
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=config['batch_size'],
                                                       shuffle=sampler is None,
                                                       num_workers=config['num_workers'],
                                                       sampler=sampler)

        print('loading validation data...')
       # val_dataset = Dataset_adv(config['val_data_path'], config['val_ori_imgs_path'], config['val_mask_path'], config['val_bbox_path'],
       #                            config['val_grid_path'], config['val_tps_path'], config['use_loc_net'])
        val_list_file = config['val_list_file'] if 'val_list_file' in config else None
        val_dataset = get_advPatch_data_loader(
                data_dir=config['datadir'],
                dataset=config['dataset'], 
                data_type='val',
                image_size=tuple(config['detector_input_size']),
                person_crop_size=tuple(config['generator_input_size']),
                patch_size_range=config['patch_size_range'],
                data_list_file=val_list_file,
                use_augmentation=False)
                #use_loc_net= config['use_loc_net'])
        #sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
        sampler = torch.utils.data.DistributedSampler(val_dataset) if args.distributed else None
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=config['batch_size'],
                                                 shuffle=False,
                                                 num_workers=config['num_workers'],
                                                 sampler=sampler)
        # preparation for processing
        if args.evaluate:  # evaluation
            # set compute dsr to True
            config['compute_dsr'] = True
            
            result_dir = None
            if config['visualize']:
                result_dir = os.path.join(checkpoint_path, 'vis_output')
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

            val_overall_loss, val_det_loss, val_tv_loss, val_det_accuracy, val_speed, speed_data_loader = \
                collaborative_adv_trainer.val(val_loader, config, result_dir=result_dir)
            print('Val: \toverall_Loss: {:4.4f}\tdet_Loss: {:4.4f}\ttv_Loss: {:4.4f}\tdet_acc: {:.3f}\tSpeed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                val_overall_loss, val_det_loss, val_tv_loss, val_det_accuracy,
                val_speed * 1000.0, speed_data_loader * 1000.0))
            return

        # resume
        if os.path.isfile(config['resume']):
            checkpoint_model_path = config['resume']
            state_dict = torch.load(checkpoint_model_path, map_location='cpu')
            print("==> using saved checkpoint model '{}".format((checkpoint_model_path)))
            model_module, epoch, iteration, _, best_det_loss = build_advPatch_model_from_checkpoint_file(model_module, checkpoint_model_path)
            print('Resume, previous detection loss %.4f at epoch %d' % (best_det_loss, epoch))
            start_epoch = epoch + 1
            start_iteration = iteration
            collaborative_adv_trainer.optimizer.load_state_dict(state_dict['optimizer'])
            collaborative_adv_trainer.scheduler.load_state_dict(state_dict['scheduler'])

        # Get the resume iteration to restart training
        val_overall_loss = None
        for epoch in range(start_epoch, config['epochs']):
            for param_group in optimizer.param_groups:
                lr_ = param_group['lr']

            train_overall_loss, train_det_loss, train_tv_loss, train_cl_loss, train_speed, speed_data_loader, num_batches =\
                collaborative_adv_trainer.train(train_loader, config, epoch, cur_iteration=start_iteration)
            train_log_msg = 'Train: [{:05d}/{:05d}]\tlr: {:.6f}\toverall_Loss: {:4.4f}\tdet_Loss: {:4.4f}\ttv_Loss: {:4.4f}\tcl_Loss: {:4.4f}\tSpeed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                epoch + 1, config['epochs'], lr_,  train_overall_loss, train_det_loss, train_tv_loss, train_cl_loss, train_speed * 1000.0, speed_data_loader * 1000.0)
            if args.rank == 0:
                logger.info(train_log_msg)

            start_iteration += num_batches

            if epoch % 1 == 0:
                val_overall_loss, val_det_loss, val_tv_loss, val_det_accuracy, val_speed, speed_data_loader = \
                    collaborative_adv_trainer.val(val_loader, config)
                val_log_msg = 'Val: [{:05d}/{:05d}]\toverall_Loss: {:4.4f}\tdet_Loss: {:4.4f}\ttv_Loss: {:4.4f}\tdet_acc: {:.3f}\tSpeed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                        epoch + 1, config['epochs'], val_overall_loss, val_det_loss, val_tv_loss, val_det_accuracy, val_speed * 1000.0, speed_data_loader * 1000.0)
                if args.rank == 0:
                    logger.info(val_log_msg)

                model_results = val_det_accuracy if config['compute_dsr'] else val_det_loss
                is_best_so_far = model_results < model_best_results
                model_best_results = min(model_results, model_best_results)

            if config['scheduler'] != 'ReduceLROnPlateau':
                scheduler.step()
            else:
                if not config['use_val_loss']:
                    scheduler.step(train_overall_loss)
                else:
                    if val_overall_loss is not None:
                        scheduler.step(val_overall_loss)

            # Save the model
            if args.rank == 0: # save once only!!!
                collaborative_adv_trainer.save_advPatch_checkpoint(
                            os.path.join(checkpoint_path, config['model_checkpoint']),
                            os.path.join(checkpoint_path, config['adv_patch_img']),
                            epoch,
                            start_iteration,
                            lr_,
                            model_best_results)

                if is_best_so_far:
                    collaborative_adv_trainer.save_advPatch_best_model(config, checkpoint_path)

    except Exception as e:  # for unexpected error logging
        print(format(e))
        raise e

if __name__ == '__main__':
    main()
