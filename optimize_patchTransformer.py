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

from dataset.dataset_util import get_advPatch_data_loader
from utils.tools import get_config
from utils.logger import get_logger
from utils.utils import set_gradient_false
from opts import arg_parser, merge_args
from nets.PatchTransformer.patchTransformer_model_builder import *
from tqdm import tqdm
from nets.AdvPatch.advPatch_util import paste_patch_to_frame
from utils.tools import AverageMeter, combine_images
from losses.mask_losses import SIMMMaskLoss, L1MaskLoss, L2MaskLoss
import torchvision.transforms as transforms
import torch.nn.functional as F


# import multiprocessing

'''
def load_filter_patch(patch_path):
    patch_img = Image.open(patch_path).convert('RGB')
    return torch.nn.Parameter(transforms.ToTensor()(patch_img), requires_grad=False)
'''

def paste_scaled_patch_to_frame(patch, patch_bb, img, img_bb, scale):
    n, c, _, _ = patch.shape
    # create tensor

    resize_patch_bb = patch_bb.float() * scale
    resize_patch_bb = resize_patch_bb.int()

    img_h, img_w = img.shape[2:]
    x = torch.cuda.FloatTensor(n, c, img_h, img_w).fill_(0)
    for i, bbox in enumerate(resize_patch_bb):
        resized_patch = F.interpolate(patch[i].unsqueeze(0), scale_factor=scale[i].item(),
                                     mode='bilinear', align_corners=False)
        pb, pl, ph, pw = bbox
        ib, il, ih, iw = img_bb[i]
        resized_tmpl = F.interpolate(resized_patch[:, :, pb:pb + ph, pl:pl + pw], size=(ih, iw),
                                     mode='bilinear', align_corners=False)
        x[i, :, ib:ib + ih, il:il + iw] = resized_tmpl.squeeze()

    return x

def visualize_transformation(names, lct_imgs, imgs, person_bb, saved_path):
        for j in range(len(names)):
            saved_name = os.path.basename(names[j])
            person_img = transforms.ToPILImage()(imgs[j].detach().cpu())

            res_img = transforms.ToPILImage()(lct_imgs[j].detach().cpu())
            bb = person_bb[j].detach().cpu()
            bb = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
            target_img = combine_images([person_img.crop(bb), res_img.crop(bb)])

            target_img.save(os.path.join(saved_path, saved_name + '.jpg'))

if os.environ.get('WSC', None):
    os.system("taskset -p 0xfffffffffffffffffffffffffffffffffffffffffff %d > /dev/null 2>&1" % os.getpid())


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()

    config = get_config(args.config)

    config = get_config(args.config)
    config = merge_args(args, config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = config['cudnn_benchmark']
    
    pt_model = PatchTransformerNetwork(config)
    pt_model, _, _, _, _ = build_patchTransformer_from_checkpoint(pt_model, config['pretrained'])

    print('Move model %s to GPU...' % os.path.basename(config['pretrained']))
    pt_model = pt_model.cuda()
    pt_model = torch.nn.DataParallel(pt_model, device_ids=device_ids)

    pt_model.eval()

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
            # use_loc_net= config['use_loc_net'])

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                       batch_size=config['batch_size'],
                                                       shuffle= False,
                                                       num_workers=config['num_workers'],
                                                       sampler=None)

    
    generator_loss = L1MaskLoss()
    #generator_loss = SIMMMaskLoss(val_range=1)

    best_loss = 99999.0
    best_alpha = 0.0
    best_beta = 0.0
    generator_input_size = config['generator_input_size']
    for alpha in [1,2,3,4,5,6,7,8,9,10, 11, 12]:
        for beta in [0.1, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]:
    #for alpha in [2]:
    #    for beta in [0.6]:
            avg_transform_loss = AverageMeter()
            with torch.no_grad(), tqdm(total=len(val_loader)) as t_bar:
                for i, (batch_data) in enumerate(val_loader):
                    name, img_batch, big_mask_batch, big_patch_bb, person_bb, small_human_img, \
                    small_patch_mask, small_patch_bb, patch_img, template_img, patch_scale = batch_data

                    img_batch, big_mask_batch = img_batch.cuda(), big_mask_batch.cuda()
                    person_bb, big_patch_bb, patch_img = person_bb.cuda(), big_patch_bb.cuda(), patch_img.cuda()
                    small_human_img, small_patch_mask, small_patch_bb = small_human_img.cuda(), small_patch_mask.cuda(), small_patch_bb.cuda()
                    template_img, patch_scale = template_img.cuda(), patch_scale.cuda()

                    resize_scale = beta / (1.0 + torch.exp(-alpha * patch_scale.view(-1, 1)))
                    resize_scale = torch.clamp(resize_scale, max=1.0)

                    ###### Forward pass ######
                    output = pt_model(patch_img, small_patch_bb, small_patch_mask, template_img, small_human_img,
                                      resize_scale)
                    transformed_patch = output['LCT'][0]

                    # print (small_patch_bb)
                    #adv_batch = paste_scaled_patch_to_frame(transformed_patch, small_patch_bb, img_batch, big_patch_bb,
                    #                                        resize_scale)
                    adv_batch = paste_patch_to_frame(transformed_patch, small_patch_bb, img_batch, big_patch_bb)
                    adv_batch = adv_batch * big_mask_batch + (1 - big_mask_batch) * img_batch

                    transform_loss = generator_loss(img_batch, adv_batch, big_patch_bb)
                    avg_transform_loss.update(transform_loss.item(), img_batch.size(0))
                    if args.visualize:
                        visualize_transformation(name, adv_batch, img_batch, person_bb, './tmp')
                    t_bar.update(1)
                if best_loss > avg_transform_loss.avg:
                    best_loss = avg_transform_loss.avg
                    best_alpha = alpha
                    best_beta = beta
                print (generator_input_size, alpha, beta, avg_transform_loss.avg, best_alpha, best_beta, best_loss)


'''
    best_loss = 99999.0
    best_alpha = 0.0
    best_beta = 0.0
    #for alpha in [2, 3,4,5,6,7,8,9,10, 11, 12]:
    #    for beta in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]:
    for alpha in [7]:
        for beta in [1.5]:
            avg_transform_loss = AverageMeter()
            with torch.no_grad(), tqdm(total=len(val_loader)) as t_bar:
                for i, (batch_data) in enumerate(val_loader):
                    name, img_batch, big_mask_batch, big_patch_bb, person_bb, small_human_img, \
                    small_patch_mask, small_patch_bb, patch_img, template_img,scale_factor, patch_scale = batch_data

                    img_batch, big_mask_batch = img_batch.cuda(), big_mask_batch.cuda()
                    person_bb, big_patch_bb, patch_img = person_bb.cuda(), big_patch_bb.cuda(), patch_img.cuda()
                    small_human_img, small_patch_mask, small_patch_bb = small_human_img.cuda(), small_patch_mask.cuda(), small_patch_bb.cuda()
                    template_img, patch_scale, scale_factor = template_img.cuda(), patch_scale.cuda(), scale_factor.cuda()

                    ###### Forward pass ######
                    output = pt_model(patch_img, small_patch_bb, small_patch_mask, template_img, small_human_img, scale_factor)
                    transformed_patch = output['LCT'][0]

                    resize_scale = beta / (1.0 + torch.exp(-alpha * patch_scale.view(-1, 1)))
                    resize_scale = torch.clamp(resize_scale, max=1.0)
        
                    #small_patch_bb = small_patch_bb.float() * resize_scale
                    #small_patch_bb = small_patch_bb.int()
             
                    #print (small_patch_bb)
                    adv_batch = paste_scaled_patch_to_frame(transformed_patch, small_patch_bb, img_batch, big_patch_bb, resize_scale)
                    adv_batch = adv_batch * big_mask_batch + (1 - big_mask_batch) * img_batch

                    transform_loss = generator_loss(img_batch, adv_batch, big_patch_bb)
                    avg_transform_loss.update(transform_loss.item(), img_batch.size(0))
                    #print (transform_loss.data)

                    #visualize_transformation(name, adv_batch, img_batch, person_bb, './tmp')
                    t_bar.update(1)
                if best_loss > avg_transform_loss.avg:
                    best_loss = avg_transform_loss.avg
                    best_alpha = alpha
                    best_beta = beta
                print (generator_input_size, alpha, beta, avg_transform_loss.avg, best_alpha, best_beta, best_loss)
'''

if __name__ == '__main__':
    main()
