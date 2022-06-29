"""
Training code for Adversarial patch training

"""
import os
import time
import shutil

import torch
from torch import nn
import torch.distributed as dist
from tqdm import tqdm

from utils.tools import AverageMeter, combine_images
import torchvision.transforms as transforms

class PatchTransformerTrainer(nn.Module):
    def __init__(self, model, optimizer, scheduler, use_gpu=True):
        super(PatchTransformerTrainer, self).__init__()
        self.model = model
        self.use_gpu = use_gpu
        self.optimizer = optimizer
        self.scheduler = scheduler

    def get_model(self):
        return self.model.module if self.use_gpu else self.model

    def train(self, data_loader, config, epoch, cur_iteration=0):
        # set training mode
        self.get_model().train()

        if torch.distributed.is_initialized():
            data_loader.sampler.set_epoch(epoch)
            rank = dist.get_rank()
        else:
            rank = 0

        batch_time = AverageMeter()
        data_time = AverageMeter()
        avg_stn_losses = AverageMeter()
        avg_pct_losses = AverageMeter()
        avg_lct_losses = AverageMeter()
        avg_overall_losses = AverageMeter()

        # Train for single eopch
        end = time.time()
        num_batch = 0
        with tqdm(total=len(data_loader), disable=rank != 0) as t_bar:
            for i, (batch_data) in enumerate(data_loader):
                data_time.update(time.time() - end)
                names, patch_img, small_patch_mask, template_img, crop_person_img, small_mask_bb, scale_factor = batch_data

                if self.use_gpu:
                    patch_img, small_patch_mask, template_img = patch_img.cuda(), small_patch_mask.cuda(), template_img.cuda()
                    crop_person_img, small_mask_bb, scale_factor = crop_person_img.cuda(), small_mask_bb.cuda(), scale_factor.cuda()

                # total number of iterations
                cur_iteration = cur_iteration + 1

                ###### Forward pass ######
                output = self.model(patch_img, small_mask_bb, small_patch_mask, template_img, crop_person_img, scale_factor)

                stn_loss = output['STN'][1]
                pct_loss = output['PCT'][1]
                lct_loss = output['LCT'][1]
                stn_loss = torch.mean(stn_loss)
                pct_loss = torch.mean(pct_loss)
                lct_loss = torch.mean(lct_loss)
                #overall_loss = stn_loss + pct_loss + lct_loss
                #overall_loss = pct_loss + lct_loss
                overall_loss = lct_loss

                self.optimizer.zero_grad()
                overall_loss.backward()
                self.optimizer.step()

                if torch.distributed.is_initialized():
                    world_size = float(torch.distributed.get_world_size())
                    stn_loss_ = stn_loss.detach()
                    pct_loss_ = pct_loss.detach()
                    lct_loss_ = lct_loss.detach()
                    overall_loss_ = overall_loss.detach()
                    torch.distributed.all_reduce(stn_loss_)
                    torch.distributed.all_reduce(pct_loss_)
                    torch.distributed.all_reduce(lct_loss_)
                    torch.distributed.all_reduce(overall_loss_)
                    stn_loss_ /= world_size
                    pct_loss_ /= world_size
                    lct_loss_ /= world_size
                    overall_loss_ /= world_size
                    batch_size = patch_img.size(0) * world_size
                    avg_stn_losses.update(stn_loss_.item(), batch_size)
                    avg_pct_losses.update(pct_loss_.item(), batch_size)
                    avg_lct_losses.update(lct_loss_.item(), batch_size)
                    avg_overall_losses.update(overall_loss_.item(), batch_size)
                else:
                    batch_size = patch_img.size(0)
                    avg_stn_losses.update(stn_loss.item(), batch_size)
                    avg_pct_losses.update(pct_loss.item(), batch_size)
                    avg_lct_losses.update(lct_loss.item(), batch_size)
                    avg_overall_losses.update(overall_loss.item(), batch_size)

                batch_time.update(time.time() - end)
                end = time.time()

                if i + 1 % config['print_iter'] == 0 and rank == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'overall-Loss {overall_loss.val:.4f} ({overall_loss.avg:.4f})\t'
                          'STN-Loss {stn_loss.val:.4f} ({stn_loss.avg:.4f})\t'
                          'PCT-Loss {pct_loss.val:.4f} ({pct_loss.avg:.4f})\t'
                          'LCT-Loss {lct_loss.val:.4f} ({lct_loss.avg:.4f})\t'.format(
                        epoch, i, len(data_loader), batch_time=batch_time,
                        data_time=data_time, overall_loss=avg_overall_losses, stn_loss=avg_stn_losses,
                        pct_loss=avg_pct_losses, lct_loss=avg_lct_losses))

                num_batch += 1
                t_bar.set_description(
                    desc=f"Epoch [{epoch}], All_L: {avg_overall_losses.avg:.4f}, STN_L: {avg_stn_losses.avg:.4f}, PCT_L: {avg_pct_losses.avg:.4f}, LCT_L: {avg_lct_losses.avg:.4f}")
                t_bar.update(1)

        return avg_overall_losses.avg, avg_stn_losses.avg, avg_pct_losses.avg, avg_lct_losses.avg, batch_time.avg, data_time.avg, num_batch

    def val(self, data_loader, config, result_dir=None):
        # set eval mode
        self.get_model().eval()

        rank = dist.get_rank() if torch.distributed.is_initialized() else 0

        dsr = 0.
        batch_time = AverageMeter()
        data_time = AverageMeter()
        avg_stn_losses = AverageMeter()
        avg_pct_losses = AverageMeter()
        avg_lct_losses = AverageMeter()
        avg_overall_losses = AverageMeter()

        # Train for single eopch
        end = time.time()

        with torch.no_grad(), tqdm(total=len(data_loader), disable=rank != 0) as t_bar:
            for i, (batch_data) in enumerate(data_loader):
                data_time.update(time.time() - end)
                names, patch_img, small_patch_mask, template_img, crop_person_img, small_mask_bb, scale_factor = batch_data

                if self.use_gpu:
                    patch_img, small_patch_mask, template_img = patch_img.cuda(), small_patch_mask.cuda(), template_img.cuda()
                    crop_person_img, small_mask_bb, scale_factor = crop_person_img.cuda(), small_mask_bb.cuda(), scale_factor.cuda()

                ###### Forward pass ######
                output = self.model(patch_img, small_mask_bb, small_patch_mask, template_img, crop_person_img, scale_factor)

                stn_loss = output['STN'][1]
                pct_loss = output['PCT'][1]
                lct_loss = output['LCT'][1]
                stn_loss = torch.mean(stn_loss)
                pct_loss = torch.mean(pct_loss)
                lct_loss = torch.mean(lct_loss)
                #overall_loss = stn_loss + pct_loss + lct_loss
                #overall_loss = pct_loss + lct_loss
                overall_loss = lct_loss

                if torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size()
                    torch.distributed.all_reduce(stn_loss)
                    torch.distributed.all_reduce(pct_loss)
                    torch.distributed.all_reduce(lct_loss)
                    torch.distributed.all_reduce(overall_loss)
                    stn_loss /= world_size
                    pct_loss /= world_size
                    lct_loss /= world_size
                    overall_loss /= world_size
                    batch_size = patch_img.size(0) * world_size
                else:
                    batch_size = patch_img.size(0)

                avg_stn_losses.update(stn_loss.item(), batch_size)
                avg_pct_losses.update(pct_loss.item(), batch_size)
                avg_lct_losses.update(lct_loss.item(), batch_size)
                avg_overall_losses.update(overall_loss.item(), batch_size)

                batch_time.update(time.time() - end)
                end = time.time()
                t_bar.set_description(
                    desc=f"All_L: {avg_overall_losses.avg:.4f}, STN_L: {avg_stn_losses.avg:.4f}, PCT_L: {avg_pct_losses.avg:.4f}, LCT_L: {avg_lct_losses.avg:.4f}")
                t_bar.update(1)

                if config['evaluate'] and config['visualize']:
                    stn_imgs = output['STN'][0]
                    pct_imgs = output['PCT'][0]
                    lct_imgs = output['LCT'][0]
                    self.visualize_transformation(names, stn_imgs, pct_imgs, lct_imgs, crop_person_img, result_dir)

        return avg_overall_losses.avg, avg_stn_losses.avg, avg_pct_losses.avg, avg_lct_losses.avg, batch_time.avg, data_time.avg


    def save_patchTransformer_checkpoint(self, model_path, epoch, iteration, lr, best_error):
        checkpoint = {}
        model = self.get_model()
        checkpoint['STN'] = model.STN.state_dict()
        checkpoint['PBM'] = model.PBM.state_dict() if model.PBM is not None else None
        checkpoint['PCT'] = model.PCT.state_dict() if model.PCT is not None else None
        checkpoint['LCT'] = model.LCT.state_dict() if model.LCT is not None else None
        #checkpoint['generator_input_size'] = model.generator_input_size
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['lr'] = lr
        checkpoint['iteration'] = iteration
        checkpoint['best_error'] = best_error

        torch.save(checkpoint, model_path)

    def save_patchTransformer_best_model(self, config, model_path):
        shutil.copyfile(os.path.join(model_path, config['model_checkpoint']),
                os.path.join(model_path, config['model_best']))

    @torch.no_grad()
    def visualize_transformation(self, names, stn_imgs, pct_imgs, lct_imgs, person_imgs, saved_path):
        for j in range(len(names)):
            saved_name = os.path.basename(names[j])
            person_img = transforms.ToPILImage()(person_imgs[j].detach().cpu())

            h, w = person_img.size
            cut_size = int (w * 0.33)
            crop_bb = (cut_size, 0, w - cut_size, h)
            target_img = person_img.crop(crop_bb)

            for item in [lct_imgs, pct_imgs, stn_imgs]:
                if item is not None:
                    res_img = transforms.ToPILImage()(item[j].detach().cpu())
                    target_img = combine_images([target_img, res_img.crop(crop_bb)])

            target_img.save(os.path.join(saved_path, saved_name + '.jpg'))
