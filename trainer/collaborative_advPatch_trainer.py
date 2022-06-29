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

from utils.tools import AverageMeter
from utils.utils import detection_accuracy
from losses.ohem_loss import Adv_OHEM


class CollaborativeAdvPatchTrainer(nn.Module):
    def __init__(self, model, optimizer, scheduler, use_gpu=True, use_ohem=False, ohem_ratio=0.75):
        super(CollaborativeAdvPatchTrainer, self).__init__()
        self.model = model
        self.use_gpu = use_gpu
        self.optimizer = optimizer
        self.scheduler = scheduler
  
        self.hard_mining = use_ohem
        if self.hard_mining:
            self.ohem_loss = Adv_OHEM(ohem_ratio)

        # adam scheduler
        #self.optimizer = optim.Adam([self.model.adv_patch_cpu], lr=self.config['lr'], amsgrad=True)
        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.5)

    def get_model(self):
        return self.model.module if self.use_gpu else self.model

    def train(self, data_loader, config, epoch, cur_iteration=0):

        if torch.distributed.is_initialized():
            data_loader.sampler.set_epoch(epoch)
            rank = dist.get_rank()
        else:
            rank = 0

        # training mode
#        self.model.train()
        self.get_model().set_train_mode()
#        mask = genearte_mask()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        avg_tv_losses = AverageMeter()
        avg_det_losses = AverageMeter()
        avg_cl_losses = AverageMeter()
        avg_overall_losses = AverageMeter()

        # Train for single eopch
        end = time.time()
        num_batch = 0
        tv_loss_weight = config['tv_loss_weight']
        with tqdm(total=len(data_loader), disable=rank != 0) as t_bar:
            for i, (batch_data) in enumerate(data_loader):
                data_time.update(time.time() - end)
                name, img_batch, big_mask_batch, big_patch_bb, person_bb, small_human_img, \
                    small_patch_mask, small_patch_bb, patch_img, template_img, patch_size_info = batch_data

                if self.use_gpu:
                    img_batch, big_mask_batch = img_batch.cuda(), big_mask_batch.cuda()
                    person_bb, big_patch_bb, patch_img = person_bb.cuda(), big_patch_bb.cuda(), patch_img.cuda()
                    small_human_img, small_patch_mask, small_patch_bb = small_human_img.cuda(), small_patch_mask.cuda(), small_patch_bb.cuda()
                    patch_size_info = patch_size_info.cuda()

                ''' 
                for ii in range(img_batch.shape[0]):
                  train_img = transforms.ToPILImage()(img_batch[ii].detach().cpu())
                  train_img.save(os.path.join('tmp', name[ii]+'.jpg'))
                '''

                # total number of iterations
                cur_iteration = cur_iteration + 1

                ###### Forward pass ######
                _, det_loss, tv_loss, cl_loss = self.model(img_batch, big_mask_batch, big_patch_bb, person_bb, small_human_img, small_patch_mask, \
                                                          small_patch_bb, patch_img, patch_size_info)
                tv_loss = torch.mean(tv_loss)
                tv_loss = tv_loss * tv_loss_weight

                #print (output, coord_w_set)
                #det_loss = torch.mean(det_loss)
                det_loss = self.ohem_loss(det_loss) if self.hard_mining else torch.mean(det_loss)
                cl_loss = torch.mean(cl_loss)
                #cl_loss *= 5.0

                overall_loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) + cl_loss

                '''
                ep_det_loss += det_loss.detach().cpu().numpy()
                # ep_nps_loss += nps_loss.detach().cpu().numpy()
                ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_loss += loss
                '''
                self.optimizer.zero_grad()
                overall_loss.backward()
                self.optimizer.step()

                if torch.distributed.is_initialized():
                    world_size = float(torch.distributed.get_world_size())
                    tv_loss_ = tv_loss.detach()
                    det_loss_ = det_loss.detach()
                    cl_loss_ = cl_loss.detach()
                    overall_loss_ = overall_loss.detach()
                    torch.distributed.all_reduce(tv_loss_)
                    torch.distributed.all_reduce(det_loss_)
                    torch.distributed.all_reduce(overall_loss_)
                    tv_loss_ /= world_size
                    det_loss_ /= world_size
                    cl_loss_ /= world_size
                    overall_loss_ /= world_size
                    batch_size = img_batch.size(0) * world_size
                    avg_tv_losses.update(tv_loss_.item(), batch_size)
                    avg_det_losses.update(det_loss_.item(), batch_size)
                    avg_cl_losses.update(cl_loss_.item(), batch_size)
                    avg_overall_losses.update(overall_loss_.item(), batch_size)
                else:
                    batch_size = img_batch.size(0)
                    avg_tv_losses.update(tv_loss.item(), batch_size)
                    avg_det_losses.update(det_loss.item(), batch_size)
                    avg_cl_losses.update(cl_loss.item(), batch_size)
                    avg_overall_losses.update(overall_loss.item(), batch_size)

                # clip
#                self.get_model().advT_patch.advT_patch.data.clamp_(0, 1)  # keep patch in image range
                #self.get_model().adv_patch_cpu.data = self.get_model().adv_patch_cpu.data * mask
                self.get_model().adv_patch_model.clip()

                # debug
                '''
                for p in self.get_model().darknet_model.parameters():
                   if p.grad is not None:
                       print ('============')
                       print (torch.sum(p.grad))
                       print ('============')
                       break
                for k, p in enumerate(self.get_model().patch_transformer.parameters()):
                   print (torch.sum(p.data))
                   if k > 2:
                       break
                   if p.grad is not None:
                       print ('============')
                       print (torch.sum(p.grad))
                       print ('============')
                       break
                for p in self.get_model().color_trans.parameters():
                   if p.grad is not None:
                       print ('============')
                       print (torch.sum(p.grad))
                       print ('============')
                       break
                '''
                batch_time.update(time.time() - end)
                end = time.time()

                if i+1 % config['print_iter'] == 0 and rank == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'overall-Loss {overall_loss.val:.4f} ({overall_loss.avg:.4f})\t'
                          'det-Loss {det_loss.val:.4f} ({det_loss.avg:.4f})\t'
                          'tv-Loss {tv_loss.val:.4f} ({tv_loss.avg:.4f})\t'
                          'cl-Loss {cl_loss.val:.4f} ({cl_loss.avg:.4f})\t'.format(
                        epoch, i, len(data_loader), batch_time=batch_time,
                        data_time=data_time, overall_loss=avg_overall_losses, det_loss=avg_det_losses, tv_loss=avg_tv_losses, cl_loss=avg_cl_losses))

                num_batch += 1
                t_bar.set_description(desc=f"Epoch [{epoch}], All_L: {avg_overall_losses.avg:.4f}, Det_L: {avg_det_losses.avg:.4f}, Tv_L: {avg_tv_losses.avg:.4f}")
                t_bar.update(1)

        return avg_overall_losses.avg, avg_det_losses.avg, avg_tv_losses.avg, avg_cl_losses.avg, batch_time.avg, data_time.avg, num_batch


    def val(self, data_loader, config, result_dir=None):
        # eval mode
#        self.model.eval()
        rank = dist.get_rank() if torch.distributed.is_initialized() else 0
        self.get_model().set_eval_mode()

        dsr = None
        batch_time = AverageMeter()
        data_time = AverageMeter()
        avg_tv_losses = AverageMeter()
        avg_det_losses = AverageMeter()
        avg_overall_losses = AverageMeter()
        histogram = {}

        # Train for single eopch
        end = time.time()
        tv_loss_weight = config['tv_loss_weight']
        val_conf_thresh = config['val_conf_thresh']
        val_nms_thresh = config['val_nms_thresh']
        val_iou_threshold = config['val_iou_threshold']
        target_object_id = self.get_model().detector.target_object_id
        with torch.no_grad(), tqdm(total=len(data_loader), disable=rank != 0) as t_bar:
            for i, (batch_data) in enumerate(data_loader):
                data_time.update(time.time() - end)
                name, img_batch, big_mask_batch, big_patch_bb, person_bb, small_human_img, \
                    small_patch_mask, small_patch_bb, patch_img, template_img, patch_size_info = batch_data

                if self.use_gpu:
                    img_batch, big_mask_batch = img_batch.cuda(), big_mask_batch.cuda()
                    person_bb, big_patch_bb, patch_img = person_bb.cuda(), big_patch_bb.cuda(), patch_img.cuda()
                    small_human_img, small_patch_mask, small_patch_bb = small_human_img.cuda(), small_patch_mask.cuda(), small_patch_bb.cuda()
                    patch_size_info = patch_size_info.cuda()
                ###### Forward pass ######
                adv_batch, det_loss, tv_loss = self.model(img_batch, big_mask_batch, big_patch_bb, person_bb, small_human_img, small_patch_mask, \
                                                          small_patch_bb, patch_img, patch_size_info)

                tv_loss = torch.mean(tv_loss)
                tv_loss = tv_loss * tv_loss_weight

                det_loss = torch.mean(det_loss)

                overall_loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                if torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size()
                    torch.distributed.all_reduce(tv_loss)
                    torch.distributed.all_reduce(det_loss)
                    torch.distributed.all_reduce(overall_loss)
                    tv_loss /= world_size
                    det_loss /= world_size
                    overall_loss /= world_size
                    batch_size = img_batch.size(0) * world_size
                else:
                    batch_size = img_batch.size(0)

                avg_tv_losses.update(tv_loss.item(), batch_size)
                avg_det_losses.update(det_loss.item(), batch_size)
                avg_overall_losses.update(overall_loss.item(), batch_size)
                
                if config['compute_dsr']:
                    #tmp = detection_accuracy(self.get_model().detector, adv_batch, name, person_bb, val_nms_thresh, val_conf_thresh,
                    #                         val_iou_threshold, target_object_id, result_dir, config['show_dsr_hist'])
                    tmp = detection_accuracy(self.get_model().detector, adv_batch, name, person_bb, big_patch_bb, val_nms_thresh, val_conf_thresh,
                                             val_iou_threshold, target_object_id, result_dir, config['show_dsr_hist'])
                    dsr = torch.cat([dsr, tmp], dim=0) if dsr is not None else tmp
                  #dsr = None
                  #self.visualize_transformation(name, adv_batch, img_batch, person_bb, result_dir)

                batch_time.update(time.time() - end)
                end = time.time()
                t_bar.set_description(desc=f"AllL: {avg_overall_losses.avg:.4f}, DetL: {avg_det_losses.avg:.4f}, TvL: {avg_tv_losses.avg:.4f}")
                t_bar.update(1)

            bin_size = 25
            if dsr is not None:
                if config['show_dsr_hist']:
                    dsr_ = dsr[:len(data_loader.dataset)][:, 0]
                    _dsr = dsr[:len(data_loader.dataset)].cpu().numpy()
                    for correct, score, dis in zip(_dsr[:,0], _dsr[:,1], _dsr[:, 2]):
                        #quantized_dis = int(dis // 50)
                        quantized_dis = int(dis // bin_size)
                        if quantized_dis not in histogram:
                            histogram[quantized_dis] = [0, 0, 0]
                        histogram[quantized_dis][0] += correct
                        histogram[quantized_dis][1] += score
                        histogram[quantized_dis][2] += 1
                    print("========================================", flush=True)
                    print("Person Heights (Pixels),Detection Acc.,Ave Score,# of Data ", flush=True)
                    for k in sorted(histogram):
                        print(f"{k * bin_size:04d}-{(k+1)*bin_size:04d},{histogram[k][0] / histogram[k][2] * 100:4.2f}%,{histogram[k][1] / histogram[k][2] * 100:4.2f}%,{histogram[k][2]}" , flush=True)
                    print("========================================", flush=True)
                    dsr = torch.mean(dsr_)
                else:
                    dsr = torch.mean(dsr[:len(data_loader.dataset)])
            else:
                dsr = 0.0
        #print ('person_crop_size (%d %d)  AVG transform loss %f' % (config['person_crop_size'][0], config['person_crop_size'][1], avg_transform_loss.avg))
        return avg_overall_losses.avg, avg_det_losses.avg, avg_tv_losses.avg, dsr, batch_time.avg, data_time.avg


    def save_advPatch_checkpoint(self, model_path, patch_path, epoch, iteration, lr, best_error):
        checkpoint = {}
#        advT_patch = self.get_model().advT_patch.detach().cpu()
#        checkpoint['advT_patch'] = advT_patch.numpy()
        checkpoint['adv_patch'] = self.get_model().adv_patch_model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['scheduler'] = self.scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['iteration'] = iteration
        checkpoint['lr'] = lr
        checkpoint['best_error'] = best_error

        checkpoint['patch_transformer'] = self.get_model().patch_transformer.state_dict()
        checkpoint['detector'] = self.get_model().detector.state_dict()
        torch.save(checkpoint, model_path)

        self.get_model().adv_patch_model.save_patch(patch_path)

    def save_advPatch_best_model(self, config, model_path):
        shutil.copyfile(os.path.join(model_path, config['model_checkpoint']),
                        os.path.join(model_path, config['model_best']))
        shutil.copyfile(os.path.join(model_path, config['adv_patch_img']),
                        os.path.join(model_path, config['adv_patch_img_best']))

        if self.get_model().collaborative_learning:
            base_fname, ext = config['adv_patch_img'].split('.')
            near_fname = os.path.join(model_path, base_fname + '_near' + '.' + ext)
            best_near_fname = os.path.join(model_path, 'best_' + base_fname + '_near' + '.' + ext)
            shutil.copy(near_fname, best_near_fname)
            far_fname = os.path.join(model_path, base_fname + '_far' + '.' + ext)
            best_far_fname = os.path.join(model_path, 'best_' + base_fname + '_far' + '.' + ext)
            shutil.copy(far_fname, best_far_fname)
