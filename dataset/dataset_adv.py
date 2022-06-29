import sys
from copy import deepcopy

import cv2
import torch.utils.data as data
from os import listdir
from utils.tools import default_loader, is_image_file, normalize, mask2bbox
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import collections, glob, json
import math

def flip_image(img, flip_method):
    return img if flip_method < 0 else img.transpose(flip_method)

def flip_bbox( bb, img_h, img_w, flip_method):
    # bb: y,x, h, w
    if flip_method == Image.FLIP_LEFT_RIGHT:
        return (bb[0], img_w - bb[1] - bb[3],  bb[2], bb[3])

    if flip_method == Image.FLIP_TOP_BOTTOM:
        return (img_h - bb[0] - bb[2], bb[1], bb[2], bb[3])

    return bb

# to be implemented
def flip_coord(bb, img_h, img_w, flip_method):
    # bb: [x1, y1], [x2, y2]
    if flip_method == Image.FLIP_LEFT_RIGHT:
        w = bb[1, 0] - bb[0, 0] + 1
        xmin = img_w - bb[0, 0] - w
        xmax = xmin + w
        ymin = bb[0, 1]
        ymax = bb[1, 1]
        if len(bb) == 3: # w
            new_bb = [[xmin, ymin], [xmax, ymax], [xmin, ymax]]
        elif len(bb) == 4: # w
            new_bb = [[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]]
        else:
            raise NotImplementedError

        return new_bb

    if flip_method == Image.FLIP_TOP_BOTTOM:
        h = bb[1, 1] - bb[0, 1] + 1
        ymin = img_h - bb[0, 1] - h
        ymax = ymin + h
        xmin = bb[0, 0]
        xmax = bb[1, 0]
        if len(bb) == 3:  # w
            new_bb = [[xmin, ymin], [xmax, ymax], [xmin, ymax]]
        elif len(bb) == 4:  # w
            new_bb = [[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]]
        else:
            raise NotImplementedError
        return new_bb


class Dataset_advPatch(data.Dataset):
    def __init__(self, file_list_path, img_path, mask_path, patch_path, image_size=(1080, 1920), person_crop_size=(256, 256), patch_size_range=None, use_augmentation=True):
        super(Dataset_advPatch, self).__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.patch_path = patch_path
        self.image_size = image_size
        self.person_crop_size = person_crop_size
        self.use_augmentation = use_augmentation
        self.patch_size_range = patch_size_range 
        assert patch_size_range is None or patch_size_range[1] > patch_size_range[0]

        data = np.loadtxt(file_list_path, dtype=str)
        self.img_file_list = data[:,0].tolist()
        self.patch_file_list = data[:, 1].tolist()
        bbox = data[:, 1:].astype(np.int32) if data.shape[1] == 5 else data[:, 2:].astype(np.int32)
        bbox[bbox<0] = 0
        self.adv_person_bbox = bbox

        '''
        for i in range(len(self.img_file_list)):
            if self.adv_person_bbox[i][0] < -20 or self.adv_person_bbox[i][1] < -20:
                self.visualize(i)
        '''
    @staticmethod
    def _pad_resize_img(img, size, mode=Image.BILINEAR):
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(0, 0, 0))
                padded_img.paste(img, (int(padding), 0))
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(0, 0, 0))
                padded_img.paste(img, (0, int(padding)))

        padded_img = padded_img.resize(size, mode)
        return padded_img

    @staticmethod
    def _yxhw_to_x1y1x2y2(b):
        return [b[1], b[0], b[1]+b[3]-1, b[0]+b[2]-1] 

    @staticmethod
    def _x1y1x2y2_to_yxhw(b):
        return [b[1], b[0], b[3]-b[1]+1, b[2]-b[0]+1]

    @staticmethod
    def _resize_box(box, from_size, to_size):
        rw = np.float(to_size[0]) / from_size[0]
        rh = np.float(to_size[1]) / from_size[1]
        box[0] *= rw
        box[1] *= rh
        box[2] *= rw
        box[3] *= rh
        return [round(item) for item in box]

    def visualize(self, index):
        # original image
        filename = self.img_file_list[index]
        img_path = os.path.join(self.img_path, filename)
        frame_img = Image.open(img_path)

        # mask image
        mask_path = os.path.join(self.mask_path, filename.replace('.jpg', '.png'))
        big_patch_mask = Image.open(mask_path)

        # bounding box of the person wearing advT
        person_bb = self.adv_person_bbox[index, ...]

        # flip images for data augmentation
        self.patch_path = patch_path
        is_flip = self.use_augmentation is True and np.random.rand() > 0.5
        if is_flip:
            pi_w, pi_h = frame_img.size
            frame_img = flip_image(frame_img, Image.FLIP_LEFT_RIGHT)
            big_patch_mask = flip_image(big_patch_mask, Image.FLIP_LEFT_RIGHT)
            person_bb = flip_bbox(self._x1y1x2y2_to_yxhw(person_bb), pi_h, pi_w, Image.FLIP_LEFT_RIGHT)
            person_bb = self._yxhw_to_x1y1x2y2(person_bb)

        # bounding box of the mask
        big_mask_bb = mask2bbox(big_patch_mask)
        big_mask_bb = self._yxhw_to_x1y1x2y2(big_mask_bb)

        # person crop
        crop_person_img = frame_img.crop(tuple(person_bb))
        #padd and resize it to 256x256
        crop_person_img = self._pad_resize_img(crop_person_img, self.person_crop_size)
        # mask crop
        small_patch_mask = big_patch_mask.crop(tuple(person_bb))
        small_patch_mask = self._pad_resize_img(small_patch_mask, self.person_crop_size, mode=Image.NEAREST)
        small_mask_bb = mask2bbox(small_patch_mask)
        small_mask_bb = self._yxhw_to_x1y1x2y2(small_mask_bb)

        # resize it to 224x224
        patch_img = frame_img.crop(tuple(big_mask_bb))
        patch_img = patch_img.resize((224, 224), Image.BILINEAR)

        # resize the original image if needed
        frm_size = frame_img.size
        if frm_size != self.image_size:
            frame_img = frame_img.resize(self.image_size, Image.BILINEAR)
            big_patch_mask = big_patch_mask.resize(self.image_size, Image.BILINEAR)
            big_mask_bb = mask2bbox(big_patch_mask)
            big_mask_bb = self._yxhw_to_x1y1x2y2(big_mask_bb)
            person_bb = self._resize_box(person_bb, frm_size, self.image_size)

        frame_img1 = ImageDraw.Draw(frame_img)
        frame_img1.rectangle(tuple(person_bb), outline='red',width=2)
        frame_img1.rectangle(tuple(big_mask_bb), outline='yellow',width=2)
        big_patch_mask1 = ImageDraw.Draw(big_patch_mask)
        big_patch_mask1.rectangle(tuple(big_mask_bb), outline='red',width=2)
        small_patch_mask1 = ImageDraw.Draw(small_patch_mask)
        small_patch_mask1.rectangle(tuple(small_mask_bb), outline='red',width=2)
        crop_person_img1 = ImageDraw.Draw(crop_person_img)
        crop_person_img1.rectangle(tuple(small_mask_bb), outline='red',width=2)
        filename = os.path.basename(filename)
        frame_img.save('tmp/frame_%s' % (filename))
        big_patch_mask.save('tmp/big_patch_mask_%s' % (filename))
        small_patch_mask.save('tmp/small_patch_mask_%s' % (filename))
        crop_person_img.save('tmp/person_%s' % (filename))
        patch_img.save('tmp/patch_%s' % (filename))

    def __getitem__(self, index):
        # original image
        filename = self.img_file_list[index]
        img_path = os.path.join(self.img_path, filename)
        frame_img = Image.open(img_path)

        # mask image
        mask_path = os.path.join(self.mask_path, filename.replace('.jpg', '.png'))
        big_patch_mask = Image.open(mask_path)

        # template_image
        template_img = Image.open(os.path.join(self.patch_path, self.patch_file_list[index]))

        # bounding box of the person wearing advT
        person_bb = self.adv_person_bbox[index, ...]

        # flip images for data augmentation
        is_flip = self.use_augmentation is True and np.random.rand() > 0.5
        if is_flip:
            pi_w, pi_h = frame_img.size
            frame_img = flip_image(frame_img, Image.FLIP_LEFT_RIGHT)
            big_patch_mask = flip_image(big_patch_mask, Image.FLIP_LEFT_RIGHT)
            template_img = flip_image(template_img, Image.FLIP_LEFT_RIGHT)
            person_bb = flip_bbox(self._x1y1x2y2_to_yxhw(person_bb), pi_h, pi_w, Image.FLIP_LEFT_RIGHT)
            person_bb = self._yxhw_to_x1y1x2y2(person_bb)

        # bounding box of the mask
        big_mask_bb = mask2bbox(big_patch_mask)
#        big_mask_bb = self._yxhw_to_x1y1x2y2(big_mask_bb)
        #print (filename, big_mask_bb[2], flush=True)

        # person crop
        crop_person_img = frame_img.crop(tuple(person_bb))
        #padd and resize it to 256x256
        crop_person_img = self._pad_resize_img(crop_person_img, self.person_crop_size)
        # mask crop
        small_patch_mask = big_patch_mask.crop(tuple(person_bb))
        small_patch_mask = self._pad_resize_img(small_patch_mask, self.person_crop_size, mode=Image.NEAREST)
        small_mask_bb = mask2bbox(small_patch_mask)
 #       small_mask_bb = self._yxhw_to_x1y1x2y2(small_mask_bb)

        # resize it to 224x224
        patch_img = frame_img.crop(tuple(self._yxhw_to_x1y1x2y2(big_mask_bb)))
        patch_img = patch_img.resize((224, 224), Image.BILINEAR)

        # resize the original image if needed
        frm_size = frame_img.size
        if frm_size != self.image_size:
            frame_img = frame_img.resize(self.image_size, Image.BILINEAR)
            big_patch_mask = big_patch_mask.resize(self.image_size, Image.BILINEAR)
            big_mask_bb = mask2bbox(big_patch_mask)
#            big_mask_bb = self._yxhw_to_x1y1x2y2(big_mask_bb)
            person_bb = self._resize_box(person_bb, frm_size, self.image_size)

        #normlization: [0 1]
        patch_height = big_mask_bb[2]

        #if patch_height > 150:
        #    print (filename, self.patch_file_list[index], person_bb[0], person_bb[1], person_bb[2], person_bb[3])

        frame_img = transforms.ToTensor()(frame_img)
        big_patch_mask = transforms.ToTensor()(big_patch_mask)
        template_img = transforms.ToTensor()(template_img)

        # while training PTnet requires input within [-1 1], applying PTNet an image doesn't require so
        # as STN is inferred from the patch directly
        crop_person_img = transforms.ToTensor()(crop_person_img)
        small_patch_mask = transforms.ToTensor()(small_patch_mask)
        patch_img = transforms.ToTensor()(patch_img)
        person_bb = torch.FloatTensor(person_bb)
        big_mask_bb = torch.IntTensor(big_mask_bb)
        small_mask_bb = torch.IntTensor(small_mask_bb)

        #scale_factor = torch.FloatTensor([(person_bb[3] - person_bb[1] + 1.0) / self.person_crop_size[0]])
        #scale_factor = self.person_crop_size[1] / (person_bb[3] - person_bb[1] + 1.0)
        #scale_factor = torch.FloatTensor([scale_factor])


        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        

        if self.patch_size_range is not None:
            patch_size_info = (patch_height - self.patch_size_range[0] + 1.0) / (self.patch_size_range[1] - self.patch_size_range[0] + 1.0)
            patch_size_info = min(max(patch_size_info,0.0), 1.0)
            #print (filename, patch_height, patch_size_info, flush=True)
            patch_size_info = torch.tensor(patch_size_info, dtype=torch.float32)
            return filename, frame_img, big_patch_mask, big_mask_bb, person_bb, crop_person_img, small_patch_mask, small_mask_bb, patch_img, template_img, patch_size_info
        else:
            return filename, frame_img, big_patch_mask, big_mask_bb, person_bb, crop_person_img, small_patch_mask, small_mask_bb, patch_img, template_img

    def __len__(self):
       return len(self.img_file_list)

class Dataset_advPatch_old(data.Dataset):
    def __init__(self, data_path, file_list_path, mask_path, bbox_path, grid_path,  tps_path=None, use_augmentation=False, use_loc_net=False, image_size=(1920, 1080)):
        self.image_size = image_size

        super(Dataset_advPatch, self).__init__()

        self.use_augmentation = use_augmentation

        dict1 = self.import_names_dict(data_path)
#        used_names = self.import_txt(file_list_path)
#        used_names = sorted(used_names)
        self.data_info = self.import_txt(file_list_path)
        used_names = sorted(list(self.data_info.keys()))

        used_dict = {}
        for i in used_names:
            try:
                used_dict[i] = dict1[i]
            except KeyError:
                continue

        coord_w_set = self._import_json(bbox_path)
        coord_r_set = self._import_json(grid_path)

        w_k = set(coord_w_set.keys())
        r_k = set(coord_r_set.keys())
        i_k = set(used_dict.keys())

        common = set(w_k & r_k & i_k)
        common = sorted(list(common))
        self.coord_w_set, self.coord_r_set, self.imgs_set, = [], [], []
        for i in common:
            self.coord_w_set.append(coord_w_set[i])
            self.coord_r_set.append(coord_r_set[i])
            self.imgs_set.append(used_dict[i])

            x = np.array(coord_w_set[i]) / 416. * 1920.
            x[:, 0] = x[:, 0] - (1920-1080)//2

            x = [x[0][0], x[0][1], x[1][0], x[1][1]]
            x = [int(round(item)) for item in x]
            fname = os.path.basename(used_dict[i])
            print (os.path.join(fname[:8], fname), x[0], x[1], x[2], x[3])

        #print (len(person_bbox))
        #with open('person_val.json', 'w', encoding='utf-8') as f:
        #    json.dump(person_bbox, f, ensure_ascii=False, indent=4)
        # self.mask_init = np.ones((self.image_size, self.image_size, 3)) * 255

        sys.exit(0)
        print('---- {} frames used'.format(len(common)))
        self.common = common
        # self.used_dict = used_dict

        # self.__getitem__(5)

#    @staticmethod
#    def import_txt(path):
#        with open(path) as f:
#            fnames = f.read().splitlines()

#        fnames = [i[9:-4] for i in fnames]
#        return fnames

    @staticmethod
    def import_txt(path):
        data = np.loadtxt(path, dtype=str)
        output = {}
        for row in data:
            fname, patch_size_label = row
            output[fname[9:-4]] = patch_size_label
        return output

    @staticmethod
    def import_names_dict(rootpath):
        files = os.listdir(rootpath)
        names_dict = {}
        for floder in files:
            for img_name in os.listdir(os.path.join(rootpath, floder)):
                if img_name.endswith('.png') or img_name.endswith('.jpg'):
                    names_dict[img_name.split('.')[0]] = os.path.join(rootpath, floder, img_name)
                    # img = Image.open(os.path.join(rootpath, floder, img_name))

        return names_dict

    def _pad_corrd(self, c):
        # actually no pad here
        coord = deepcopy(c)
        if coord.ndim == 3:
            # from 540*960 to self.image_size*self.image_size
            off_x = 0 #(960 - 540) / 2
            coord[:, :, 0] = coord[:, :, 0] + off_x
            coord[:, :, 0] = coord[:, :, 0] / 540 * self.image_size
            coord[:, :, 1] = coord[:, :, 1] / 960 * self.image_size
        if coord.ndim == 2:
            # from 540*960 to self.image_size*self.image_size
            off_x = 0 # (960 - 540) / 2
            coord[:, 0] = coord[:, 0] + off_x
            coord[:, 0] = coord[:, 0] / 540 * self.image_size
            coord[:, 1] = coord[:, 1] / 960 * self.image_size
        elif coord.ndim == 1:
            # from 960*540 to self.image_size*self.image_size
            off_x = 0 #(960 - 540) / 2
            coord[1] = coord[1] + off_x
            coord[0] = coord[0] / 960 * self.image_size
            coord[1] = coord[1] / 540 * self.image_size
        else:
            raise RuntimeError("Unsupported dimension: {}".format(coord.ndim))

        return coord.astype(np.int32)

    @staticmethod
    def _import_json(path):
        result = {}
        file_list = sorted(glob.glob(path + '/*.json'))
        # print('{} json files are found'.format(len(file_list)))
        for i in range(len(file_list)):
            with open(file_list[i]) as json_file:
                j = json.load(json_file)
                if isinstance(j, list):
                    for d in j:
                        result.update(d)
                else:
                    result.update(j)
        result = collections.OrderedDict(sorted(result.items()))
        return result

    @staticmethod
    def _pad_resize_img(img, size):
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(0, 0, 0))
                padded_img.paste(img, (int(padding), 0))
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(0, 0, 0))
                padded_img.paste(img, (0, int(padding)))
        resize = transforms.Resize(size)
        padded_img = resize(padded_img)
        return padded_img

    @staticmethod
    def _resize_img(img, size):
        resize = transforms.Resize(size)
        resized_img = resize(img)
        return resized_img

    def _wide_coord(self, coord):
        coord[:, 0] = ((coord[:, 0] / 416 * 1920.) - ((1920 - 1080) // 2))/1080. * self.image_size[1]
        coord[:, 1] = coord[:, 1] / 416. * self.image_size[0]
        return coord

    def __getitem__(self, index):
        img_path = self.imgs_set[index]
        #print(index, img_path)
        name = img_path.split('/')[-1][:-4]
        ori_img = Image.open(img_path)

        resized_img = ori_img  # self._resize_img(ori_img, size=self.image_size)

        coord_red_ori = np.array(self.coord_r_set[index])
        coord_w_ori = np.array(self.coord_w_set[index])

        # coord_red_ori = self._pad_corrd(coord_red_ori/2.)

        # #####
        # M2 = cv2.getPerspectiveTransform(np.array([[0, 0], [self.image_size-1, 0], [0, self.image_size-1], [self.image_size-1, self.image_size-1]], dtype=np.float32),
        #                                  np.array([coord_red_ori[0], coord_red_ori[1], coord_red_ori[2],
        #                                            coord_red_ori[3]], dtype=np.float32)
        #                                  )
        #
        # mask_iregular = np.uint8(cv2.warpPerspective(self.mask_init, M2, (self.image_size, self.image_size)))
        # mask_iregular[mask_iregular > 100] = 255
        # mask_iregular[mask_iregular <= 100] = 0
        # mask_iregular = Image.fromarray(mask_iregular)
        # if not os.path.exists(img_path[:-38] + '600x600_wide_masks/' + name[:-5]):
        #     os.mkdir(img_path[:-38] + '600x600_wide_masks/' + name[:-5])
        #
        # mask_iregular.save(img_path[:-38] + '600x600_wide_masks/' + name[:-5] + '/' + name + '.png')
        # return 0

        if self.image_size == 416:
            mask_iregular = Image.open(img_path[:-38] + 'wide_masks/' + name[:-5] + '/' + name + '.png')
        elif self.image_size == 600:
            mask_iregular = Image.open(img_path[:-38] + '600x600_wide_masks/' + name[:-5] + '/' + name + '.png')
        elif self.image_size == (1920, 1080):
            mask_iregular = Image.open(img_path[:-38] + 'ori_masks/' + name[:-5] + '/' + name + '.png')
        else:
            raise NotImplementedError

        bb = mask2bbox(mask_iregular)

        # small image
        w = coord_w_ori / 416. * 1920.
        w[:, 0] = w[:, 0] - (1920-1080)//2
        w = np.round(w).astype(np.int32)
        small_human_img = ori_img.crop((*w[0], *w[1]))
        small_human_img = self._pad_resize_img(small_human_img, size=(256, 256))
        small_mask_img = Image.open(img_path[:-38] + 'small_masks/' + name[:-5] + '/' + name + '.png')
        # small_mask_img = mask_iregular.crop((*coord_w_ori[0], *coord_w_ori[1]))
        # small_mask_img = self._pad_resize_img(small_mask_img, size=(256, 256))

        small_bb = mask2bbox(small_mask_img)
        crop_bb = (*np.min(np.array(((self.coord_r_set[index]))), 0), *np.max(np.array(((self.coord_r_set[index]))), 0))
        crop_img = ori_img.crop(crop_bb)
        crop_img = crop_img.resize((224, 224), Image.BILINEAR)  # resize 224 x 224 (ResNet input)

        coord_w_ori = self._wide_coord(coord_w_ori)
        coord_w_set_plus = np.expand_dims(np.array([coord_w_ori[0, 0], coord_w_ori[1, 1]]), 0)
        coord_w_ori = np.concatenate((coord_w_ori, coord_w_set_plus), 0)  # 0: upper_left, 1:lower_right, 2: lower_left

        if self.use_augmentation is True:
            # flip or not
            if np.random.rand() > 0.5:
                resized_img = flip_image(resized_img, Image.FLIP_LEFT_RIGHT)
                crop_img = flip_image(crop_img, Image.FLIP_LEFT_RIGHT)
                mask_iregular = flip_image(mask_iregular, Image.FLIP_LEFT_RIGHT)
                pi_w, pi_h = resized_img.size
                assert pi_w == self.image_size[1] and pi_h == self.image_size[0]
                coord_red_ori = flip_coord(coord_red_ori, pi_h, pi_w, Image.FLIP_LEFT_RIGHT) # the entire image
                coord_w_ori = flip_coord(coord_w_ori, pi_h, pi_w, Image.FLIP_LEFT_RIGHT) #the entire image
                bb = flip_bbox(bb, pi_h, pi_w, Image.FLIP_LEFT_RIGHT)

                small_human_img = flip_image(small_human_img, Image.FLIP_LEFT_RIGHT)
                small_mask_img = flip_image(small_mask_img, Image.FLIP_LEFT_RIGHT)

                sh_w, sh_h = small_human_img.size # the padded person image
                small_bb = flip_bbox(small_bb, sh_h, sh_w, Image.FLIP_LEFT_RIGHT)
                if 1:
                    the_small_bb = mask2bbox(small_mask_img)
                    the_bb = mask2bbox(mask_iregular)
                    assert bb == the_bb, (bb, the_bb)
                    assert small_bb == the_small_bb, (small_bb, the_small_bb)

        #[0 1]
        resized_img = transforms.ToTensor()(resized_img)
        #[-1 1]
        crop_img = transforms.ToTensor()(crop_img)
        mask_iregular = transforms.ToTensor()(mask_iregular)
        coord_red_ori = torch.tensor(coord_red_ori, dtype=torch.float32)
        coord_w_ori = torch.tensor([coord_w_ori[0][0], coord_w_ori[0][1], coord_w_ori[1][0], coord_w_ori[1][1]], dtype=torch.float32)
        small_human_img = transforms.ToTensor()(small_human_img)
        small_mask_img = transforms.ToTensor()(small_mask_img)
        bb = torch.IntTensor(bb)
        small_bb = torch.IntTensor(small_bb)

        # patch size
        patch_size_level = int(self.data_info[name])

        return resized_img, mask_iregular, bb, crop_img, name, coord_w_ori, coord_red_ori, small_human_img, small_mask_img, small_bb, torch.tensor(patch_size_level, dtype=torch.float32)

    def __len__(self):
       return len(self.common)

class Dataset_advT_color(data.Dataset):
    def __init__(self, data_path, file_list_path, mask_path, bbox_path, grid_path,  tps_path=None, use_augmentation=False, use_loc_net=False, image_size=416):
        self.image_size = image_size

        super(Dataset_advT_color, self).__init__()

        self.use_augmentation = use_augmentation

        dict1 = self.import_names_dict(data_path)
#        used_names = self.import_txt(file_list_path)
        self.data_info = self.import_txt(file_list_path)
        used_names = sorted(list(self.data_info.keys()))
        used_dict = {}
        for i in used_names:
            try:
                used_dict[i] = dict1[i]
            except KeyError:
                continue

        coord_w_set = self._import_json(bbox_path)
        coord_r_set = self._import_json(grid_path)

        w_k = set(coord_w_set.keys())
        r_k = set(coord_r_set.keys())
        i_k = set(used_dict.keys())

        common = set(w_k & r_k & i_k)
        common = sorted(list(common))
        self.coord_w_set, self.coord_r_set, self.imgs_set, = [], [], []
        for i in common:
            self.coord_w_set.append(coord_w_set[i])
            self.coord_r_set.append(coord_r_set[i])
            self.imgs_set.append(used_dict[i])

        self.mask_init = np.ones((self.image_size, self.image_size, 3)) * 255

        print('---- {} frames used'.format(len(common)))
        self.common = common
        # self.used_dict = used_dict

        # self.__getitem__(5)

#    @staticmethod
#    def import_txt(path):
#        with open(path) as f:
#            fnames = f.read().splitlines()

#        fnames = [i[9:-4] for i in fnames]
#        return fnames

    @staticmethod
    def import_txt(path):
        data = np.loadtxt(path, dtype=str)
        output = {}
        for row in data:
            fname, patch_size_label = row
            output[fname[9:-4]] = patch_size_label
        return output

    @staticmethod
    def import_names_dict(rootpath):
        files = os.listdir(rootpath)
        names_dict = {}
        for floder in files:
            for img_name in os.listdir(os.path.join(rootpath, floder)):
                if img_name.endswith('.png') or img_name.endswith('.jpg'):
                    names_dict[img_name.split('.')[0]] = os.path.join(rootpath, floder, img_name)
                    # img = Image.open(os.path.join(rootpath, floder, img_name))

        return names_dict

    def _pad_corrd(self, c):
        # actually no pad here
        coord = deepcopy(c)
        if coord.ndim == 3:
            # from 540*960 to self.image_size*self.image_size
            off_x = 0 #(960 - 540) / 2
            coord[:, :, 0] = coord[:, :, 0] + off_x
            coord[:, :, 0] = coord[:, :, 0] / 540 * self.image_size
            coord[:, :, 1] = coord[:, :, 1] / 960 * self.image_size
        if coord.ndim == 2:
            # from 540*960 to self.image_size*self.image_size
            off_x = 0 # (960 - 540) / 2
            coord[:, 0] = coord[:, 0] + off_x
            coord[:, 0] = coord[:, 0] / 540 * self.image_size
            coord[:, 1] = coord[:, 1] / 960 * self.image_size
        elif coord.ndim == 1:
            # from 960*540 to self.image_size*self.image_size
            off_x = 0 #(960 - 540) / 2
            coord[1] = coord[1] + off_x
            coord[0] = coord[0] / 960 * self.image_size
            coord[1] = coord[1] / 540 * self.image_size
        else:
            raise RuntimeError("Unsupported dimension: {}".format(coord.ndim))

        return coord.astype(np.int32)

    @staticmethod
    def _import_json(path):
        result = {}
        file_list = sorted(glob.glob(path + '/*.json'))
        # print('{} json files are found'.format(len(file_list)))
        for i in range(len(file_list)):
            with open(file_list[i]) as json_file:
                j = json.load(json_file)
                if isinstance(j, list):
                    for d in j:
                        result.update(d)
                else:
                    result.update(j)
        result = collections.OrderedDict(sorted(result.items()))
        return result

    @staticmethod
    def _pad_resize_img(img, size):
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(0, 0, 0))
                padded_img.paste(img, (int(padding), 0))
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(0, 0, 0))
                padded_img.paste(img, (0, int(padding)))
        resize = transforms.Resize(size)
        padded_img = resize(padded_img)
        return padded_img

    @staticmethod
    def _resize_img(img, size):
        resize = transforms.Resize(size)
        resized_img = resize(img)
        return resized_img

    def _wide_coord(self, coord):
        coord[:, 0] = ((coord[:, 0] / 416 * 1920.) - ((1920 - 1080) // 2))/1080. * self.image_size
        if self.image_size != 416:
            coord[:, 1] = coord[:, 1] / 416. * self.image_size
        return coord

    def __getitem__(self, index):
        img_path = self.imgs_set[index]
        name = img_path.split('/')[-1][:-4]
        ori_img = Image.open(img_path)

        resized_img = self._resize_img(ori_img, size=(self.image_size, self.image_size))

        coord_red_ori = np.array(self.coord_r_set[index])
        coord_w_ori = np.array(self.coord_w_set[index])

        coord_red_ori = self._pad_corrd(coord_red_ori/2.)

        # #####
        # M2 = cv2.getPerspectiveTransform(np.array([[0, 0], [self.image_size-1, 0], [0, self.image_size-1], [self.image_size-1, self.image_size-1]], dtype=np.float32),
        #                                  np.array([coord_red_ori[0], coord_red_ori[1], coord_red_ori[2],
        #                                            coord_red_ori[3]], dtype=np.float32)
        #                                  )
        #
        # mask_iregular = np.uint8(cv2.warpPerspective(self.mask_init, M2, (self.image_size, self.image_size)))
        # mask_iregular[mask_iregular > 100] = 255
        # mask_iregular[mask_iregular <= 100] = 0
        # mask_iregular = Image.fromarray(mask_iregular)
        # if not os.path.exists(img_path[:-38] + '600x600_wide_masks/' + name[:-5]):
        #     os.mkdir(img_path[:-38] + '600x600_wide_masks/' + name[:-5])
        #
        # mask_iregular.save(img_path[:-38] + '600x600_wide_masks/' + name[:-5] + '/' + name + '.png')
        # return 0

        if self.image_size == 416:
            mask_iregular = Image.open(img_path[:-38] + 'wide_masks/' + name[:-5] + '/' + name + '.png')
        elif self.image_size == 600:
            mask_iregular = Image.open(img_path[:-38] + '600x600_wide_masks/' + name[:-5] + '/' + name + '.png')
        else:
            raise NotImplementedError

        bb = mask2bbox(mask_iregular)

        # small image
        w = coord_w_ori / 416. * 1920.
        w[:, 0] = w[:, 0] - (1920-1080)//2
        w = np.round(w).astype(np.int32)
        small_human_img = ori_img.crop((*w[0], *w[1]))
        small_human_img = self._pad_resize_img(small_human_img, size=(256, 256))
        small_mask_img = Image.open(img_path[:-38] + 'small_masks/' + name[:-5] + '/' + name + '.png')
        # small_mask_img = mask_iregular.crop((*coord_w_ori[0], *coord_w_ori[1]))
        # small_mask_img = self._pad_resize_img(small_mask_img, size=(256, 256))

        small_bb = mask2bbox(small_mask_img)
        crop_bb = (*np.min(np.array(((self.coord_r_set[index]))), 0), *np.max(np.array(((self.coord_r_set[index]))), 0))
        crop_img = ori_img.crop(crop_bb)
        crop_img = crop_img.resize((224, 224), Image.BILINEAR)  # resize 224 x 224 (ResNet input)

        coord_w_ori = self._wide_coord(coord_w_ori)
        coord_w_set_plus = np.expand_dims(np.array([coord_w_ori[0, 0], coord_w_ori[1, 1]]), 0)
        coord_w_ori = np.concatenate((coord_w_ori, coord_w_set_plus), 0)  # 0: upper_left, 1:lower_right, 2: lower_left

        '''   
        if bb[2] >=55:
            print (name, '1', bb[2])
        else:
            print (name, '0', bb[2])

        vis_img = ImageDraw.Draw(resized_img)
        vis_img.text((20, 20), "%d %d"%(bb[2], bb[3]), fill="red")
        vis_img.rectangle([bb[1], bb[0], bb[1]+bb[3], bb[0]+bb[2]], outline='red',width=2)
        resized_img.save('tmp/%s.png' % (name))
        '''

        if self.use_augmentation:
            # flip or not
            if np.random.rand() > 0.5:
                resized_img = flip_image(resized_img, Image.FLIP_LEFT_RIGHT)
                crop_img = flip_image(crop_img, Image.FLIP_LEFT_RIGHT)
                mask_iregular = flip_image(mask_iregular, Image.FLIP_LEFT_RIGHT)
                pi_w, pi_h = resized_img.size
                assert pi_w == self.image_size and pi_h == self.image_size
                coord_red_ori = flip_coord(coord_red_ori, pi_h, pi_w, Image.FLIP_LEFT_RIGHT) # the entire image
                coord_w_ori = flip_coord(coord_w_ori, pi_h, pi_w, Image.FLIP_LEFT_RIGHT) #the entire image
                bb = flip_bbox(bb, pi_h, pi_w, Image.FLIP_LEFT_RIGHT)

                small_human_img = flip_image(small_human_img, Image.FLIP_LEFT_RIGHT)
                small_mask_img = flip_image(small_mask_img, Image.FLIP_LEFT_RIGHT)

                sh_w, sh_h = small_human_img.size # the padded person image
                small_bb = flip_bbox(small_bb, sh_h, sh_w, Image.FLIP_LEFT_RIGHT)
                if 1:
                    the_small_bb = mask2bbox(small_mask_img)
                    the_bb = mask2bbox(mask_iregular)
                    assert bb == the_bb, (bb, the_bb)
                    assert small_bb == the_small_bb, (small_bb, the_small_bb)

        #[0 1]
        resized_img = transforms.ToTensor()(resized_img)
        #[-1 1]
        crop_img = transforms.ToTensor()(crop_img)
        mask_iregular = transforms.ToTensor()(mask_iregular)
        coord_red_ori = torch.tensor(coord_red_ori, dtype=torch.float32)
#        coord_w_ori = torch.tensor(coord_w_ori, dtype=torch.float32)
        coord_w_ori = torch.tensor([coord_w_ori[0][0], coord_w_ori[0][1], coord_w_ori[1][0], coord_w_ori[1][1]], dtype=torch.float32)
        small_human_img = transforms.ToTensor()(small_human_img)
        small_mask_img = transforms.ToTensor()(small_mask_img)
        bb = torch.IntTensor(bb)
        small_bb = torch.IntTensor(small_bb)

        # assign a weight within [1, 2] to a person by his height in [120 350]
       # _, person_height = coord_w_ori[1] - coord_w_ori[0]
        person_weight = int(self.data_info[name])
#        person_height = min(max(person_height, 120), 350)
        # person_weight = 2.0 * math.exp(-1.2537 * (person_height- 120) / self.image_size.0)
        #print (name, person_weight, small_bb.numpy()[2:])

        return resized_img, mask_iregular, bb, crop_img, name, coord_w_ori, coord_red_ori, small_human_img, small_mask_img, small_bb, torch.tensor(person_weight, dtype=torch.float32)

    def __len__(self):
       return len(self.common)


class Dataset_advPatchNew(data.Dataset):
    def __init__(self, data_path, file_list_path, mask_path, bbox_path, image_size=(1920, 1080)):
        super(Dataset_advPatchNew, self).__init__()



    @staticmethod
    def import_txt(path):
        data = np.loadtxt(path, dtype=str)
        output = {}
        for row in data:
            fname, patch_size_label = row
            output[fname[9:-4]] = patch_size_label
        return output

    @staticmethod
    def _pad_resize_img(img, size):
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(0, 0, 0))
                padded_img.paste(img, (int(padding), 0))
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(0, 0, 0))
                padded_img.paste(img, (0, int(padding)))
        resize = transforms.Resize(size)
        padded_img = resize(padded_img)
        return padded_img

    @staticmethod
    def _resize_img(img, size):
        resize = transforms.Resize(size)
        resized_img = resize(img)
        return resized_img

    def __getitem__(self, index):
        img_path = self.imgs_set[index]
        #print(index, img_path)
        name = img_path.split('/')[-1][:-4]
        ori_img = Image.open(img_path)

        resized_img = ori_img  # self._resize_img(ori_img, size=self.image_size)

        coord_red_ori = np.array(self.coord_r_set[index])
        coord_w_ori = np.array(self.coord_w_set[index])

        # coord_red_ori = self._pad_corrd(coord_red_ori/2.)

        # #####
        # M2 = cv2.getPerspectiveTransform(np.array([[0, 0], [self.image_size-1, 0], [0, self.image_size-1], [self.image_size-1, self.image_size-1]], dtype=np.float32),
        #                                  np.array([coord_red_ori[0], coord_red_ori[1], coord_red_ori[2],
        #                                            coord_red_ori[3]], dtype=np.float32)
        #                                  )
        #
        # mask_iregular = np.uint8(cv2.warpPerspective(self.mask_init, M2, (self.image_size, self.image_size)))
        # mask_iregular[mask_iregular > 100] = 255
        # mask_iregular[mask_iregular <= 100] = 0
        # mask_iregular = Image.fromarray(mask_iregular)
        # if not os.path.exists(img_path[:-38] + '600x600_wide_masks/' + name[:-5]):
        #     os.mkdir(img_path[:-38] + '600x600_wide_masks/' + name[:-5])
        #
        # mask_iregular.save(img_path[:-38] + '600x600_wide_masks/' + name[:-5] + '/' + name + '.png')
        # return 0

        if self.image_size == 416:
            mask_iregular = Image.open(img_path[:-38] + 'wide_masks/' + name[:-5] + '/' + name + '.png')
        elif self.image_size == 600:
            mask_iregular = Image.open(img_path[:-38] + '600x600_wide_masks/' + name[:-5] + '/' + name + '.png')
        elif self.image_size == (1920, 1080):
            mask_iregular = Image.open(img_path[:-38] + 'ori_masks/' + name[:-5] + '/' + name + '.png')
        else:
            raise NotImplementedError

        bb = mask2bbox(mask_iregular)

        # small image
        w = coord_w_ori / 416. * 1920.
        w[:, 0] = w[:, 0] - (1920-1080)//2
        w = np.round(w).astype(np.int32)
        small_human_img = ori_img.crop((*w[0], *w[1]))
        small_human_img = self._pad_resize_img(small_human_img, size=(256, 256))
        small_mask_img = Image.open(img_path[:-38] + 'small_masks/' + name[:-5] + '/' + name + '.png')
        # small_mask_img = mask_iregular.crop((*coord_w_ori[0], *coord_w_ori[1]))
        # small_mask_img = self._pad_resize_img(small_mask_img, size=(256, 256))

        small_bb = mask2bbox(small_mask_img)
        crop_bb = (*np.min(np.array(((self.coord_r_set[index]))), 0), *np.max(np.array(((self.coord_r_set[index]))), 0))
        crop_img = ori_img.crop(crop_bb)
        crop_img = crop_img.resize((224, 224), Image.BILINEAR)  # resize 224 x 224 (ResNet input)

        coord_w_ori = self._wide_coord(coord_w_ori)
        coord_w_set_plus = np.expand_dims(np.array([coord_w_ori[0, 0], coord_w_ori[1, 1]]), 0)
        coord_w_ori = np.concatenate((coord_w_ori, coord_w_set_plus), 0)  # 0: upper_left, 1:lower_right, 2: lower_left

        if self.use_augmentation is True:
            # flip or not
            if np.random.rand() > 0.5:
                resized_img = flip_image(resized_img, Image.FLIP_LEFT_RIGHT)
                crop_img = flip_image(crop_img, Image.FLIP_LEFT_RIGHT)
                mask_iregular = flip_image(mask_iregular, Image.FLIP_LEFT_RIGHT)
                pi_w, pi_h = resized_img.size
                assert pi_w == self.image_size[1] and pi_h == self.image_size[0]
                coord_red_ori = flip_coord(coord_red_ori, pi_h, pi_w, Image.FLIP_LEFT_RIGHT) # the entire image
                coord_w_ori = flip_coord(coord_w_ori, pi_h, pi_w, Image.FLIP_LEFT_RIGHT) #the entire image
                bb = flip_bbox(bb, pi_h, pi_w, Image.FLIP_LEFT_RIGHT)

                small_human_img = flip_image(small_human_img, Image.FLIP_LEFT_RIGHT)
                small_mask_img = flip_image(small_mask_img, Image.FLIP_LEFT_RIGHT)

                sh_w, sh_h = small_human_img.size # the padded person image
                small_bb = flip_bbox(small_bb, sh_h, sh_w, Image.FLIP_LEFT_RIGHT)
                if 1:
                    the_small_bb = mask2bbox(small_mask_img)
                    the_bb = mask2bbox(mask_iregular)
                    assert bb == the_bb, (bb, the_bb)
                    assert small_bb == the_small_bb, (small_bb, the_small_bb)

        #[0 1]
        resized_img = transforms.ToTensor()(resized_img)
        #[-1 1]
        crop_img = transforms.ToTensor()(crop_img)
        mask_iregular = transforms.ToTensor()(mask_iregular)
        coord_red_ori = torch.tensor(coord_red_ori, dtype=torch.float32)
        coord_w_ori = torch.tensor([coord_w_ori[0][0], coord_w_ori[0][1], coord_w_ori[1][0], coord_w_ori[1][1]], dtype=torch.float32)
        small_human_img = transforms.ToTensor()(small_human_img)
        small_mask_img = transforms.ToTensor()(small_mask_img)
        bb = torch.IntTensor(bb)
        small_bb = torch.IntTensor(small_bb)

        # patch size
        patch_size_level = int(self.data_info[name])

        return resized_img, mask_iregular, bb, crop_img, name, coord_w_ori, coord_red_ori, small_human_img, small_mask_img, small_bb, torch.tensor(patch_size_level, dtype=torch.float32)


if __name__ == '__main__':
    # data = Dataset_advT(data_path='/home/xukaidi/Workspace/adversarial-yolo/new_data/test/frames_padded',
    #             ori_imgs_path='/home/xukaidi/Workspace/adversarial-yolo/new_data/test/ori_images',
    #             mask_path='/home/xukaidi/Workspace/adversarial-yolo/new_data/test/irregular_masks',
    #             bbox_path='/home/xukaidi/Workspace/adversarial-yolo/new_data/test/boundbox',
    #             grid_path='/home/xukaidi/Workspace/adversarial-yolo/new_data/test/gridpoints/corner_points',
    #             tps_path='/home/xukaidi/Workspace/adversarial-yolo/data/TPS_pairs_old.npy', use_loc_net=True,
    #                     use_augmentation=False)
    #
    # data_loader = torch.utils.data.DataLoader(dataset=data,
    #                                           batch_size=48,
    #                                           shuffle=True,
    #                                           num_workers=8)
    #
    # for i, (batch_data) in enumerate(data_loader):
    #     img_batch, mask_batch, bb, crop, name, coord_w_set, coord_r_set, small_human_img, small_mask_img, small_bb = batch_data
    #     imgs = img_batch # * (1-mask_batch)
    #     for id, img in enumerate(imgs):
    #         img = transforms.ToPILImage()((img+1.)/2.)
    #         # bbox = bb[id].type(torch.int32)
    #         # img = img.crop((bbox[1].numpy() + 0, bbox[0].numpy() + 0, bbox[1].numpy() + bbox[3].numpy(), bbox[0].numpy() + bbox[2].numpy()))
    #         img = img.crop((*coord_w_set[id][0].numpy(), *coord_w_set[id][1].numpy()))
    #         img.save('tmp/' + name[id] + '.png')
    #
    #     break

    # data_ = Dataset_advT_color(data_path='../../adv_data/dataset_colorful/ori_framses',
    #                    file_list_path='../../adv_data/dataset_colorful/all_file_names.txt',
    #                    mask_path='anything here',
    #                    bbox_path='../../adv_data/dataset_colorful/person_bbox',
    #                    grid_path='../../adv_data/dataset_colorful/pattern_coord',
    #                    use_loc_net=True, use_augmentation=True, image_size=600)
    #
    # data_loader = torch.utils.data.DataLoader(dataset=data_,
    #                                           batch_size=64,
    #                                           shuffle=True,
    #                                           num_workers=1)
    #
    # for i, (batch_data) in enumerate(data_loader):
    #     print(i)
    #     img_batch, mask_batch, bb, crop, name, coord_w_set, coord_r_set, small_human_img, small_mask_img, small_bb, _ = batch_data
    #     imgs = img_batch # * (1-mask_batch)
    #     for id, img in enumerate(imgs):
    #         img = transforms.ToPILImage()(img)
    #         bbox = bb[id].type(torch.int32)
    #         img1 = img.crop((bbox[1].numpy() + 0, bbox[0].numpy() + 0, bbox[1].numpy() + bbox[3].numpy(), bbox[0].numpy() + bbox[2].numpy()))
    #         img2 = img.crop((*coord_w_set[id][0].numpy(), *coord_w_set[id][1].numpy()))
    #         img1.save('tmp/' + name[id] + '_pattern.png')
    #         img2.save('tmp/' + name[id] + '_human.png')
    #
    #     imgs = small_human_img
    #     for id, img in enumerate(imgs):
    #         img2 = img * (1-small_mask_img[id])
    #         img2 = transforms.ToPILImage()(img2)
    #         img2.save('tmp/' + name[id] + '_human_small.png')
    #         img = transforms.ToPILImage()(img)
    #         bbox = small_bb[id].type(torch.int32)
    #         img1 = img.crop((bbox[1].numpy() + 0, bbox[0].numpy() + 0, bbox[1].numpy() + bbox[3].numpy(),
    #                          bbox[0].numpy() + bbox[2].numpy()))
    #         img1.save('tmp/' + name[id] + '_pattern_small.png')
    #
    #     break

    data_ = Dataset_advPatch(data_path='../../adv_data/dataset_colorful/ori_framses',
                               file_list_path='../../adv_data/dataset_colorful/all_file_names.txt',
                               mask_path='anything here',
                               bbox_path='../../adv_data/dataset_colorful/person_bbox',
                               grid_path='../../adv_data/dataset_colorful/pattern_coord',
                               use_loc_net=True, use_augmentation=True, image_size=(1920, 1080))

    data_loader = torch.utils.data.DataLoader(dataset=data_,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=1)

    for i, (batch_data) in enumerate(data_loader):
        print(i)
        img_batch, mask_batch, bb, crop, name, coord_w_set, coord_r_set, small_human_img, small_mask_img, small_bb, _ = batch_data
        imgs = img_batch  # * (1-mask_batch)
        for id, img in enumerate(imgs):
            img = transforms.ToPILImage()(img)
            bbox = bb[id].type(torch.int32)
            img1 = img.crop((bbox[1].numpy() + 0, bbox[0].numpy() + 0, bbox[1].numpy() + bbox[3].numpy(),
                             bbox[0].numpy() + bbox[2].numpy()))
            img2 = img.crop((*coord_w_set[id][0].numpy(), *coord_w_set[id][1].numpy()))
            img1.save('tmp/' + name[id] + '_pattern.png')
            img2.save('tmp/' + name[id] + '_human.png')

        imgs = small_human_img
        for id, img in enumerate(imgs):
            img2 = img * (1 - small_mask_img[id])
            img2 = transforms.ToPILImage()(img2)
            img2.save('tmp/' + name[id] + '_human_small.png')
            img = transforms.ToPILImage()(img)
            bbox = small_bb[id].type(torch.int32)
            img1 = img.crop((bbox[1].numpy() + 0, bbox[0].numpy() + 0, bbox[1].numpy() + bbox[3].numpy(),
                             bbox[0].numpy() + bbox[2].numpy()))
            img1.save('tmp/' + name[id] + '_pattern_small.png')

        break
