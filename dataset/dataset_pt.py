import torch.utils.data as data
from utils.tools import normalize, mask2bbox, default_loader, is_image_file

import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw


def flip_image(img, flip_method):
    return img if flip_method < 0 else img.transpose(flip_method)

def flip_bbox(bb, img_h, img_w, flip_method):
    # bb: y,x, h, w
    if flip_method == Image.FLIP_LEFT_RIGHT:
        return (bb[0], img_w - bb[1] - bb[3], bb[2], bb[3])

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
        if len(bb) == 3:  # w
            new_bb = [[xmin, ymin], [xmax, ymax], [xmin, ymax]]
        elif len(bb) == 4:  # w
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


class Dataset_PatchTransformer(data.Dataset):
    def __init__(self, file_list_path, img_path, mask_path, patch_path, person_crop_size=(256, 256), use_augmentation=True):
        super(Dataset_PatchTransformer, self).__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.patch_path = patch_path
        self.person_crop_size = person_crop_size
        self.use_augmentation = use_augmentation

        # use the ground truth patch or a new patch provided in the path
        self.use_gt_patch = True if os.path.isdir(self.patch_path) else False

        data = np.loadtxt(file_list_path, dtype=str)
        self.img_file_list = data[:, 0].tolist()
        self.patch_file_list = data[:, 1].tolist()
        bbox = data[:, 2:].astype(np.int32)
        bbox[bbox < 0] = 0
        self.adv_person_bbox = bbox

        '''
        for i in range(len(self.img_file_list)):
                self.visualize(i)
#                if i > 50:
#                    break
        sys.exit(0)
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
        return [b[1], b[0], b[1] + b[3] - 1, b[0] + b[2] - 1]

    @staticmethod
    def _x1y1x2y2_to_yxhw(b):
        return [b[1], b[0], b[3] - b[1] + 1, b[2] - b[0] + 1]

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
        is_flip = self.use_augmentation is True and np.random.rand() > 2.0
        if is_flip:
            pi_w, pi_h = frame_img.size
            frame_img = flip_image(frame_img, Image.FLIP_LEFT_RIGHT)
            big_patch_mask = flip_image(big_patch_mask, Image.FLIP_LEFT_RIGHT)
            template_img = flip_image(template_img, Image.FLIP_LEFT_RIGHT)
            person_bb = flip_bbox(self._x1y1x2y2_to_yxhw(person_bb), pi_h, pi_w, Image.FLIP_LEFT_RIGHT)
            person_bb = self._yxhw_to_x1y1x2y2(person_bb)

        print(img_path, self.patch_file_list[index], person_bb)
        # target person image and maks (person crop)
        crop_person_img = frame_img.crop(tuple(person_bb))
        # padd and resize it to 256x256
        crop_person_img = self._pad_resize_img(crop_person_img, self.person_crop_size)
        # mask crop
        small_patch_mask = big_patch_mask.crop(tuple(person_bb))
        small_patch_mask = self._pad_resize_img(small_patch_mask, self.person_crop_size, mode=Image.NEAREST)
        # bounding box of the small mask
        small_mask_bb = mask2bbox(small_patch_mask)
        small_mask_bb = self._yxhw_to_x1y1x2y2(small_mask_bb)

        # resize it to 224x224
        big_mask_bb = mask2bbox(big_patch_mask)
        patch_img = frame_img.crop(tuple(self._yxhw_to_x1y1x2y2(big_mask_bb)))
        patch_img = patch_img.resize((224, 224), Image.BILINEAR)
        big_mask_bb = mask2bbox(big_patch_mask)
        big_mask_bb = self._yxhw_to_x1y1x2y2(big_mask_bb)

        print(img_path, self.patch_file_list[index], small_mask_bb, big_mask_bb)
        '''
        big_patch_mask1 = ImageDraw.Draw(big_patch_mask)
        big_patch_mask1.rectangle(tuple(big_mask_bb), outline='red', width=2)
        small_patch_mask1 = ImageDraw.Draw(small_patch_mask)
        small_patch_mask1.rectangle(tuple(small_mask_bb), outline='red', width=2)
        crop_person_img1 = ImageDraw.Draw(crop_person_img)
        crop_person_img1.rectangle(tuple(small_mask_bb), outline='red', width=2)
        filename = os.path.basename(filename)
        #frame_img.save('tmp/frame_%s' % (filename))
        big_patch_mask.save('tmp/big_patch_mask_%s' % (filename))
        small_patch_mask.save('tmp/small_patch_mask_%s' % (filename))
        crop_person_img.save('tmp/person_%s' % (filename))
        patch_img.save('tmp/patch_%s' % (filename))
        '''

    def __getitem__(self, index):
        # original image
        filename = self.img_file_list[index]
        img_path = os.path.join(self.img_path, filename)
        frame_img = Image.open(img_path)

        # mask image
        mask_path = os.path.join(self.mask_path, filename.replace('.jpg', '.png'))
        big_patch_mask = Image.open(mask_path)

        # template_image
        template_file_path = os.path.join(self.patch_path, self.patch_file_list[index]) if self.use_gt_patch is True else self.patch_path
        template_img = Image.open(template_file_path)
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

        # target person image and maks (person crop)
        crop_person_img = frame_img.crop(tuple(person_bb))
        # padd and resize it to 256x256
        crop_person_img = self._pad_resize_img(crop_person_img, self.person_crop_size)
        # mask crop
        small_patch_mask = big_patch_mask.crop(tuple(person_bb))
        small_patch_mask = self._pad_resize_img(small_patch_mask, self.person_crop_size, mode=Image.NEAREST)
        # bounding box of the small mask
        small_mask_bb = mask2bbox(small_patch_mask)
        # print (img_path, small_mask_bb)

        # resize it to 224x224
        big_mask_bb = mask2bbox(big_patch_mask)
        patch_img = frame_img.crop(tuple(self._yxhw_to_x1y1x2y2(big_mask_bb)))
        patch_img = patch_img.resize((224, 224), Image.BILINEAR)

        crop_person_img = transforms.ToTensor()(crop_person_img)
        #crop_person_img = normalize(crop_person_img)

        template_img = transforms.ToTensor()(template_img)
        #template_img = normalize(template_img)

        small_patch_mask = transforms.ToTensor()(small_patch_mask)
        
        # while training PTnet requires input within [-1 1], applying PTNet an image doesn't require so
        # as STN is inferred from the patch directly
        # input range [-1 1] required by STN
        patch_img = transforms.ToTensor()(patch_img)
        #patch_img = normalize(patch_img)
        # this has been fixed in the generator, so normalization is no longer needed.

        small_mask_bb = torch.IntTensor(small_mask_bb)
        # resize ratio
        scale_factor = self.person_crop_size[1] / (person_bb[3] - person_bb[1] + 1.0)
        scale_factor = torch.FloatTensor([scale_factor])

        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        #print (filename, template_file_path, patch_img.shape, small_patch_mask.shape, crop_person_img.shape, template_img.shape, small_mask_bb)
        return filename, patch_img, small_patch_mask, template_img, crop_person_img, small_mask_bb, scale_factor
        # return filename.replace('pattern4', pattern), input_img, mask_img, template_img, target_img, torch.IntTensor(bb)

# return input_img, mask_img, template_img, target_img, torch.IntTensor(bb)

    def __len__(self):
        return len(self.img_file_list)


# use preprocessed images and masks
class Dataset_PatchTransformer_old(data.Dataset):
    def __init__(self, data_path, list_file, use_STN=False, crop_mask=False, template_resize=False, patch_path=None, flip_data=True, return_name=False):
        super(Dataset_PatchTransformer_old, self).__init__()
        self.data_path = data_path
        self.list_file = list_file
        self.use_STN = use_STN
        self.crop_mask = crop_mask
        self.template_resize=template_resize
        self.flip_method = [Image.FLIP_LEFT_RIGHT] if flip_data else []
#        self.flip_method = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM] if flip_data else []
        self.return_name = return_name
        self.patch_path = patch_path if os.path.isfile(patch_path) else None

        self.samples = self._parse_list()

    def _parse_list(self):
        with open(os.path.join(self.data_path, self.list_file), 'r') as f:
            file_list = f.read().splitlines()

        # -9999: no flip; FLIP_LEFT_RIGHT: flip left to right; FLIP_TOP_BOTTOM: flip bottom to top
        image_index_list = [(item, -9999) for item in file_list]
        for method in self.flip_method:
            image_index_list += [(item, method) for item in file_list]

        if Image.FLIP_TOP_BOTTOM in self.flip_method:
            print('FLIP_TOP_BOTTOM is applied')
        if Image.FLIP_LEFT_RIGHT in self.flip_method:
            print('FLIP_LEFT_RIGHT is applied')

        print('Data size is %d' % (len(image_index_list)))
        return image_index_list

    def _flip_image(self, img, flip_method):
        return img if flip_method < 0 else img.transpose(flip_method)

    def _flip_bbox(self, bb, img_h, img_w, flip_method):
        # bb: y,x, h, w
        if flip_method == Image.FLIP_LEFT_RIGHT:
            return (bb[0], img_w - bb[1] - bb[3], bb[2], bb[3])
            #return (img_w - bb[0] - bb[2], bb[1], bb[2], bb[3])

        if flip_method == Image.FLIP_TOP_BOTTOM:
            return (img_h - bb[0] - bb[2], bb[1], bb[2], bb[3])

        return bb

    def __getitem__(self, index):
             filename, flip_method = self.samples[index]

             # target image
             target_img_path = os.path.join(self.data_path, filename)
             target_img = default_loader(target_img_path)

             # mask image
             mask_img_path = os.path.join(self.data_path, filename[0:-4] + '_mask.png')
             mask_img = default_loader(mask_img_path)
             bb = mask2bbox(mask_img)
             assert len(bb) > 0

             # template image
             template_img_path = os.path.join(self.data_path, filename[:-18] + '.png') \
                 if self.patch_path is None else self.patch_path

             #pattern='pattern%d'%(np.random.randint(low=0, high=3))
             #template_img_path = os.path.join(self.data_path, filename[0:-18]+'.png').replace('pattern4', pattern)
             #print(filename, template_img_path)
             template_img = default_loader(template_img_path)

             # input image
             if self.use_STN: # input is the frame image
                 input_img = target_img.copy()
                 crop_bb = (bb[1], bb[0], bb[1] + bb[3] - 1, bb[0] + bb[2] - 1)
                 input_img = input_img.crop(crop_bb)
                 input_img = input_img.resize((224, 224), Image.BILINEAR) # resize 224 x 224 (imagenet format)
             else: #input is the constructed one using the template
                 input_img_path = os.path.join(self.data_path, filename[0:-4] + '_trans.png')
                 input_img = default_loader(input_img_path)
                 if self.crop_mask:
                     crop_bb = (bb[1], bb[0], bb[1] + bb[3] - 1, bb[0] + bb[2] - 1)
                     input_img = input_img.crop(crop_bb)
                     input_img = input_img.resize((256, 256), Image.BILINEAR) # resize to 256 by 256
                     target_img = target_img.crop(crop_bb)
                     target_img = target_img.resize((256, 256), Image.BILINEAR)
                     mask_img = mask_img.crop(crop_bb)
                     mask_img = mask_img.resize((256, 256), Image.BILINEAR)

             # we place the template into the target image beforehand
             if self.template_resize:
                 template_img = template_img.resize((bb[3], bb[2]), Image.BILINEAR)
                 template_pix = np.array(template_img)
                 bg_pix = np.array(mask_img.copy())
                 bg_pix[bb[0]:bb[0] + bb[2], bb[1]:bb[1] + bb[3], :] = template_pix
                 template_img = Image.fromarray(bg_pix)

              # put data into tensor and input range [0 1]
             target_img = transforms.ToTensor()(self._flip_image(target_img, flip_method))  # turn the image to a tensor
             #target_img = normalize(target_img)
             #target_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(target_img)

             #input range [-1 1] required by STN
             input_img = transforms.ToTensor()(self._flip_image(input_img, flip_method))  # turn the image to a tensor
             #input_img = normalize(input_img)
             #input_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input_img)

             #input range [0 1]
             template_img = transforms.ToTensor()(self._flip_image(template_img, flip_method))  # turn the image to a tensor
             #template_img = normalize(template_img)
             #template_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(template_img)

             # no need to normalize
             mask_img = self._flip_image(mask_img, flip_method)
             if 1:
                 the_bb = mask2bbox(mask_img)
             mask_img = transforms.ToTensor()(mask_img)  # turn the image to a tensor

             # the input image size is (256, 256)
             bb = self._flip_bbox(bb, 256, 256, flip_method)
             assert(bb[0] == the_bb[0] and bb[1] == the_bb[1] and bb[2] == the_bb[2] and bb[3] == the_bb[3])

             if self.return_name:
                 return filename, input_img, mask_img, template_img, target_img, torch.IntTensor(bb), torch.FloatTensor([1.0])
                 #return filename.replace('pattern4', pattern), input_img, mask_img, template_img, target_img, torch.IntTensor(bb)
             else:
                 return input_img, mask_img, template_img, target_img, torch.IntTensor(bb), torch.FloatTensor([1.0])

    def _find_samples_in_subfolders(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        # item = (path, class_to_idx[target])
                        # samples.append(item)
                        samples.append(path)
        return samples

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':

    data = Dataset_PatchTransformer(image_path='../../adv_data/dataset_newprinter/ori_frames',
                                  file_list_path='../../adv_data/dataset_new_printer/train.txt',
                                  mask_path='../../adv_data/dataset_newprinter/mask',
                                  patch_path='../../adv_data/advT-patches-newprinter',
                                  use_augmentation=True)

    data_loader = torch.utils.data.DataLoader(dataset=data_,
                                          batch_size=72,
                                          shuffle=True,
                                          num_workers=1)

    for i, (batch_data) in enumerate(data_loader):
        print(i)
        img_batch, mask_batch, crop, small_bb, _ = batch_data
        imgs = img_batch  # * (1-mask_batch)
