import os
from dataset.dataset_adv import Dataset_advPatch, Dataset_advPatch_old
from dataset.dataset_pt import Dataset_PatchTransformer, Dataset_PatchTransformer_old

ADV_T_DATASETS = {
    'new_data': {
        'train': {
            'data_path': 'ori_framses',
            'file_list_path': 'train_old.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
        'val': {
            'data_path': 'ori_framses',
            'file_list_path': 'test_old.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
    },
    'unpadded_new_data': {
        'train': {
            'data_path': 'ori_framses',
            'file_list_path': 'train_old.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
        'val': {
            'data_path': 'ori_framses',
            'file_list_path': 'test_old.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
    },
    'small_data': {
        'train': {
            'data_path': 'ori_framses',
            'file_list_path': 'train_old_small.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
        'val': {
            'data_path': 'ori_framses',
            'file_list_path': 'test_old.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
    },
    'color_data': {
        'train': {
            'data_path': 'ori_framses',
            'file_list_path': 'train.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
        'val': {
            'data_path': 'ori_framses',
            'file_list_path': 'val.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
    },
    'nips_data': {
        'train': {
            'data_path': 'ori_framses',
            'file_list_path': 'train_nips.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
        'val': {
            'data_path': 'ori_framses',
            'file_list_path': 'val_nips.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
    },
    'unpadded_nips_data': {
        'train': {
            'data_path': 'ori_framses',
            'file_list_path': 'train_nips.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
        'val': {
            'data_path': 'ori_framses',
            'file_list_path': 'val_nips.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
    },
    'unpadded_nips_data_near': {
        'train': {
            'data_path': 'ori_framses',
            'file_list_path': 'train_nips_near.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
        'val': {
            'data_path': 'ori_framses',
            'file_list_path': 'val_nips_near.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
    },
    'unpadded_nips_data_far': {
        'train': {
            'data_path': 'ori_framses',
            'file_list_path': 'train_nips_far.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
        'val': {
            'data_path': 'ori_framses',
            'file_list_path': 'val_nips_far.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
    },
    'PT_new_data': {
        'train': {
            'data_path': 'train/frames_padded',
            'ori_imgs_path': 'train/ori_images',
            'mask_path': 'train/irregular_masks',
            'bbox_path': 'train/boundbox',
            'grid_path': 'train/gridpoints/corner_points',
            'tps_path': 'TPS_pairs_old.npy'},
        'val': {
            'data_path': 'test/frames_padded',
            'ori_imgs_path': 'test/ori_images',
            'mask_path': 'test/irregular_masks',
            'bbox_path': 'test/boundbox',
            'grid_path': 'test/gridpoints/corner_points',
            'tps_path': 'nTPS_pairs_old.npy'}
    },
    'unpadded_dist_data': {
        'train': {
            'data_path': 'ori_framses',
            'file_list_path': 'train_dist.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
        'val': {
            'data_path': 'ori_framses',
            'file_list_path': 'val_dist.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
    },
    'unpadded_dist_data_near': {
        'train': {
            'data_path': 'ori_framses',
            'file_list_path': 'train_dist_near.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
        'val': {
            'data_path': 'ori_framses',
            'file_list_path': 'val_dist_near.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
    },
    'unpadded_dist_data_far': {
        'train': {
            'data_path': 'ori_framses',
            'file_list_path': 'train_dist_far.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
        'val': {
            'data_path': 'ori_framses',
            'file_list_path': 'val_dist_far.txt',
            'mask_path': 'anything here',
            'bbox_path': 'person_bbox',
            'grid_path': 'pattern_coord',
            'tps_path': 'anything here'},
    },

    'PT_color_data': {
        'train': {
            'data_path': 'train/frames_padded',
            'ori_imgs_path': 'train/ori_images',
            'mask_path': 'train/irregular_masks',
            'bbox_path': 'train/boundbox',
            'grid_path': 'train/gridpoints/corner_points',
            'tps_path': 'TPS_pairs_old.npy'},
        'val': {
            'data_path': 'test/frames_padded',
            'ori_imgs_path': 'test/ori_images',
            'mask_path': 'test/irregular_masks',
            'bbox_path': 'test/boundbox',
            'grid_path': 'test/gridpoints/corner_points',
            'tps_path': 'nTPS_pairs_old.npy'}
    },
    'advT_data': {
        'train': {
            'file_list_path': 'train_advT.txt',
            'img_path': 'ori_framses',
            'mask_path': 'ori_masks'},
        'val': {
            'file_list_path': 'val_advT.txt',
            'img_path': 'ori_framses',
            'mask_path': 'ori_masks'}
    },
    'advT_far_data': {
        'train': {
            'file_list_path': 'train_advT_far.txt',
            'img_path': 'ori_framses',
            'mask_path': 'ori_masks'},
        'val': {
            'file_list_path': 'val_advT_far.txt',
            'img_path': 'ori_framses',
            'mask_path': 'ori_masks'}
    },
    'advT_near_data': {
        'train': {
            'file_list_path': 'train_advT_near.txt',
            'img_path': 'ori_framses',
            'mask_path': 'ori_masks'},
        'val': {
            'file_list_path': 'val_advT_near.txt',
            'img_path': 'ori_framses',
            'mask_path': 'ori_masks'}
    },
    'neu': {
        'train': {
            'file_list_path': 'train_neu.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'
            },
        'val': {
            'file_list_path': 'val_neu.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'
    }
   },
   'neu_near': {
        'train': {
            'file_list_path': 'train_neu_near.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'
            },
        'val': {
            'file_list_path': 'val_neu_near.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'
    }
   },
   'neu_far': {
        'train': {
            'file_list_path': 'train_neu_far.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'
            },
        'val': {
            'file_list_path': 'val_neu_far.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'
    }
   },
   'neu_color': {
        'train': {
            'file_list_path': 'train_neu_color.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'
        },
        'val': {
            'file_list_path': 'val_neu_color.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'
        }
    },
    'neu_color_small': {
        'train': {
            'file_list_path': 'train_neu_color_small.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks'
        },
        'val': {
            'file_list_path': 'val_neu_color_small.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks'
        }
    },
    'cvpr': {
        'train': {
            'file_list_path': 'train_cvpr.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks'
        },
        'val': {
            'file_list_path': 'val_cvpr.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks'
        }
    },
    'think_data': {
        'train': {
            'file_list_path': 'train_think.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks'
        },
        'val': {
            'file_list_path': 'val_think.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks'
        }
    },
    'think_neu_data': {
        'train': {
            'file_list_path': 'train_think_neu.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks'
        },
        'val': {
            'file_list_path': 'val_think_neu.txt',
            'img_path': 'ori_frames',
            'mask_path': 'ori_masks'
        }
    }
}

PATCH_TRANSFORMER_DATASETS = {
    'colorpalette_data': {
        'train': {
            'img_path': 'ori_frames',
            'file_list_path': 'train.txt',
            'mask_path': 'mask',
            'patch_path': 'patterns'},
        'val': {
            'img_path': 'ori_frames',
            'file_list_path': 'val.txt',
            'mask_path': 'mask',
            'patch_path': 'patterns'},
    },
    'advT_data': {
        'train': {
            'img_path': 'ori_framses',
            'file_list_path': 'train_advT.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
        'val': {
            'img_path': 'ori_framses',
            'file_list_path': 'val_advT.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
    },
    'neu': {
        'train': {
            'img_path': 'ori_frames',
            'file_list_path': 'train_neu.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
        'val': {
            'img_path': 'ori_frames',
            'file_list_path': 'val_neu.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
    },
    'neu_color': {
        'train': {
            'img_path': 'ori_frames',
            'file_list_path': 'train_neu_color.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
        'val': {
            'img_path': 'ori_frames',
            'file_list_path': 'val_neu_color.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
    },
    'neu_color_small': {
        'train': {
            'img_path': 'ori_frames',
            'file_list_path': 'train_neu_color_small.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
        'val': {
            'img_path': 'ori_frames',
            'file_list_path': 'val_neu_color_small.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
    },
    'binney': {
        'train': {
            'img_path': 'ori_frames',
            'file_list_path': 'train_binney.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
        'val': {
            'img_path': 'ori_frames',
            'file_list_path': 'val_binney.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
    },
    'cvpr': {
        'train': {
            'img_path': 'ori_frames',
            'file_list_path': 'train_cvpr.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
        'val': {
            'img_path': 'ori_frames',
            'file_list_path': 'val_cvpr.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
    },
    'pattern0-1-4-5': {
        'train': {
            'data_path': 'ori_frames',
            'file_list_path': 'train.txt',
            'patch_path': ''},
        'val': {
            'data_path': 'ori_frames',
            'file_list_path': 'val.txt',
            'patch_path': ''},
        },
    'think_data': {
        'train': {
            'img_path': 'ori_frames',
            'file_list_path': 'train_think_small.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
        'val': {
            'img_path': 'ori_frames',
            'file_list_path': 'val_think.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
    },
    'think_neu_data': {
        'train': {
            'img_path': 'ori_frames',
            'file_list_path': 'train_think_neu.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
        'val': {
            'img_path': 'ori_frames',
            'file_list_path': 'val_think_neu.txt',
            'mask_path': 'ori_masks',
            'patch_path': 'patterns'},
    },
    }

def get_advPatch_data_loader(data_dir, dataset, data_type, image_size, person_crop_size, data_list_file=None, patch_size_range=None, use_augmentation=False, use_loc_net=True):

    data_info = {}
    for key, value in ADV_T_DATASETS[dataset][data_type].items():
        data_info[key] = os.path.join(data_dir, value)

   # if 'unpadded' in dataset:
   #     print ('Using UNPADDED data: %s .....' % (dataset))
   # else:
   #     print ('Using PADDED data %s .....' % (dataset))


     #overwrite the predefined the file
    if data_list_file is not None:
        data_info['file_list_path'] = os.path.join(data_dir, data_list_file)

    if dataset in ['advT_data', 'advT_near_data', 'advT_far_data', 'neu', 'neu_near', 'neu_far', 'neu_color', 'neu_color_small', 'cvpr', 'think_data', 'think_neu_data']:
        return Dataset_advPatch(file_list_path=data_info['file_list_path'],
                                  img_path=data_info['img_path'],
                                  mask_path=data_info['mask_path'],
                                  image_size = image_size,
                                  person_crop_size = person_crop_size,
                                  patch_path=data_info['patch_path'],
                                  patch_size_range = patch_size_range,
                                  use_augmentation=use_augmentation)
    elif dataset == 'unpadded_new_data' or dataset == 'unpadded_nips_data' or dataset == 'unpadded_dist_data' or \
       dataset == 'unpadded_nips_data_near' or dataset == 'unpadded_nips_data_far' or \
        dataset == 'unpadded_dist_data_near' or dataset == 'unpadded_dist_data_far':
        return Dataset_advPatch_old(data_path=data_info['data_path'],
                                  file_list_path=data_info['file_list_path'],
                                  mask_path=data_info['mask_path'],
                                  bbox_path=data_info['bbox_path'],
                                  grid_path=data_info['grid_path'],
                                  tps_path=data_info['tps_path'],
                                  use_augmentation=use_augmentation,
                                  use_loc_net=use_loc_net)
    elif dataset == 'new_data' or dataset == 'nips_data':
        return Dataset_advPatch_old(data_path=data_info['data_path'],
                                  file_list_path=data_info['file_list_path'],
                                   mask_path=data_info['mask_path'],
                                   bbox_path=data_info['bbox_path'],
                                   grid_path=data_info['grid_path'],
                                   tps_path=data_info['tps_path'],
                                   use_augmentation=use_augmentation,
                                   use_loc_net=use_loc_net)


    raise ValueError('Dataset %s not available' % (dataset))


def get_patchTransformer_data_loader(data_dir, dataset, data_type, data_list_file=None, patch_path=None, person_crop_size=(256, 256), use_augmentation=False):

    data_info = {}
    for key, value in PATCH_TRANSFORMER_DATASETS[dataset][data_type].items():
        data_info[key] = os.path.join(data_dir, value)

     #overwrite the predefined the file
    if data_list_file is not None:
        data_info['file_list_path'] = os.path.join(data_dir, data_list_file)

    if patch_path is not None:
        data_info['patch_path'] = patch_path

    if dataset in ['advT_data', 'colorpalette_data', 'neu', 'neu_color', 'neu_color_small', 'cvpr', 'binney', 'think_data', 'think_neu_data']:
        return Dataset_PatchTransformer(file_list_path=data_info['file_list_path'],
                                img_path=data_info['img_path'],
                                mask_path=data_info['mask_path'],
                                patch_path=data_info['patch_path'],
                                person_crop_size=person_crop_size,
                                use_augmentation=use_augmentation)

    if dataset == 'pattern0-1-4-5':
        return Dataset_PatchTransformer_old(list_file=data_info['file_list_path'],
                                data_path=data_dir,
                                use_STN=True,
                                crop_mask=False,
                                template_resize=False,
                                patch_path=data_info['patch_path'],
                                flip_data = use_augmentation,
                                return_name = True)

    raise ValueError('Dataset %s not available' % (dataset))

