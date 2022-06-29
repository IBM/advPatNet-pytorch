import sys
import os
import argparse
import json
import numpy as np
import glob

# change this to your own robust detector (i.e. a model with 100% detection accuracy)
ROBUST_DETECTOR_NAME = 'DFaster_RCNN_R101_COCO'

def single_bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    # Intersection area
    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0.0) * max(inter_rect_y2 - inter_rect_y1 + 1, 0.0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

#remove false alarms
def prune_detection(detection):
    person_num = len(detection)
    if person_num == 0:
        return detection

    if person_num == 1:
        return detection[0]

    areas =[(det[2] - det[0]) * (det[3] - det[1]) for det in detection]
    I = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)
    #print (areas, I)
    return detection[I[0]]

def is_matched(gt_bb, detections, det_thresh, match_thresh):
    for det in detections:
        if det[4] > det_thresh and single_bbox_iou(gt_bb, det[:4]) >= match_thresh:
            return True

    return False

def match_results(gt_detections, detections, det_thresh, match_thresh, skip_num, skip_info=False):
    success = 0
    total_cnt = 0
    cut_skip = 0
    detection_skip = 0
    frame_cnt = 0
    for frame_num, det in gt_detections.items():
        frame_cnt += 1
        if int(frame_num.split('_')[-1]) < skip_num: # exclude it
            cut_skip += 1
            if skip_info:
                print('skip %, index < %d ' % (frame_num, skip_num))
            continue

        # det is a list of list
        pruned_det = prune_detection(det)
        person_num = len(pruned_det)
        if person_num <= 0:
            detection_skip += 1
            if skip_info:
                print ('skip %s, person_num: %d' % (frame_num, person_num))
            continue

        if pruned_det and is_matched(pruned_det[:4], detections[frame_num], det_thresh, match_thresh):
            success += 1

        total_cnt += 1
    return success, total_cnt, cut_skip, detection_skip, frame_cnt

def evaluate_adv_model(data_dir, data_list, attack_model, det_thresh=0.7, match_thresh=0.1, skip_num=0, skip_info=False):
    gt_files = get_file_list(data_dir, ROBUST_DETECTOR_NAME, data_list)
    gt_detections = [ load_detection(item) for item in gt_files ]

    detection_files = get_file_list(data_dir, attack_model, data_list)
    detections = [load_detection(item) for item in detection_files]
    
    matching_results = [match_results(gt, detection, det_thresh, match_thresh, skip_num=skip_num, \
                         skip_info=skip_info) for gt, detection in zip(gt_detections, detections)]

    return matching_results

def load_detection(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def get_file_list(data_dir, model, data_list):
    return [os.path.join(data_dir, model +'+'+ item + '.json') for item in data_list]

def get_dataset_list(data_dir, detector):
#    print (adv_patches)
    dataset_list = glob.glob(data_dir+'/*.json')
    dataset_list = [os.path.basename(item).split('.')[0] for item in dataset_list if detector in item]
    dataset_list = [item.split('+')[-1] for item in dataset_list]
    #for adv_patch in adv_patches:
    #    dataset_list = [item for item in dataset_list if adv_patch in item.split('_')]
#    dataset_list = [item for item in dataset_list if adv_patch in item.split('_')]
    return dataset_list

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch evaluation')
    parser.add_argument('--data_list', help="a list of video files")
    parser.add_argument('--data_dir', type=str, default='../../ICLR_detection_results', help="where are the detection results")
    parser.add_argument('--adv_patch', help='adversarial patch')
    parser.add_argument('--victim_model', type=str,  help='victim model')
    parser.add_argument('--skip_num', type=int, default=0, help='how many frames to be skipped')
    parser.add_argument('--skip_info', dest='skip_info', action='store_true', help='print skip info')
    parser.add_argument('--detection_thresh', dest='detection_thresh', type=float, default=0.7, help='threshold for detection_score')

    return parser

def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    adv_patches = args.adv_patch.split(',')
    all_dataset_list = get_dataset_list(args.data_dir, args.victim_model) if args.data_list is None else args.data_list.split(',')
    #print (all_dataset_list)
    #dataset_list = [item for item in dataset_list if 'PCTN' not in item]
    for adv_patch in sorted(adv_patches):
        print('\n======== %s =================' % (adv_patch))

        #dataset_list = get_dataset_list(args.data_dir, args.victim_model, adv_patch)
        dataset_list = [item for item in all_dataset_list if adv_patch in item.split('_')]

        assert dataset_list, 'no dataset found. please check the detetor name and results directory!'

        # determine the model to be evaluated for the data list
        MATCH_THRESH = 0.1
        results = evaluate_adv_model(args.data_dir, dataset_list, args.victim_model, det_thresh=args.detection_thresh, match_thresh=MATCH_THRESH, \
              skip_num=args.skip_num, skip_info=args.skip_info)
        results = np.array(results)
        tot_results = np.sum(results, axis=0)
        for dataset, r in sorted(zip(dataset_list, results), key=lambda t: t[0]):
           print ('%10s ASR %4.2f Detected: %3d Processed: %3d Cut skip: %3d Detection skip: %3d Total: %3d' % \
                 (dataset, (1.0 - r[0]/r[1]), r[0], r[1], r[2], r[3], r[4]))
        print ('----------------------------------------------')
        print ('%10s ASR %4.2f Detected: %3d Processed: %3d Cut skip: %3d Detection skip: %3d Total: %3d'  % \
                ('All', (1.0 - tot_results[0]/tot_results[1]), tot_results[0], tot_results[1], tot_results[2], tot_results[3], tot_results[4]))

if __name__ == '__main__':
    main()
