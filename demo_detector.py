import sys
import os
import argparse
from detector.build_object_detector import build_object_detector
from utils.tools import get_config
from opts import arg_parser, merge_args
import glob
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

def load_detection(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def prune_detection(detection):
    person_num = len(detection)
    if person_num <= 2:
        return detection

    areas =[(det[2] - det[0]) * (det[3] - det[1]) for det in detection]
    I = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)
    #print (areas, I)
    first_person = detection[I[0]]
    xc = (first_person[0] + first_person[2]) / 2.0
    second_person = None
    for k in range(1, len(I)):
        det = detection[I[k]]
        half_width = abs((first_person[2] - first_person[0])) / 2.0
        xc_temp = (det[0] + det[2]) / 2.0
        if abs(xc_temp - xc) > half_width:
            second_person = det
            break

    return [first_person, second_person] if second_person is not None else [first_person]

def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()

    config = get_config(args.config)
    config = merge_args(args, config)
    #config['detector_input_size'] = [540, 960]
    detector = build_object_detector(config)
    
    device_ids = args.gpu_ids
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
    device_ids = list(range(len(device_ids)))
    detector.cuda(int(device_ids[0]))

    input_size = config['detector_input_size']
    target_obj_id = detector.target_object_id
    results = {}

    video_name = args.test_dir.split('/')[-1]
    #gt_detections = load_detection('../../CVPR_detection_results/20201106/DFaster_RCNN_R101_COCO+' + video_name+'.json')
    with torch.no_grad():
        for img_path in sorted(glob.glob(args.test_dir+"/*.png")):
            frame_img = Image.open(img_path)
            w, h = frame_img.size
            if w != input_size[0] or h != input_size[1]:
                frame_img = frame_img.resize(input_size, Image.BILINEAR)
    
            '''   
            frm_name = img_path.split('/')[-1] 
            frm_name = frm_name.split('.')[0]
            gt_det = gt_detections[frm_name]
            pruned_det = prune_detection(gt_det)
            bb = list(map(int, pruned_det[0][:4]))
            ph = bb[3] - bb[1] + 1
            pw = bb[2] - bb[0] + 1
            bb = [bb[0]+int(pw*0.15), bb[1]+int(ph*0.15), bb[2]-int(pw*0.25), bb[3]-int(ph*0.4)]
            np_im = np.array(frame_img)
            #print (np_im.shape, bb)
            np_im[bb[1]:bb[3],bb[0]:bb[2],:] = 128
            frame_img = Image.fromarray(np_im, 'RGB')
            '''
            frame_img = transforms.ToTensor()(frame_img)
            frame_img = frame_img.cuda()
            frame_img = torch.unsqueeze(frame_img, dim=0)
            #detections = detector.detect(frame_img, nms_thresh=config['val_nms_thresh'], conf_thresh=config['val_conf_thresh'])
            detections = detector.detector_detect(frame_img, nms_thresh=config['val_nms_thresh'], conf_thresh=config['val_conf_thresh'])
            person_detection = []
            for idx, detection in enumerate(detections):
                for det in detection:
                    if det is None:
                       continue
                    if det[-1] == target_obj_id:  # only count the person
                        det = det.detach().cpu().numpy().tolist()
                        person_detection.append(det)
            filename = os.path.basename(img_path)
            print (filename)
            results[filename.split('.')[0]] = person_detection
    if not os.path.isdir(args.detection_output_dir):
        os.mkdir(args.detection_output_dir)
    output_filename = os.path.join(args.detection_output_dir, detector.name + '+' + os.path.basename(args.test_dir) + '.json')
    print ('Results are written to %s' % output_filename)
    with open(output_filename, 'w') as f:
        json.dump(results, f)

'''
if __name__ == '__main__':
    main()

    config = merge_args(args, config)

    detector = build_object_detector(config).cuda()
    for img in glob.glob("Path/to/dir/*.jpg"):
        frame_img = Image.open(img)
        frame_img = transforms.ToTensor()(frame_img)
        frame_img = frame_img.cuda()
        torch.squeeze(frame_image, dim=0)
        detection_results = detector.detect(frame_img, nms_thresh=config['val_nms_thresh'], conf_thresh=config['val_conf_thresh'])
        print (detection_results)
'''

if __name__ == '__main__':
    main()
