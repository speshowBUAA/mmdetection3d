from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_nearest_3d
import torch
import numpy as np
import json, os
from tqdm import tqdm

CLASSES = ['Car', 'Pedestrian', 'Truck', 'Motorcycle', 'Bicycle']
# def demo():
#     bboxes1 = np.array([[1,1,1,2,2,2,0]], dtype=np.float64)
#     bboxes2 = np.array([[1,1,1,1,1,1,0]], dtype=np.float64)
#     print(bboxes2.shape)
#     bboxes1 = torch.from_numpy(bboxes1)
#     bboxes2 = torch.from_numpy(bboxes2)
#     ret = bbox_overlaps_nearest_3d(bboxes2, bboxes1)
#     print(bboxes1)
#     print(bboxes2)
#     print(ret)

def calculate_F1(max, compensate = 0, threshold = 0.5):
    """
    Args:
        max (M,N)
    return:
        AP,TP,FP,FN,precision,recall,F1
    """
    if compensate < 0:
        compensate = 0

    TP = 0
    FP = 0
    FN = 0
    for i in range(len(max)):
        if max[i] > threshold:
            TP = TP + 1
        else:
            FP = FP + 1
    
    FN = compensate
    # precision = TP/(TP + FP)
    # recall = TP/(TP + FN)
    F1 = 2*TP / (2*TP+FN+FP)
    return TP,FP,FN,F1

def parse_anno(anno, cls):
    bboxes = []
    for i in range(len(anno)):
        if anno[i]['obj_type'] == cls:
            yaw = anno[i]['psr']['rotation']['z']
            x = anno[i]['psr']['position']['x']
            y = anno[i]['psr']['position']['y']
            z = anno[i]['psr']['position']['z']
            w = anno[i]['psr']['scale']['z']
            h = anno[i]['psr']['scale']['y']
            l = anno[i]['psr']['scale']['x']
            bboxes.append(np.array([x,y,z,w,l,h,yaw], dtype=np.float64))
    return np.array(bboxes)

def main():
    root_dir = './data_40_frame_test'
    # dir_list = os.listdir('.')
    dir_list = ['annotator_00', 'annotator_01', 'annotator_02', 'annotator_03', 'annotator_04', 'annotator_05']
    for _dir in dir_list:
        if '.' in _dir:
            continue
        annos_gt = os.listdir(os.path.join(root_dir, _dir, 'label_gt'))
        total_TP = 0
        total_FP = 0
        total_FN = 0
        for anno in annos_gt:
            if not os.path.exists(os.path.join(os.path.join(root_dir, _dir, 'label'), anno)):
                print("annotation not finished ! {} : {}".format(_dir, anno))
                continue
            anno_path = os.path.join(os.path.join(root_dir, _dir, 'label', anno))
            anno_gt_path = os.path.join(os.path.join(root_dir, _dir, 'label_gt', anno))
            with open(anno_path, 'r') as f:
                anno_dt = json.load(f)            
            with open(anno_gt_path, 'r') as f:
                anno_gt = json.load(f)
            
            for cls in CLASSES:
                bboxes_dt = parse_anno(anno_dt, cls)
                bboxes_gt = parse_anno(anno_gt, cls)
                if len(bboxes_dt) == 0 and len(bboxes_gt) == 0:
                    continue
                elif len(bboxes_gt) == 0:
                    TP = 0
                    FP = len(bboxes_dt)
                    FN = 0
                elif len(bboxes_dt) == 0:
                    TP = 0
                    FP = 0
                    FN = len(bboxes_gt)
                else:
                    bboxes_dt = torch.from_numpy(bboxes_dt)
                    bboxes_gt = torch.from_numpy(bboxes_gt)
                    ret = bbox_overlaps_nearest_3d(bboxes_dt, bboxes_gt)
                    max,idx = ret.max(1)
                    TP,FP,FN,F1 = calculate_F1(max, compensate=(len(bboxes_gt) - len(bboxes_dt)))
                total_TP = total_TP + TP
                total_FP = total_FP + FP
                total_FN = total_FN + FN
        total_precision = total_TP/(total_TP + total_FP)
        total_recall = total_TP/(total_TP + total_FN)
        total_F1 = 2*total_TP / (2*total_TP+total_FN+total_FP)

        info_str = '{}: TP {} FP {} FN {} precision {:.2f} recall {:.2f} F1_score {:.2f}'.format(_dir, total_TP, total_FP, total_FN, total_precision, total_recall, total_F1)
        print(info_str)
        

if __name__ == '__main__':
    main()