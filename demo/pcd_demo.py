# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from pathlib import Path
import glob

from matplotlib.pyplot import box
from mmdet3d.apis import inference_detector, init_model, show_result_meshlab
from os import path as osp
import numpy as np
from tqdm import tqdm
import os

class_names = [
    'Car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'Cyclist',
    'Cyclist', 'Pedestrian', 'traffic_cone', 'barrier'
]
kitti_class_names = ['Car', 'Pedestrian', 'Cyclist']

def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device, prune_model=False)
    
    # test a single image
    result, data = inference_detector(model, args.pcd)
    boxes_3d = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
    labels_3d = result[0]['pts_bbox']['labels_3d'].numpy()
    scores_3d = result[0]['pts_bbox']['scores_3d'].numpy()
    
    ## kitti_format
    # kitti_file = args.pcd.split('/')[-1].split('.bin')[0] + '.txt'
    # with open(kitti_file, 'wt') as f:
    #     for i in range(len(labels_3d)):
    #         obj_type = class_names[labels_3d[i]]
    #         b_x0 = 0
    #         b_x1 = 50
    #         b_y0 = 0
    #         b_y1 = 50
    #         kitti_label_str = '{} {} {} {} {} {} {} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(obj_type, 0, 0, 0,
    #                 b_x0,b_y0,b_x1,b_y1,boxes_3d[i][3],boxes_3d[i][5],boxes_3d[i][4],boxes_3d[i][0],boxes_3d[i][1],boxes_3d[i][2]+boxes_3d[i][5]/2,boxes_3d[i][6],scores_3d[i])
    #         f.write(kitti_label_str + '\n')

    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='det')

def demo():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    args = parser.parse_args()

    results = {}

    root_path = Path(args.data_path)
    ext = args.ext
    data_file_list = glob.glob(str(root_path / f'*{ext}')) if root_path.is_dir() else [root_path]

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    for data_path in tqdm(data_file_list):
        # test a single image
        pred_results = [{}]
        result, data = inference_detector(model, data_path)
        pred_results[0]['boxes_3d'] = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
        pred_results[0]['scores_3d'] = result[0]['pts_bbox']['scores_3d'].numpy()
        pred_results[0]['labels_3d'] = result[0]['pts_bbox']['labels_3d'].numpy() + 1
        results[data_path] = pred_results
        pts_filename = data['img_metas'][0][0]['pts_filename']
        data['img_metas'][0][0]['pts_filename'] = 'mmdet3d_' + osp.split(pts_filename)[-1]
    result_file = args.data_path.split('/')[3] + '_0.npy'
    np.save(result_file, results)

    # # kitti_format
    # for data_path in tqdm(data_file_list):
    #     # test a single image
    #     result, data = inference_detector(model, data_path)
    #     pred_dir = 'data/2022-02-18/testing/pred_2/'
    #     boxes_3d = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
    #     labels_3d = result[0]['pts_bbox']['labels_3d'].numpy()
    #     scores_3d = result[0]['pts_bbox']['scores_3d'].numpy()
    #     kitti_file = data_path.split('/')[-1].split('.bin')[0] + '.txt'
    #     with open(os.path.join(pred_dir, kitti_file), 'wt') as f:
    #         for i in range(len(labels_3d)):
    #             obj_type = class_names[labels_3d[i]]
    #             b_x0 = 0
    #             b_x1 = 50
    #             b_y0 = 0
    #             b_y1 = 50
    #             kitti_label_str = '{} {} {} {} {} {} {} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(obj_type, 0, 0, 0,
    #                     b_x0,b_y0,b_x1,b_y1,boxes_3d[i][3],boxes_3d[i][5],boxes_3d[i][4],boxes_3d[i][0],boxes_3d[i][1],boxes_3d[i][2]+boxes_3d[i][5]/2,-boxes_3d[i][6]-np.pi/2,scores_3d[i])
    #             f.write(kitti_label_str + '\n')

    print('Demo done.')

if __name__ == '__main__':
    # main()
    demo()
