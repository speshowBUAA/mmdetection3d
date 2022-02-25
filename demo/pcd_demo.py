# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_model, show_result_meshlab

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
    # boxes_3d = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
    # labels_3d = result[0]['pts_bbox']['labels_3d'].numpy() + 1
    # scores_3d = result[0]['pts_bbox']['scores_3d'].numpy()
    # for i in range(len(labels_3d)):
    #     if labels_3d[i] == 1:
    #         if scores_3d[i] > 0.5:
    #             print(boxes_3d[i,6])

    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='det')


if __name__ == '__main__':
    main()
