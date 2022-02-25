from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_nearest_3d
import torch
import numpy as np

def main():
    bboxes1 = np.array([[1,1,1,2,2,2,0]], dtype=np.float64)
    bboxes2 = np.array([[1,1,1,1,1,1,0]], dtype=np.float64)
    print(bboxes2.shape)
    bboxes1 = torch.from_numpy(bboxes1)
    bboxes2 = torch.from_numpy(bboxes2)
    ret = bbox_overlaps_nearest_3d(bboxes2, bboxes1)
    print(bboxes1)
    print(bboxes2)
    print(ret)

if __name__ == '__main__':
    main()