import numpy as np
import pickle

def main():
    # file = '/home/yexiubo/project/mmdetection3d/demo/1645148360990170543.bin'
    # pcd = np.fromfile(file, dtype=np.float32, count=-1).reshape(-1, 5)
    # print(pcd)

    # file = 'data/kitti/kitti_infos_train.pkl'
    # file = 'data/2022-02-18/2022-02-18_infos_train.pkl'
    # file = 'data/nuscenes_mini/nuscenes_infos_train.pkl'
    # with open(file, 'rb') as f:
    #     info = pickle.load(f)
    #     print(info.keys())
    #     # print(info[0].keys())

    file = 'data/2022-02-18/2022-02-18_infos_train.pkl'
    with open(file, 'rb') as f:
        data_infos = pickle.load(f)
        # print(info[0])
        gt_annos = [info['annos'] for info in data_infos]
        print(gt_annos)    

if __name__ == '__main__':
    main()