import numpy as np
import pickle

def main():
    file = '/home/yexiubo/project/mmdetection3d/demo/1645148360990170543.bin'
    pcd = np.fromfile(file, dtype=np.float32, count=-1).reshape(-1, 5)
    print(pcd)

    # file = 'data/kitti/kitti_infos_train.pkl'
    file = 'data/nuscenes/nuscenes_infos_10sweeps_train.pkl'
    with open(file, 'rb') as f:
        info = pickle.load(f)
        print(info)

if __name__ == '__main__':
    main()