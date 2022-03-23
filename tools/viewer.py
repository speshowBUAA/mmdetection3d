"""
author: hova88
date: 2021/03/16
"""
import numpy as np 
import numpy as np
from visual_tools import draw_clouds_with_boxes, draw_clouds
from ground_tools import my_ransac
import open3d as o3d
import yaml

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


def dataloader(cloud_path , boxes_path):
    # cloud = np.loadtxt(cloud_path).reshape(-1,5)
    cloud = np.fromfile(cloud_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
    boxes = np.loadtxt(boxes_path).reshape(-1,7)
    # cloud = np.fromfile(cloud_path, dtype=np.float32, count=-1).reshape([-1,4])
    # boxes = np.loadtxt(boxes_path).reshape(-1,9)[:, :7]
    return cloud , boxes 

def remove_ground(cloud):
    indices, model = my_ransac.my_ransac_v2(cloud[:, :3],
                                distance_threshold=0.3,)
    o_indices = []
    for i in range(cloud.shape[0]):
        if i not in indices:
            o_indices.append(i)
    object_cloud = cloud[o_indices, :]
    return object_cloud
    
def main():
    # cloud_path = '/data/bags/bin/1642667837101218365.bin'
    # cloud_path = "../data/2022-02-18/training/velodyne_reduced/1645148890102077243.bin"
    cloud_path = "../1645179542771259609.bin"
    boxes_path = '../boxes.txt'
    # boxes_path = '../gt_boxes.txt'
    cloud ,boxes = dataloader(cloud_path, boxes_path)
    draw_clouds_with_boxes(cloud ,boxes)

def test():
    # path = "../test/testdata/voxel.npy"
    # cloud = np.load(path).reshape(-1, 4)
    # print(cloud.shape)
    # draw_clouds(cloud)

    # path = "../test/testdata/np_raw_feats.npy"
    # cloud = np.load(path).reshape(-1, 10)
    # cloud = cloud [:, :4]
    # print(cloud.shape)
    # print(np.max(cloud, axis=0))
    # draw_clouds(cloud)

    # path = "../test/testdata/0_Model_pfe_input_gather_feature.txt"
    # cloud = np.loadtxt(path).reshape(-1, 10)
    # print(np.max(cloud, axis=0))
    # cloud = cloud [:, :4]
    # print(cloud.shape)
    # print(np.max(cloud, axis=0))
    # draw_clouds(cloud)

    # path = "../test/testdata/0_dev_points.txt"
    # cloud = np.loadtxt(path).reshape(-1, 4)
    # print(cloud.shape)
    # print(np.max(cloud, axis=0))
    # draw_clouds(cloud)

    # path = "../test/testdata/0_dev_pfe_gather_feature.txt"
    # cloud = np.loadtxt(path).reshape(-1, 10)
    # cloud = cloud [:, :4]
    # print(cloud.shape)
    # print(np.max(cloud, axis=0))
    # draw_clouds(cloud)

    # path = "../test/testdata/1606813517797756000.bin"
    # cloud = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
    # cloud_out = cloud.reshape(-1)
    # cloud_out.tofile('1606813517797756000_dim4.bin')
    # print(cloud_out.shape)
    # draw_clouds(cloud)

    # path = "../test/testdata/pts.bin"
    # cloud = np.fromfile(path, dtype=np.float32, count=-1)
    # cloud = cloud.reshape([-1,4])
    # print(cloud.shape)
    # draw_clouds(cloud)

    # path = "../data/nuscenes/nuscenes_gt_database/8022d9bd957549778ec8a483fe40204d_car_0.bin"
    # cloud = np.fromfile(path, dtype=np.float32, count=-1)
    # cloud = cloud.reshape([-1,5])
    # print(cloud.shape)
    # draw_clouds(cloud)

    # path = "../data/2022-02-18/2022-02-18_gt_database/1645148360990170543_Pedestrian_18.bin"
    # cloud = np.fromfile(path, dtype=np.float32, count=-1)
    # cloud = cloud.reshape([-1,4])
    # print(cloud.shape)
    # draw_clouds(cloud)

    path = "../data/2022-02-18/training/velodyne/1645148890102077243.bin"
    cloud = np.fromfile(path, dtype=np.float32, count=-1)
    cloud = cloud.reshape([-1,4])
    print(cloud.shape)

    # draw_clouds(cloud)

if __name__ == "__main__":
    main()
    # test()