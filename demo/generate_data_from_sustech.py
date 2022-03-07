import json,os
from cv2 import bitwise_xor
import numpy as np
import shutil
from tqdm import tqdm
import open3d as o3d

def get_occluded(pcd, anno):
    occlude = []
    for obj in anno:
        # yaw = obj['psr']['rotation']['z']
        # x = obj['psr']['position']['x']
        # y = obj['psr']['position']['y']
        # z = obj['psr']['position']['z']
        # w = obj['psr']['scale']['x']
        # h = obj['psr']['scale']['y']
        # l = obj['psr']['scale']['z']
        # if w > h:
        #     max_dim = w
        # else:
        #     max_dim = h

        # in_count = 0
        # for pts in pcd:
        #     if (pts[2] > (z - l/2)) and (pts[2] < (z + l/2)):
        #         if (abs(pts[0] - x) + abs(pts[1] - y)) < 1.1*max_dim:
        #             in_count = in_count + 1
        
        # vox_size = w*h*l
        # density = in_count / vox_size
        # if density > 70:
        #     occlude.append(0)
        # elif density > 20:
        #     occlude.append(1)
        # else:
        #     occlude.append(2)
        occlude.append(0)
    return occlude

def main():
    dir_list = ['2022-01-20', '2022-02-18-09-40-26_3', '2022-02-18-09-40-49_4', '2022-02-18-09-41-11_5',
                '2022-02-18-09-43-10_10',
                '2022-02-18-09-47-21_21', '2022-02-18-09-47-43_22', '2022-02-18-09-48-06_23',
                '2022-02-18-09-48-28_24', '2022-02-18-09-48-51_25', '2022-02-18-09-49-14_26',
                '2022-02-18-09-49-36_27', '2022-02-18-09-49-59_28', '2022-02-18-09-50-21_29',
                '2022-02-18-09-50-44_30', '2022-02-18-09-51-05_31', '2022-02-18-09-51-25_32',
                '2022-02-18-09-51-46_33', '2022-02-18-09-52-07_34', '2022-02-18-09-52-28_35',
                '2022-02-18-09-52-51_36', '2022-02-18-09-53-14_37', '2022-02-18-09-53-36_38']
    # dir_list = ['2022-02-18-09-48-06_23']

    data_root = '/home/yexiubo/project/SUSTechPOINTS/data_02_18/'
    pcd_root = '/data/2022-02-18-merge/'
    output_dir = '/data/privatedata/2022-02-18/total'
    total_annos = []
    obj_type = dict()
    for _dir in tqdm(dir_list):
        annos = os.listdir(os.path.join(data_root, _dir, 'label'))
        for anno in tqdm(annos):
            anno_path = os.path.join(os.path.join(data_root, _dir, 'label', anno))
            with open(anno_path, 'r') as f:
                anno_data = json.load(f)
            pcd_path = os.path.join(os.path.join(pcd_root, _dir, 'bin', anno.replace('json', 'bin')))
            pcd = np.fromfile(pcd_path, dtype=np.float32, count=-1).reshape(-1, 5)[:,:4]
            occlude = get_occluded(pcd, anno_data)

            with open(os.path.join(output_dir,'label_2', anno.replace('json', 'txt')), 'wt') as f:
                for i in range(len(anno_data)):
                    yaw = anno_data[i]['psr']['rotation']['z']
                    x = anno_data[i]['psr']['position']['x']
                    y = anno_data[i]['psr']['position']['y']
                    z = anno_data[i]['psr']['position']['z']
                    w = anno_data[i]['psr']['scale']['z']
                    h = anno_data[i]['psr']['scale']['y']
                    l = anno_data[i]['psr']['scale']['x']
                    b_x0 = (x + 50 - w/2) * 10
                    b_y0 = (y + 50 - l/2) * 10
                    b_x1 = (x + 50 + w/2) * 10
                    b_y1 = (y + 50 + l/2) * 10
                    if anno_data[i]['obj_type'] == 'Motorcycle' or anno_data[i]['obj_type'] == 'Bicycle':
                        anno_data[i]['obj_type'] = 'Cyclist'
                        b_x0 = (x + 50 - w/2 * 4) * 10
                        b_y0 = (y + 50 - l/2 * 4) * 10
                        b_x1 = (x + 50 + w/2 * 4) * 10
                        b_y1 = (y + 50 + l/2 * 4) * 10
                    elif anno_data[i]['obj_type'] == 'Pedestrian':
                        b_x0 = (x + 50 - w/2 * 8) * 10
                        b_y0 = (y + 50 - l/2 * 8) * 10
                        b_x1 = (x + 50 + w/2 * 8) * 10
                        b_y1 = (y + 50 + l/2 * 8) * 10
                    
                    b_x0 = 0
                    b_x1 = 50
                    b_y0 = 0
                    b_y1 = 50
                    kitti_label_str = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(anno_data[i]['obj_type'], 0, occlude[i], 0,
                    b_x0,b_y0,b_x1,b_y1,h,w,l,x,y,z,yaw,1)
                    # f.write(kitti_label_str + '\n')
                    if anno_data[i]['obj_type'] not in obj_type.keys():
                        obj_type[anno_data[i]['obj_type']] = 0
                    else:
                        obj_type[anno_data[i]['obj_type']] = obj_type[anno_data[i]['obj_type']] + 1
            
            new_pcd_path = os.path.join(output_dir,'velodyne', anno.replace('json', 'bin'))
            # shutil.copyfile(pcd_path, new_pcd_path)
            # pcd.tofile(new_pcd_path)       
        total_annos.extend(annos)
    print(obj_type)
    print(len(total_annos))

def trans_sustech():
    label_dir = '/home/yexiubo/project/SUSTechPOINTS/data_02_18_test/pred/label'
    lidar_dir = '/home/yexiubo/project/SUSTechPOINTS/data_02_18_test/pred/lidar'
    bbox_dir = '/data/privatedata/2022-02-18/testing/pred_2/'
    bin_dir = '/data/privatedata/2022-02-18/testing/velodyne/'

    # #generate pcd files
    # for binfile in tqdm(os.listdir(bin_dir)):
    #     pts = np.fromfile(os.path.join(bin_dir, binfile), dtype=np.float32, count=-1).reshape(-1, 4)
    #     pc = o3d.geometry.PointCloud()
    #     pc.points = o3d.utility.Vector3dVector(pts[:,:3])
    #     if not os.path.exists(lidar_dir):
    #         os.makedirs(lidar_dir)
    #     pcd_file = os.path.join(lidar_dir, binfile.replace('.bin', '.pcd'))
    #     o3d.io.write_point_cloud(pcd_file, pc)

    for file in os.listdir(bbox_dir):
        result_list = []
        with open(os.path.join(bbox_dir, file), 'rt') as f:
            data = f.readlines()
            for i in range(len(data)):
                anno = data[i].split(' ')
                obj_type = anno[0]
                if obj_type == 'Cyclist':
                    obj_type == 'Motorcycle'
                score = anno[15]
                obj = dict()
                pos = dict()
                if np.float(score) > 0.3:
                    obj['obj_id'] = i + 1
                    obj['obj_type'] = obj_type
                    pos['position'] = {'x':np.float(anno[11]), 'y':np.float(anno[12]), 'z':np.float(anno[13])}
                    pos['rotation'] = {'x':0, 'y':0, 'z':np.float(anno[14])}
                    pos['scale'] = {'x':np.float(anno[10]), 'y':np.float(anno[8]), 'z':np.float(anno[9])}
                    obj['psr'] = pos
                    result_list.append(obj)
                else:
                    continue
        file_name = os.path.join(label_dir, file.replace('txt', 'json'))
        with open(file_name, 'w') as f:
            json.dump(result_list, f, indent=2)

def generate_check():
    gt_label_dir = '/home/yexiubo/project/SUSTechPOINTS/data_02_18_test/gt/label'
    gt_lidar_dir = '/home/yexiubo/project/SUSTechPOINTS/data_02_18_test/gt/lidar'
    pred_label_dir = '/home/yexiubo/project/SUSTechPOINTS/data_02_18_test/pred/label'
    pred_lidar_dir = '/home/yexiubo/project/SUSTechPOINTS/data_02_18_test/pred/lidar'

    output_label_dir = '/home/yexiubo/project/SUSTechPOINTS/data_02_18_test/check/label'
    output_lidar_dir = '/home/yexiubo/project/SUSTechPOINTS/data_02_18_test/check/lidar'
    for file in tqdm(os.listdir(pred_label_dir)):
        key = file.split('.')[0]
        pred_key = str(int(key) + 1)
        old_gt_label_file = os.path.join(gt_label_dir, key+'.json')
        old_pred_label_file = os.path.join(pred_label_dir, key+'.json')
        old_gt_lidar_file = os.path.join(gt_lidar_dir, key+'.pcd')
        old_pred_lidar_file = os.path.join(pred_lidar_dir, key+'.pcd')

        new_gt_label_file = os.path.join(output_label_dir, key+'.json')
        new_pred_label_file = os.path.join(output_label_dir, pred_key+'.json')
        new_gt_lidar_file = os.path.join(output_lidar_dir, key+'.pcd')
        new_pred_lidar_file = os.path.join(output_lidar_dir, pred_key+'.pcd')
        shutil.copyfile(old_gt_label_file, new_gt_label_file)
        shutil.copyfile(old_gt_lidar_file, new_gt_lidar_file)
        shutil.copyfile(old_pred_label_file, new_pred_label_file)
        shutil.copyfile(old_pred_lidar_file, new_pred_lidar_file)


if __name__ == '__main__':
    # main()
    trans_sustech()
    generate_check()
