import json,os
import numpy as np
import shutil
from tqdm import tqdm

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
    dir_list = ['2022-01-20', '2022-02-18-09-39-16_0', '2022-02-18-09-40-26_3', '2022-02-18-09-40-49_4',
                '2022-02-18-09-47-21_21', '2022-02-18-09-47-43_22', '2022-02-18-09-48-06_23',
                '2022-02-18-09-48-28_24', '2022-02-18-09-48-51_25', '2022-02-18-09-49-14_26',
                '2022-02-18-09-49-36_27', '2022-02-18-09-49-59_28', '2022-02-18-09-50-21_29',
                '2022-02-18-09-50-44_30', '2022-02-18-09-51-05_31', '2022-02-18-09-53-14_37',
                '2022-02-18-09-53-36_38']
    # dir_list = ['2022-02-18-09-48-06_23']

    data_root = '/home/yexiubo/project/SUSTechPOINTS/data/'
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
                    z = anno_data[i]['psr']['position']['z'] - anno_data[i]['psr']['scale']['z'] / 2
                    w = anno_data[i]['psr']['scale']['z']
                    h = anno_data[i]['psr']['scale']['y']
                    l = anno_data[i]['psr']['scale']['x']
                    if anno_data[i]['obj_type'] == 'Motorcycle' or anno_data[i]['obj_type'] == 'Bicycle':
                        anno_data[i]['obj_type'] = 'Cyclist'
                    kitti_label_str = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(anno_data[i]['obj_type'], 0, occlude[i], 0,
                    0,0,0,0,h,w,l,x,y,z,yaw,1)
                    f.write(kitti_label_str + '\n')
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
    
if __name__ == '__main__':
    main()
