_base_ = [
    '../_base_/models/hv_pointpillars_fpn_nus.py',
    '../_base_/datasets/nus-3d-custom.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]
# model settings
model = dict(
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range = [-25, -25, -5, 25, 25, 3],
        max_voxels=(30000, 5000)),
    pts_voxel_encoder = dict(
        point_cloud_range = [-25, -25, -5, 25, 25, 3],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[200, 200]),
    pts_backbone = dict(
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01)),
    pts_neck=dict(
        _delete_=True,
        type='SECONDFPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        in_channels=384,
        feat_channels=384,
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            # ranges=[
            #     [-49.6, -49.6, -1.80032795, 49.6, 49.6, -1.80032795],
            #     [-49.6, -49.6, -1.74440365, 49.6, 49.6, -1.74440365],
            #     [-49.6, -49.6, -1.68526504, 49.6, 49.6, -1.68526504],
            #     [-49.6, -49.6, -1.67339111, 49.6, 49.6, -1.67339111],
            #     [-49.6, -49.6, -1.61785072, 49.6, 49.6, -1.61785072],
            #     [-49.6, -49.6, -1.80984986, 49.6, 49.6, -1.80984986],
            #     [-49.6, -49.6, -1.763965, 49.6, 49.6, -1.763965],
            # ],
            ranges=[
                [-24.6, -24.6, -1.80032795, 24.6, 24.6, -1.80032795],
                [-24.6, -24.6, -1.74440365, 24.6, 24.6, -1.74440365],
                [-24.6, -24.6, -1.68526504, 24.6, 24.6, -1.68526504],
                [-24.6, -24.6, -1.67339111, 24.6, 24.6, -1.67339111],
                [-24.6, -24.6, -1.61785072, 24.6, 24.6, -1.61785072],
                [-24.6, -24.6, -1.80984986, 24.6, 24.6, -1.80984986],
                [-24.6, -24.6, -1.763965, 24.6, 24.6, -1.763965],
            ],
            sizes=[
                [1.95017717, 4.60718145, 1.72270761],  # car
                [2.4560939, 6.73778078, 2.73004906],  # truck
                [2.87427237, 12.01320693, 3.81509561],  # trailer
                [0.60058911, 1.68452161, 1.27192197],  # bicycle
                [0.66344886, 0.7256437, 1.75748069],  # pedestrian
                [0.39694519, 0.40359262, 1.06232151],  # traffic_cone
                [2.49008838, 0.48578221, 0.98297065],  # barrier
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True)))

# #custom hook config
# custom_hooks = [
#     dict(type='SparseTrainingHook')
# ]