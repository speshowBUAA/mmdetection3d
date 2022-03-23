voxel_size = [0.25, 0.25, 8]
model = dict(
    type='MVXFasterRCNN',
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=[-25, -25, -5, 25, 25, 3],
        voxel_size=[0.25, 0.25, 8],
        max_voxels=(30000, 5000)),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=[0.25, 0.25, 8],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-25, -25, -5, 25, 25, 3],
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[200, 200]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='BN2d', eps=0.001, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='BN2d', eps=0.001, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-24.6, -24.6, -1.80032795, 24.6, 24.6, -1.80032795],
                    [-24.6, -24.6, -1.74440365, 24.6, 24.6, -1.74440365],
                    [-24.6, -24.6, -1.68526504, 24.6, 24.6, -1.68526504],
                    [-24.6, -24.6, -1.67339111, 24.6, 24.6, -1.67339111],
                    [-24.6, -24.6, -1.61785072, 24.6, 24.6, -1.61785072],
                    [-24.6, -24.6, -1.80984986, 24.6, 24.6, -1.80984986],
                    [-24.6, -24.6, -1.763965, 24.6, 24.6, -1.763965]],
            sizes=[[1.95017717, 4.60718145, 1.72270761],
                   [2.4560939, 6.73778078, 2.73004906],
                   [2.87427237, 12.01320693, 3.81509561],
                   [0.60058911, 1.68452161, 1.27192197],
                   [0.66344886, 0.7256437, 1.75748069],
                   [0.39694519, 0.40359262, 1.06232151],
                   [2.49008838, 0.48578221, 0.98297065]],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500)))
dataset_type = 'KittiDataset'
data_root = 'data/2022-03-18/'
class_names = ['Pedestrian']
point_cloud_range = [-25, -25, -5, 25, 25, 3]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root='data/2022-03-18/',
    info_path='data/2022-03-18/2022-03-18_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=['Pedestrian'],
    sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6))
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=dict(backend='disk')),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(
        type='ObjectRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=['Pedestrian']),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Pedestrian'],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian'],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='KittiDataset',
            data_root='data/2022-03-18/',
            ann_file='data/2022-03-18/2022-03-18_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    file_client_args=dict(backend='disk')),
                dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.78539816, 0.78539816],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0]),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[-50, -50, -5, 50, 50, 3]),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-50, -50, -5, 50, 50, 3]),
                dict(type='PointShuffle'),
                dict(type='DefaultFormatBundle3D', class_names=['Pedestrian']),
                dict(
                    type='Collect3D',
                    keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            modality=dict(use_lidar=True, use_camera=False),
            classes=['Pedestrian'],
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type='KittiDataset',
        data_root='data/2022-03-18/',
        ann_file='data/2022-03-18/2022-03-18_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[-50, -50, -5, 50, 50, 3]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Pedestrian'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Pedestrian'],
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='KittiDataset',
        data_root='data/2022-03-18/',
        ann_file='data/2022-03-18/2022-03-18_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[-50, -50, -5, 50, 50, 3]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Pedestrian'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Pedestrian'],
        test_mode=True,
        box_type_3d='LiDAR'))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[20, 23])
momentum_config = None
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs_0318/work_dir_rotscaletrans'
load_from = 'work_dir_rotscaletrans/epoch_24.pth'
resume_from = None
workflow = [('train', 1)]
evaluation = dict(interval=25)
gpu_ids = [0]
