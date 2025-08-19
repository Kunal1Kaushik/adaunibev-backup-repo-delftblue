eval_interval = 1
samples_per_gpu = 1
workers_per_gpu = 2
max_epochs = 18
save_interval = 1
log_interval = 18
fusion_method = 'avg'
feature_norm = None
modality_dropout_prob = 0.0
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes_unibev/'
train_ann_file = 'nuscenes_infos_train.pkl'
val_ann_file = 'nuscenes_infos_val.pkl'
work_dir = './outputs/train_2500weights/unibev_avg_dim_256_nus_LC_full'
load_from = 'unibev_avgfusion_moddrop_trained/unibev_avgfusion_moddrop_trained.pth'
resume_from = None
plugin = True
plugin_dir = 'mmdet3d/unibev_plugin/'
point_cloud_range = [-54, -54, -5, 54, 54, 3]
voxel_size = [0.075, 0.075, 0.2]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
file_client_args = dict(backend='disk')
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_scale = (1600, 900)
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
dec_scale_factor = 1
_encoder_layers_ = 3
_num_levels_ = 1
_num_points_in_pillar_cam_ = 4
_num_points_in_pillar_lidar_ = 4
bev_h_ = 200
bev_w_ = 200
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
runner = dict(type='EpochBasedRunner', max_epochs=18)
train_pipeline = [
    dict(
        type='LoadPointsFromFileCorruption',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweepsCorruption',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=dict(backend='disk'),
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='PointsRangeFilter', point_cloud_range=[-54, -54, -5, 54, 54, 3]),
    dict(
        type='ObjectRangeFilter', point_cloud_range=[-54, -54, -5, 54, 54, 3]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]),
    dict(type='PointShuffle'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]),
    dict(
        type='CustomCollect3D',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFileCorruption',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweepsCorruption',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=dict(backend='disk'),
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ],
                with_label=False),
            dict(type='CustomCollect3D', keys=['points', 'img'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes_unibev/',
        ann_file='data/nuscenes_unibev/nuscenes_infos_train.pkl',
        load_interval=4,
        pipeline=[
            dict(
                type='LoadPointsFromFileCorruption',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(
                type='LoadPointsFromMultiSweepsCorruption',
                sweeps_num=10,
                use_dim=[0, 1, 2, 3, 4],
                file_client_args=dict(backend='disk'),
                pad_empty_sweeps=True,
                remove_close=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-54, -54, -5, 54, 54, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-54, -54, -5, 54, 54, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='CustomCollect3D',
                keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes_unibev/',
        ann_file='data/nuscenes_unibev/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=[
            dict(
                type='LoadPointsFromFileCorruption',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(
                type='LoadPointsFromMultiSweepsCorruption',
                sweeps_num=10,
                use_dim=[0, 1, 2, 3, 4],
                file_client_args=dict(backend='disk'),
                pad_empty_sweeps=True,
                remove_close=True),
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'trailer', 'bus',
                            'construction_vehicle', 'bicycle', 'motorcycle',
                            'pedestrian', 'traffic_cone', 'barrier'
                        ],
                        with_label=False),
                    dict(type='CustomCollect3D', keys=['points', 'img'])
                ])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes_unibev/',
        ann_file='data/nuscenes_unibev/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=[
            dict(
                type='LoadPointsFromFileCorruption',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(
                type='LoadPointsFromMultiSweepsCorruption',
                sweeps_num=10,
                use_dim=[0, 1, 2, 3, 4],
                file_client_args=dict(backend='disk'),
                pad_empty_sweeps=True,
                remove_close=True),
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'trailer', 'bus',
                            'construction_vehicle', 'bicycle', 'motorcycle',
                            'pedestrian', 'traffic_cone', 'barrier'
                        ],
                        with_label=False),
                    dict(type='CustomCollect3D', keys=['points', 'img'])
                ])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
model = dict(
    type='UniBEV',
    use_grid_mask=True,
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=[0.075, 0.075, 0.2],
        point_cloud_range=[-54, -54, -5, 54, 54, 3],
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[128, 128],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=1,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='UniBEV_Head',
        bev_h=200,
        bev_w=200,
        num_query=900,
        num_classes=10,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='UniBEVTransformer',
            embed_dims=256,
            fusion_method='avg',
            drop_modality=0.0,
            feature_norm=None,
            img_encoder=dict(
                type='ImgEncoder',
                num_layers=3,
                pc_range=[-54, -54, -5, 54, 54, 3],
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='ImgLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttentionImg',
                            pc_range=[-54, -54, -5, 54, 54, 3],
                            deformable_attention=dict(
                                type='MSDeformableAttention3DImg',
                                embed_dims=256,
                                num_points=8,
                                num_levels=1),
                            embed_dims=256)
                    ],
                    ffn_cfgs=dict(type='FFN', embed_dims=256),
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            pts_encoder=dict(
                type='PtsEncoder',
                num_layers=3,
                pc_range=[-54, -54, -5, 54, 54, 3],
                num_points_in_pillar_lidar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='PtsLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttentionPts',
                            pc_range=[-54, -54, -5, 54, 54, 3],
                            deformable_attention=dict(
                                type='MSDeformableAttention3DPts',
                                embed_dims=256,
                                num_points=8,
                                num_levels=1),
                            embed_dims=256)
                    ],
                    ffn_cfgs=dict(type='FFN', embed_dims=256),
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1)
                    ],
                    ffn_cfgs=dict(type='FFN', embed_dims=256),
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-54, -54, -5, 54, 54, 3],
            max_num=300,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=200,
            col_num_embed=200),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='HungarianAssigner3DBEVFormer',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1CostBEVFormer', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-54, -54, -5, 54, 54, 3]))))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(
            type='LoadPointsFromFileCorruption',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5),
        dict(
            type='LoadPointsFromMultiSweepsCorruption',
            sweeps_num=10,
            use_dim=[0, 1, 2, 3, 4],
            file_client_args=dict(backend='disk'),
            pad_empty_sweeps=True,
            remove_close=True),
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(
            type='NormalizeMultiviewImage',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1600, 900),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(type='PadMultiViewImage', size_divisor=32),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'trailer', 'bus',
                        'construction_vehicle', 'bicycle', 'motorcycle',
                        'pedestrian', 'traffic_cone', 'barrier'
                    ],
                    with_label=False),
                dict(type='CustomCollect3D', keys=['points', 'img'])
            ])
    ])
optimizer = dict(
    type='AdamW',
    lr=0.001,
    paramwise_cfg=dict(
        custom_keys=dict(fusion_weight_predictor=dict(lr_mult=1.0))),
    weight_decay=0.0)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='fixed', warmup=None)
total_epochs = 18
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=18,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
custom_hooks = [dict(type='CheckpointLateStageHook', start=1, priority=60)]
workflow = [('train', 1), ('val', 1)]
gpu_ids = range(0, 2)
