import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from torch.nn import functional as F
from mmdet3d.core import bbox3d2result
from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from mmdet3d.unibev_plugin.models.utils.grid_mask import GridMask
from mmcv.ops import Voxelization
import time
import copy
import numpy as np
import mmdet3d


@DETECTORS.register_module()
class UniBEVV(MVXTwoStageDetector):
    """
    UniBEV model:
        a multi-modal fusion model based on BEVFormer_Deformable(cam) and BEVVoxelDetr (lidar).
        using unified bev query to build BEV embedding from image features and point features
        todo
        # temporal information is not applied.
        # video_test_mode (bool): Decide whether to use temporal information during inference.
    Args:
    """

    def __init__(self,
                 use_lidar = True,
                 use_camera = True,
                 use_radar = False,

                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,

                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):
        super(UniBEV,
              self).__init__(pts_voxel_layer,
                             pts_voxel_encoder,
                             pts_middle_encoder,
                             pts_fusion_layer,
                             img_backbone,
                             pts_backbone,
                             img_neck,
                             pts_neck,
                             pts_bbox_head,
                             img_roi_head,
                             img_rpn_head,
                             train_cfg,
                             test_cfg,
                             pretrained)
        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_radar = use_radar

        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

        if radar_voxel_layer:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        if radar_voxel_encoder:
            self.radar_voxel_encoder = builder.build_voxel_encoder(radar_voxel_encoder)
        if radar_middle_encoder:
            self.radar_middle_encoder = builder.build_middle_encoder(radar_middle_encoder)

        self.fusion_method = pts_bbox_head['transformer'].get('fusion_method', None)

    def extract_img_feat(self, img, img_metas=None):
        """Extract features of images."""
        if img is not None:
            B = img.size(0)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_pts_feat(self, pts):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_radar_feat(self, radar, img_metas):
        """Extract features of points."""

        voxels, num_points, coors = self.radar_voxelize(radar)

        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        if self.with_pts_backbone:
            x = self.radar_middle_encoder(voxel_features, coors, batch_size)
            x = self.pts_backbone(x)
            if self.with_pts_neck:
                x = self.pts_neck(x)
            return x
        else:
            x = self.radar_middle_encoder(voxel_features, coors, batch_size)
            return [x]

    def extract_feat(self, img, points, radar_points, img_metas=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas) if self.use_camera else None
        pts_feats = self.extract_pts_feat(points) if self.use_lidar else None
        radar_feats = self.extract_radar_feat(radar_points, img_metas) if self.use_radar else None

        return img_feats, pts_feats, radar_feats

    @torch.no_grad()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @torch.no_grad()
    @force_fp32()
    def radar_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_dummy(self, img, points):
        dummy_metas = None
        return self.forward_test(img=img, points=points, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      radar=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.use_camera:
            assert img is not None
        if self.use_lidar:
            assert points is not None
        if self.use_radar:
            assert radar is not None
        # print(img.shape)
        # print(len(img_metas))
        # print(img_metas[0].keys())
        # if img is not None:
        #     len_queue = img.size(1)
        #     img = img[:, -1, ...]
        # else:
        #     len_queue = 3
        #     img = None
        # img_metas = [each[len_queue - 1] for each in img_metas]
        img_feats, lidar_feats, radar_feats = self.extract_feat(img=img, points=points, radar_points= radar, img_metas=img_metas)

        losses = dict()
        if self.use_lidar == True and self.use_radar == False:
            pts_feats = lidar_feats
        elif self.use_lidar == False and self.use_radar == True:
            pts_feats = radar_feats
        elif self.use_lidar == True and self.use_radar == True:
            raise ValueError('Unsupported Modality Mode: Cam: {}, Lidar:{}, Radar:{}'.format(self.use_camera, self.use_lidar, self.use_radar))
        else:
            pts_feats = None

        outs = self.pts_bbox_head(img_feats, pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses_pts = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, points=None, radar=None, **kwargs):

        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        # num_augs = len(points)
        # if num_augs != len(img_metas):
        #     raise ValueError(
        #         'num of augmentations ({}) != num of image meta ({})'.format(
        #             len(points), len(img_metas)))

        img = [img] if img is None else img
        points = [points] if points is None else points
        radar = [radar] if radar is None else radar

        bbox_results, bev_embeds = self.simple_test(
            points[0], img_metas[0], img[0], radar[0], **kwargs)
        return bbox_results

    def simple_test(self, points, img_metas, img=None, radar = None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, lidar_feats, radar_feats = self.extract_feat(img=img, points=points, radar_points=radar, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]

        if self.use_lidar == True and self.use_radar == False:
            pts_feats = lidar_feats
        elif self.use_lidar == False and self.use_radar == True:
            pts_feats = radar_feats
        elif self.use_lidar == True and self.use_radar == True:
            raise ValueError('Unsupported Modality Mode: Cam: {}, Lidar:{}, Radar:{}'.format(self.use_camera, self.use_lidar, self.use_radar))
        else:
            pts_feats = None

        #bev_masks = self.points_to_bev_mask_batch(points, bev_size=(200, 200), pc_range=[-54, -54, -5, 54, 54, 3])
        #for i in range(len(img_metas)):
            #img_metas[i]['no_points'] = bev_masks[i]

        outs = self.pts_bbox_head(img_feats, pts_feats, img_metas)

        bbox_list_head = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_pts = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list_head
        ]

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list, outs['bev_embed']

    def get_bev_embed_feats(self, points, img_metas, img=None, radar = None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats, radar_feats = self.extract_feat(img=img, points=points, radar_points=radar, img_metas=img_metas)

        outs = self.pts_bbox_head(img_feats, pts_feats, img_metas)

        #return outs['pts_bev_embed'], outs['img_bev_embed']
        return outs['pts_bev_embed']

    def get_pts_middle_encoder(self, pts):
        """Test function without augmentaiton."""
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)

        return x

    def points_to_bev_mask_batch(self, points_list, 
                                bev_size=(200, 200), 
                                pc_range=[-54, -54, -5, 54, 54, 3]):
        """
        points_list: list of (N_i, 5) tensors - each point cloud with 5 features per point
        bev_size: tuple (H, W) output BEV size
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        
        Returns:
            bev_masks: (B, H, W) uint8 tensor, 1 = no points, 0 = occupied
        """
        B = len(points_list)
        H, W = bev_size
        x_min, y_min, _, x_max, y_max, _ = pc_range

        # Get device from first tensor
        device = points_list[0].device  
        bev_masks = torch.ones((B, H, W), device=device, dtype=torch.uint8)

        for b in range(B):
            pts = points_list[b][:, :2]  # (N_i, 2), x and y only

            # Normalize to [0, 1)
            x_norm = (pts[:, 0] - x_min) / (x_max - x_min)
            y_norm = (pts[:, 1] - y_min) / (y_max - y_min)

            # Keep only points within the range
            mask_in_range = (x_norm >= 0) & (x_norm < 1) & (y_norm >= 0) & (y_norm < 1)
            x_norm = x_norm[mask_in_range]
            y_norm = y_norm[mask_in_range]

            # Convert to BEV grid indices
            i = (y_norm * H).long().clamp(0, H - 1)
            j = (x_norm * W).long().clamp(0, W - 1)

            bev_masks[b, i, j] = 0  # set occupied

        return bev_masks


@DETECTORS.register_module()
class UniBEV(MVXTwoStageDetector):
    """
    UniBEV model:
        a multi-modal fusion model based on BEVFormer_Deformable(cam) and BEVVoxelDetr (lidar).
        using unified bev query to build BEV embedding from image features and point features
        todo
        # temporal information is not applied.
        # video_test_mode (bool): Decide whether to use temporal information during inference.
    Args:
    """

    def __init__(self,
                 use_lidar = True,
                 use_camera = True,
                 use_radar = False,

                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,

                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):
        super(UniBEV,
              self).__init__(pts_voxel_layer,
                             pts_voxel_encoder,
                             pts_middle_encoder,
                             pts_fusion_layer,
                             img_backbone,
                             pts_backbone,
                             img_neck,
                             pts_neck,
                             pts_bbox_head,
                             img_roi_head,
                             img_rpn_head,
                             train_cfg,
                             test_cfg,
                             pretrained)
        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_radar = use_radar

        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

        if radar_voxel_layer:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        if radar_voxel_encoder:
            self.radar_voxel_encoder = builder.build_voxel_encoder(radar_voxel_encoder)
        if radar_middle_encoder:
            self.radar_middle_encoder = builder.build_middle_encoder(radar_middle_encoder)

        self.fusion_method = pts_bbox_head['transformer'].get('fusion_method', None)

        # Define your fusion weight prediction network
        self.fusion_weight_predictor = FusionWeightPredictor()

        # Freeze all original model parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()  # Optional, to fix BatchNorm running stats

        # 3. Unfreeze fusion_weight_predictor for training
        for param in self.fusion_weight_predictor.parameters():
            param.requires_grad = True

    

    def extract_img_feat(self, img, img_metas=None):
        """Extract features of images."""
        if img is not None:
            B = img.size(0)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_pts_feat(self, pts):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_radar_feat(self, radar, img_metas):
        """Extract features of points."""

        voxels, num_points, coors = self.radar_voxelize(radar)

        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        if self.with_pts_backbone:
            x = self.radar_middle_encoder(voxel_features, coors, batch_size)
            x = self.pts_backbone(x)
            if self.with_pts_neck:
                x = self.pts_neck(x)
            return x
        else:
            x = self.radar_middle_encoder(voxel_features, coors, batch_size)
            return [x]

    def extract_feat(self, img, points, radar_points, img_metas=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas) if self.use_camera else None
        pts_feats = self.extract_pts_feat(points) if self.use_lidar else None
        radar_feats = self.extract_radar_feat(radar_points, img_metas) if self.use_radar else None

        return img_feats, pts_feats, radar_feats

    @torch.no_grad()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @torch.no_grad()
    @force_fp32()
    def radar_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_dummy(self, img, points):
        dummy_metas = None
        return self.forward_test(img=img, points=points, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      radar=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        # 1. Get features from the frozen detector
        with torch.no_grad():
            img_feats, lidar_feats, radar_feats = self.extract_feat(img=img, points=points, radar_points= radar, img_metas=img_metas)

        losses = dict()
        #losses_adaptation = {}
        if self.use_lidar == True and self.use_radar == False:
            pts_feats = lidar_feats
        elif self.use_lidar == False and self.use_radar == True:
            pts_feats = radar_feats
        elif self.use_lidar == True and self.use_radar == True:
            raise ValueError('Unsupported Modality Mode: Cam: {}, Lidar:{}, Radar:{}'.format(self.use_camera, self.use_lidar, self.use_radar))
        else:
            pts_feats = None

        # 2. Create a (B, H, W) mask indicating LiDAR-empty BEV regions
        #lidar_masks = self.points_to_bev_mask_batch(points, bev_size=(200, 200), pc_range=[-54, -54, -5, 54, 54, 3])  # 1 = no points

        '''# Parameters
        bev_h, bev_w = 200, 200
        x_min, x_max = -54.0, 54.0
        y_min, y_max = -54.0, 54.0

        # Create 1D coordinates for each axis
        x_coords = torch.linspace(x_min, x_max, bev_w)  # shape [200]
        y_coords = torch.linspace(y_min, y_max, bev_h)  # shape [200]

        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # both shape [200, 200]

        # Stack to get final position tensor: [200, 200, 2]
        position_mask = torch.stack((x_grid, y_grid), dim=-1)'''

        device = points[0].device

        beam_angles = torch.linspace(10.67, -30.67, steps=32).to(device)
        depth_images = self.pointcloud_to_depth_image_batched(points, beam_angles, h_res=0.4)  # [B, 32, 900]

        corrupted_masks = []
        for depth_img in depth_images:
            depth_img = depth_img.to(device)
            occupancy_mask = None

            #corrupted_mask = self.mark_corrupted_bev_cells_gpu_fast(
            #    depth_img,
            #    beam_angles,
            #    occupancy_mask,
            #    lidar_height=1.84023,
            #    bev_range=(-54, 54, -54, 54),
            #    grid_size=(200, 200),
            #    h_res=0.4,
            #)
            corrupted_mask = self.mark_azimuthal_missing_beams_simple(
                depth_img,
                bev_range=(-54, 54, -54, 54),
                grid_size=(200, 200),
                h_res=0.4,
            )
            corrupted_masks.append(corrupted_mask)

        corrupted_masks_batched = torch.stack(corrupted_masks, dim=0)
        # predict fusion weights and compute fused features
        camera_weights = self.fusion_weight_predictor(corrupted_masks_batched, img_metas)

        for i in range(len(img_metas)):
            img_metas[i]['fusion_weights'] = camera_weights[i]
            #img_metas[i]['use_adaptation'] = True

        outs_adaptation = self.pts_bbox_head(img_feats, pts_feats, img_metas)
        loss_inputs_adaptation = [gt_bboxes_3d, gt_labels_3d, outs_adaptation]
        losses_pts_adaptation = self.pts_bbox_head.loss(*loss_inputs_adaptation, img_metas=img_metas)

        #for i in range(len(img_metas)):
            #img_metas[i]['use_adaptation'] = False

        #outs = self.detector.pts_bbox_head(img_feats, pts_feats, img_metas)
        #loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        #losses_pts = self.detector.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        # 1. Get original detection losses
        #L_det = losses_pts_adaptation['loss_cls'] + losses_pts_adaptation['loss_bbox']
        #L_det_ref = losses_pts['loss_cls'] + losses_pts['loss_bbox']  # From 0.5,0.5 fusion

        #for i in range(len(img_metas)):
            #if img_metas[i]['is_corrupted'] == True:
                #L_total = L_det
            #else:
                #lambda_clean = 1.0
                #L_total = L_det + lambda_clean * torch.clamp(L_det - L_det_ref, min=0.0)         

        # 3. Replace total 'loss' entry with your custom penalized loss
        # (this is what will be used in backward())
        #losses_adaptation['loss'] = L_total
        
        losses.update(losses_pts_adaptation)

        # 4. Optional: remove any other 'loss' entry that might get counted again
        # (this depends on how your trainer sums losses, but safer to keep just one total)
        # Alternatively, leave them for logging, and trust `parse_losses` to use only 'loss'

        return losses


    def forward_test(self, img_metas, img=None, points=None, radar=None, **kwargs):

        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        # num_augs = len(points)
        # if num_augs != len(img_metas):
        #     raise ValueError(
        #         'num of augmentations ({}) != num of image meta ({})'.format(
        #             len(points), len(img_metas)))

        img = [img] if img is None else img
        points = [points] if points is None else points
        radar = [radar] if radar is None else radar

        bbox_results, bev_embeds = self.simple_test(
            points[0], img_metas[0], img[0], radar[0], **kwargs)
        return bbox_results

    def simple_test(self, points, img_metas, img=None, radar = None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, lidar_feats, radar_feats = self.extract_feat(img=img, points=points, radar_points=radar, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]

        if self.use_lidar == True and self.use_radar == False:
            pts_feats = lidar_feats
        elif self.use_lidar == False and self.use_radar == True:
            pts_feats = radar_feats
        elif self.use_lidar == True and self.use_radar == True:
            raise ValueError('Unsupported Modality Mode: Cam: {}, Lidar:{}, Radar:{}'.format(self.use_camera, self.use_lidar, self.use_radar))
        else:
            pts_feats = None

        # 2. Create a (B, H, W) mask indicating LiDAR-empty BEV regions
        #lidar_masks = self.points_to_bev_mask_batch(points, bev_size=(200, 200), pc_range=[-54, -54, -5, 54, 54, 3])  # 1 = no points

        '''# Parameters
        bev_h, bev_w = 200, 200
        x_min, x_max = -54.0, 54.0
        y_min, y_max = -54.0, 54.0

        # Create 1D coordinates for each axis
        x_coords = torch.linspace(x_min, x_max, bev_w)  # shape [200]
        y_coords = torch.linspace(y_min, y_max, bev_h)  # shape [200]

        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # both shape [200, 200]

        # Stack to get final position tensor: [200, 200, 2]
        position_mask = torch.stack((x_grid, y_grid), dim=-1)'''

        device = points[0].device

        beam_angles = torch.linspace(10.67, -30.67, steps=32).to(device)
        depth_images = self.pointcloud_to_depth_image_batched(points, beam_angles, h_res=0.4)  # [B, 32, 900]

        corrupted_masks = []
        for depth_img in depth_images:
            depth_img = depth_img.to(device)
            occupancy_mask = None

            #corrupted_mask = self.mark_corrupted_bev_cells_gpu_fast(
            #    depth_img,
            #    beam_angles,
            #    occupancy_mask,
            #    lidar_height=1.84023,
            #    bev_range=(-54, 54, -54, 54),
            #    grid_size=(200, 200),
            #    h_res=0.4,
            #)
            corrupted_mask = self.mark_azimuthal_missing_beams_simple(
                depth_img,
                bev_range=(-54, 54, -54, 54),
                grid_size=(200, 200),
                h_res=0.4,
            )
            corrupted_masks.append(corrupted_mask)

        corrupted_masks_batched = torch.stack(corrupted_masks, dim=0)
        # predict fusion weights and compute fused features
        camera_weights = self.fusion_weight_predictor(corrupted_masks_batched, img_metas)

        for i in range(len(img_metas)):
            img_metas[i]['fusion_weights'] = camera_weights[i]

        outs = self.pts_bbox_head(img_feats, pts_feats, img_metas)

        bbox_list_head = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_pts = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list_head
        ]

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list, outs['bev_embed']

    def get_bev_embed_feats(self, points, img_metas, img=None, radar = None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats, radar_feats = self.extract_feat(img=img, points=points, radar_points=radar, img_metas=img_metas)

        outs = self.pts_bbox_head(img_feats, pts_feats, img_metas)

        #return outs['pts_bev_embed'], outs['img_bev_embed']
        return outs['pts_bev_embed']

    def get_pts_middle_encoder(self, pts):
        """Test function without augmentaiton."""
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)

        return x

    def points_to_bev_mask_batch(self, points_list, 
                                bev_size=(200, 200), 
                                pc_range=[-54, -54, -5, 54, 54, 3]):
        """
        points_list: list of (N_i, 5) tensors - each point cloud with 5 features per point
        bev_size: tuple (H, W) output BEV size
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        
        Returns:
            bev_masks: (B, H, W) uint8 tensor, 1 = no points, 0 = occupied
        """
        B = len(points_list)
        H, W = bev_size
        x_min, y_min, _, x_max, y_max, _ = pc_range

        # Get device from first tensor
        device = points_list[0].device  
        bev_masks = torch.ones((B, H, W), device=device, dtype=torch.uint8)

        for b in range(B):
            pts = points_list[b][:, :2]  # (N_i, 2), x and y only

            # Normalize to [0, 1)
            x_norm = (pts[:, 0] - x_min) / (x_max - x_min)
            y_norm = (pts[:, 1] - y_min) / (y_max - y_min)

            # Keep only points within the range
            mask_in_range = (x_norm >= 0) & (x_norm < 1) & (y_norm >= 0) & (y_norm < 1)
            x_norm = x_norm[mask_in_range]
            y_norm = y_norm[mask_in_range]

            # Convert to BEV grid indices
            i = (y_norm * H).long().clamp(0, H - 1)
            j = (x_norm * W).long().clamp(0, W - 1)

            bev_masks[b, i, j] = 0  # set occupied

        return bev_masks

    def pointcloud_to_depth_image_batched(self, points_list, beam_angles_deg, h_res=0.4):
        """
        Convert a list of point clouds [N_i, 5] (length B) to depth images [B, H, W]
        Each cell contains range if any point falls in the bin, NaN otherwise.
        """

        B = len(points_list)
        H = beam_angles_deg.shape[0]
        W = int(360 / h_res)

        device = points_list[0].device
        beam_angles_deg = beam_angles_deg.to(device)

        depth_images = torch.full((B, H, W), float('nan'), device=device)

        for b, points in enumerate(points_list):
            points = points.to(device)
            x, y, z = points[:, 0], points[:, 1], points[:, 2]

            vertical_angle = torch.rad2deg(torch.atan2(z, torch.sqrt(x**2 + y**2)))   # [N]
            horizontal_angle = torch.rad2deg(torch.atan2(y, x)) % 360                # [N]

            vert_angle_diff = torch.abs(vertical_angle[:, None] - beam_angles_deg[None, :])  # [N, H]
            v_idx = torch.argmin(vert_angle_diff, dim=1)                                      # [N]
            h_idx = (horizontal_angle / h_res).long().clamp(0, W - 1)                         # [N]

            ranges = torch.sqrt(x**2 + y**2 + z**2)

            depth_images[b, v_idx, h_idx] = ranges

        return depth_images


    '''def pointcloud_to_depth_image_batched(self, points_batched, beam_angles_deg, h_res=0.4):
        """
        Convert batched point clouds [B, N, 5] to binary depth images [B, H, W]
        Each cell is 1.0 if any point falls in the bin, NaN otherwise.
        """

        B, N, _ = points_batched.shape
        H = beam_angles_deg.shape[0]
        W = int(360 / h_res)

        device = points_batched.device
        beam_angles_deg = beam_angles_deg.to(device)

        x, y, z = points_batched[..., 0], points_batched[..., 1], points_batched[..., 2]

        # Calculate vertical and horizontal angles
        vertical_angle = torch.rad2deg(torch.atan2(z, torch.sqrt(x**2 + y**2)))  # [B, N]
        horizontal_angle = torch.rad2deg(torch.atan2(y, x)) % 360               # [B, N]

        # Get nearest vertical beam index
        beam_angles = beam_angles_deg[None, None, :]                            # [1,1,H]
        vert_angle_diff = torch.abs(vertical_angle[..., None] - beam_angles)   # [B, N, H]
        v_idx = torch.argmin(vert_angle_diff, dim=-1)                           # [B, N]

        # Get horizontal bin index
        h_idx = (horizontal_angle / h_res).long().clamp(0, W - 1)               # [B, N]

        # Flatten for indexing
        flat_b = torch.arange(B, device=device).unsqueeze(1).expand(-1, N).flatten()  # [B*N]
        flat_v = v_idx.flatten()
        flat_h = h_idx.flatten()

        # Initialize with NaNs
        depth_image = torch.full((B, H, W), float('nan'), device=device)

        # Mark presence of point as 1.0
        #depth_image[flat_b, flat_v, flat_h] = 1.0

        ranges = torch.sqrt(x**2 + y**2 + z**2)
        depth_image[flat_b, flat_v, flat_h] = ranges.flatten()


        return depth_image'''

    def precompute_azimuth_masks_torch(self, bev_shape, bev_range, h_res, device):
        bev_H, bev_W = bev_shape
        x_min, x_max, y_min, y_max = bev_range  # Should be (x_min, x_max, y_min, y_max)

        # Create grid in BEV coordinates: X goes left to right, Y goes bottom to top
        x_coords = torch.linspace(x_min, x_max, bev_W, device=device)  # width
        y_coords = torch.linspace(y_min, y_max, bev_H, device=device)  # height
        xv, yv = torch.meshgrid(x_coords, y_coords, indexing='xy')  # xv: [W, H], yv: [W, H]

        # Transpose to get [H, W] shape matching BEV format
        #xv, yv = xv.T, yv.T  # Now both are [H, W]

        # Azimuth angle from LiDAR origin to each cell
        azimuths = torch.atan2(yv, xv).rad2deg() % 360  # [H, W]

        # Azimuth bin centers
        az_bin_count = int(360 / h_res)
        bin_centers = torch.arange(0, 360, h_res, device=device)  # [N_bins]

        # Compute masks for each azimuth bin
        azimuth_masks = torch.abs((azimuths[None, :, :] - bin_centers[:, None, None] + 180) % 360 - 180) < (h_res / 2)

        return azimuth_masks  # Shape: [N_bins, H, W]


    def mark_corrupted_bev_cells_gpu_fast(
        self,
        depth_image,            # [32, W]
        beam_angles,            # [32]
        occupancy_mask,         # [H, W]
        lidar_height=1.84023,
        bev_range=(-54, 54, -54, 54),
        grid_size=(200, 200),
        h_res=0.4
    ):
        device = depth_image.device
        H_img, W_img = depth_image.shape
        bev_H, bev_W = grid_size
        x_min, x_max, y_min, y_max = bev_range

        # Create BEV grid
        x_coords = torch.linspace(x_min, x_max, bev_W, device=device)
        y_coords = torch.linspace(y_min, y_max, bev_H, device=device)
        xv, yv = torch.meshgrid(x_coords, y_coords, indexing='xy')
        #xv, yv = xv.T, yv.T  # [H, W]

        dist_ground = torch.sqrt(xv ** 2 + yv ** 2)  # [H, W]
        #elev_angle_grid = torch.atan2(-lidar_height, dist_ground).rad2deg()  # [H, W]
        # [H, W] tensor of elevation angles from LiDAR to each BEV grid center
        elev_angle_grid = torch.atan2(
            torch.full_like(dist_ground, -lidar_height), dist_ground
        ).rad2deg()


        # Closest beam index
        elev_diffs = torch.abs(elev_angle_grid[..., None] - beam_angles.view(1, 1, -1))
        closest_beam_idx = torch.argmin(elev_diffs, dim=2)  # [H, W]

        azimuths = (torch.atan2(yv, xv).rad2deg() % 360).float()  # [H, W]
        az_bin_count = int(360 / h_res)
        az_bin_idx = torch.floor(azimuths / h_res).long().clamp(max=az_bin_count - 1)

        # Build azimuth masks efficiently
        azimuth_masks = torch.stack([
            (az_bin_idx == b) for b in range(az_bin_count)
        ], dim=0)  # [N_bins, H, W]

        # Identify missing beams
        missing_mask = torch.isnan(depth_image)
        beam_idx, az_idx = torch.nonzero(missing_mask, as_tuple=True)

        # Build per-beam azimuth bin mask: [32, 900]
        missing_bin_mask = torch.zeros((H_img, az_bin_count), dtype=torch.bool, device=device)
        missing_bin_mask[beam_idx, az_idx] = True

        # Expand into shape [32, 900, H, W] and [H, W] into [1, 1, H, W]
        dist_ground_batched = dist_ground.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        closest_beam_idx_batched = closest_beam_idx.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        corrupted_mask = torch.zeros((bev_H, bev_W), dtype=torch.bool, device=device)

        beam_angles_rad = torch.deg2rad(beam_angles)

        # Create a grid of azimuth bin centers
        bin_centers = torch.arange(0, 360, h_res, device=device)
        bin_angles_rad = torch.deg2rad(bin_centers)  # [900]
        cos_a = torch.cos(bin_angles_rad)  # [900]
        sin_a = torch.sin(bin_angles_rad)

        for b in range(H_img):
            angle = beam_angles_rad[b]
            if not missing_bin_mask[b].any():
                continue

            selected_bins = torch.nonzero(missing_bin_mask[b], as_tuple=False).flatten()
            selected_masks = azimuth_masks[selected_bins]  # [N, H, W]

            bin_cos = cos_a[selected_bins]
            bin_sin = sin_a[selected_bins]

            if angle < 0:  # Downward beam
                ground_range = lidar_height / torch.abs(torch.sin(angle))
                xg = ground_range * bin_cos
                yg = ground_range * bin_sin
                out_of_range = (
                    (xg < x_min) | (xg > x_max) |
                    (yg < y_min) | (yg > y_max)
                )

                # Corrupt all cells in these azimuth masks
                corrupt_all = selected_masks[out_of_range]

                corrupted_mask |= corrupt_all.any(dim=0)

                # Within range: use grid distance
                in_range = ~out_of_range
                if in_range.any():
                    selected_masks_ir = selected_masks[in_range]
                    dist_thresh = ground_range
                    within_range = dist_ground <= dist_thresh
                    beyond_range = (dist_ground > dist_thresh) & (closest_beam_idx == b)
                    mask_ir = (within_range | beyond_range) & selected_masks_ir.any(dim=0)
                    corrupted_mask |= mask_ir

            elif angle == 0:  # Flat beam
                corrupted_mask |= selected_masks.any(dim=0)

            else:  # Upward beam
                dist_max = 3.16 / torch.sin(angle)
                xg = dist_max * bin_cos
                yg = dist_max * bin_sin
                out_of_range = (
                    (xg < x_min) | (xg > x_max) |
                    (yg < y_min) | (yg > y_max)
                )

                corrupt_all = selected_masks[out_of_range]
                corrupted_mask |= corrupt_all.any(dim=0)

                if (~out_of_range).any():
                    selected_masks_ir = selected_masks[~out_of_range]
                    mask_ir = (dist_ground <= dist_max) & selected_masks_ir.any(dim=0)
                    corrupted_mask |= mask_ir

        return corrupted_mask.to(dtype=torch.uint8)

    def mark_azimuthal_missing_beams_simple(
        self,
        depth_image,            # [32, W]
        bev_range=(-54, 54, -54, 54),
        grid_size=(200, 200),
        h_res=0.4
    ):
        device = depth_image.device
        num_beams, W_img = depth_image.shape
        bev_H, bev_W = grid_size
        x_min, x_max, y_min, y_max = bev_range

        # Create BEV grid
        x_coords = torch.linspace(x_min, x_max, bev_W, device=device)
        y_coords = torch.linspace(y_min, y_max, bev_H, device=device)
        xv, yv = torch.meshgrid(x_coords, y_coords, indexing='xy')  # [H, W]

        # Compute azimuth angle for each cell
        azimuths = (torch.atan2(yv, xv).rad2deg() % 360).float()  # [H, W]
        az_bin_count = int(360 / h_res)
        az_bin_idx = torch.floor(azimuths / h_res).long().clamp(max=az_bin_count - 1)

        # Count how many beams are missing per azimuth bin
        missing_mask = torch.isnan(depth_image)  # [32, W]
        beam_idx, az_idx = torch.nonzero(missing_mask, as_tuple=True)  # Indices of missing
        az_bin_per_pixel = torch.floor((az_idx.float() * 360 / W_img) / h_res).long()
        az_bin_per_pixel = az_bin_per_pixel.clamp(0, az_bin_count - 1)

        # Count missing occurrences per bin
        az_missing_count = torch.bincount(az_bin_per_pixel, minlength=az_bin_count)
        az_missing_ratio = az_missing_count.float() / num_beams  # [N_bins]

        # Assign to BEV grid
        corrupted_mask = az_missing_ratio[az_bin_idx]  # [H, W]

        return corrupted_mask

'''@DETECTORS.register_module() # different weight per grid for empty regions
class FusionWeightPredictor(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, occupancy_mask, position_mask):
        """
        Args:
            occupancy_mask: [B, H, W] - 1 where LiDAR is absent, 0 where LiDAR exists
            position_mask:  [H, W, 2] - static x, y grid coordinates (normalized)

        Returns:
            camera_weights: [B, 1, H, W] - weights ∈ [0.5, 1.0] only where occupancy_mask == 1
        """
        B, H, W = occupancy_mask.shape

        # Prepare input for MLP
        occ = occupancy_mask.unsqueeze(-1)                        # [B, H, W, 1]
        pos = position_mask.unsqueeze(0).expand(B, -1, -1, -1)    # [B, H, W, 2]
        pos = pos.to(occ.device)
        x = torch.cat([occ, pos], dim=-1)                         # [B, H, W, 3]
        x = x.view(B * H * W, 3)                                  # [BHW, 3]

        # Pass through MLP
        logits = self.net(x).view(B, 1, H, W)                     # [B, 1, H, W]

        # Fusion logic: weights = 0.5 if LiDAR exists, else > 0.5
        occ_mask = occupancy_mask.unsqueeze(1)                    # [B, 1, H, W]
        camera_weights = 0.5 + 0.5 * logits * occ_mask            # Only modifies where occ == 1

        return camera_weights'''


'''@DETECTORS.register_module()
class FusionWeightPredictor(nn.Module): #single different weight for empty regions
    def __init__(self, init_value=-0.07):  # Start near your heuristic
        super().__init__()
        # Learnable parameter in [0, 1]; we'll scale it to [0.5, 1.0]
        self.raw_weight = nn.Parameter(torch.tensor(init_value))

    def forward(self, occupancy_mask, position_mask=None):
        """
        Args:
            occupancy_mask: [B, H, W] - 1 where LiDAR is absent
            position_mask:  [H, W, 2] - static x, y grid coordinates (normalized)

        Returns:
            camera_weights: [B, 1, H, W]
        """
        B, H, W = occupancy_mask.shape

        # Get scalar camera weight: w ∈ [0.5, 1.0]
        weight = 0.5 + 0.5 * self.raw_weight.sigmoid()  # Makes it trainable and safe

        # Generate fusion weights map
        camera_weights = torch.full((B, 1, H, W), 0.5, device=occupancy_mask.device)
        camera_weights = camera_weights + (weight - 0.5) * occupancy_mask.unsqueeze(1)

        return camera_weights'''


'''@DETECTORS.register_module()
class FusionWeightPredictor(nn.Module): #different weight per radial ring for empty regions
    def __init__(self, num_rings=2, init_value=-6.0):
        super().__init__()
        self.num_rings = num_rings
        # Learnable raw weights for each ring
        self.raw_weights = nn.Parameter(torch.full((num_rings,), init_value))

    def forward(self, occupancy_mask, position_mask):
        """
        Args:
            occupancy_mask: [B, H, W] — 1 where LiDAR is absent, 0 where present
            position_mask:  [H, W, 2] — x and y grid positions in meters

        Returns:
            camera_weights: [B, 1, H, W]
        """
        B, H, W = occupancy_mask.shape
        device = occupancy_mask.device

        # Compute radial distance grid from center
        xy = position_mask.to(device)  # [H, W, 2]
        radial_dist = torch.norm(xy, dim=-1)  # [H, W]

        # Define ring boundaries
        max_r = radial_dist.max()
        ring_edges = torch.linspace(0, max_r, self.num_rings + 1, device=device)  # [num_rings + 1]

        # Sigmoid weights scaled to [0.5, 1.0]
        scaled_weights = 0.5 + 0.5 * self.raw_weights.sigmoid()  # [num_rings]

        # Initialize weight map
        weight_map = torch.zeros(H, W, device=device)

        for i in range(self.num_rings):
            if i == self.num_rings - 1:
                # Include outermost edge in last ring
                in_ring = (radial_dist >= ring_edges[i]) & (radial_dist <= ring_edges[i + 1])
            else:
                in_ring = (radial_dist >= ring_edges[i]) & (radial_dist < ring_edges[i + 1])
            weight_map[in_ring] = scaled_weights[i]


        # Broadcast and apply occupancy_mask
        camera_weights = torch.full((B, 1, H, W), 0.5, device=device)  # default is 0.5
        camera_weights += (weight_map.unsqueeze(0).unsqueeze(1) - 0.5) * occupancy_mask.unsqueeze(1)

        return camera_weights'''


'''@DETECTORS.register_module()
class FusionWeightPredictor(nn.Module):
    def __init__(self, height=200, width=200, init_value=-6.0):
        super().__init__()
        self.height = height
        self.width = width

        # One learnable weight per grid cell
        self.raw_weights = nn.Parameter(torch.full((height, width), init_value))

    def forward(self, occupancy_mask, position_mask=None):
        """
        Args:
            occupancy_mask: [B, H, W] — 1 where LiDAR is absent, 0 where present
            position_mask:  (unused) kept for compatibility

        Returns:
            camera_weights: [B, 1, H, W]
        """
        B, H, W = occupancy_mask.shape
        device = occupancy_mask.device

        assert H == self.height and W == self.width, \
            f"Input occupancy map size ({H}, {W}) doesn't match expected size ({self.height}, {self.width})"

        # Sigmoid weights scaled to [0.5, 1.0]
        scaled_weights = 0.5 + 0.5 * self.raw_weights.sigmoid()  # [H, W]

        # Default fusion is 0.5, add learned weight only where LiDAR is absent
        camera_weights = torch.full((B, 1, H, W), 0.5, device=device)
        camera_weights += (scaled_weights.unsqueeze(0).unsqueeze(1) - 0.5) * occupancy_mask.unsqueeze(1)

        return camera_weights'''


'''@DETECTORS.register_module()
class FusionWeightPredictor(nn.Module):
    def __init__(self, height=200, width=200, init_value=-6.0):
        super().__init__()
        self.height = height
        self.width = width

        # Learnable camera trust delta per grid cell (like before)
        self.raw_weights = nn.Parameter(torch.full((height, width), init_value))

        # Learn a nonlinear mapping from severity ∈ [0,1] to scaling factor ∈ [0,1]
        self.severity_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, corruption_mask, position_mask=None):
        """
        Args:
            corruption_mask: [B, H, W], values in [0, 1]
        Returns:
            camera_weights: [B, 1, H, W]
        """
        B, H, W = corruption_mask.shape
        device = corruption_mask.device

        assert H == self.height and W == self.width, \
            f"Input mask size ({H}, {W}) doesn't match expected size ({self.height}, {self.width})"

        # Get learnable fusion weights per cell, in [0.5, 1.0]
        scaled_weights = 0.5 + 0.5 * self.raw_weights.sigmoid()  # [H, W]
        delta = scaled_weights - 0.5  # [H, W]

        # Prepare severity for MLP: reshape to [B*H*W, 1]
        severity_input = corruption_mask.view(-1, 1)  # [B*H*W, 1]
        severity_scale = self.severity_mlp(severity_input).view(B, 1, H, W)  # [B, 1, H, W]

        # Combine base 0.5 with scaled delta based on MLP output
        camera_weights = torch.full((B, 1, H, W), 0.5, device=device)
        camera_weights += delta.unsqueeze(0).unsqueeze(1) * severity_scale

        return camera_weights'''


'''@DETECTORS.register_module()
class FusionWeightPredictor(nn.Module):
    def __init__(self, height=200, width=200, init_value=-6.0):
        super().__init__()
        self.height = height
        self.width = width

        # Learnable camera trust delta per grid cell (like before)
        self.raw_weights = nn.Parameter(torch.full((height, width), init_value))

    def forward(self, corruption_mask, position_mask=None):
        """
        Args:
            corruption_mask: [B, H, W], values in [0, 1] representing fraction of beams dropped (x / 32)
        Returns:
            camera_weights: [B, 1, H, W]
        """
        B, H, W = corruption_mask.shape
        device = corruption_mask.device

        assert H == self.height and W == self.width, \
            f"Input mask size ({H}, {W}) doesn't match expected size ({self.height}, {self.width})"

        # Get learnable fusion weights per cell, in [0.5, 1.0]
        scaled_weights = 0.5 + 0.5 * self.raw_weights.sigmoid()  # [H, W]
        delta = scaled_weights - 0.5  # [H, W]

        # Apply severity as (x/32)^2
        severity_scale = corruption_mask ** 2  # [B, H, W]
        severity_scale = severity_scale.view(B, 1, H, W)  # [B, 1, H, W]

        # Combine base 0.5 with scaled delta based on severity
        camera_weights = torch.full((B, 1, H, W), 0.5, device=device)
        camera_weights += delta.unsqueeze(0).unsqueeze(1) * severity_scale

        return camera_weights'''


'''@DETECTORS.register_module()
class FusionWeightPredictor(nn.Module):
    def __init__(self, height=200, width=200, patch_size=20, init_value=-6.0):
        super().__init__()
        self.height = height
        self.width = width
        self.patch_size = patch_size
        assert height % patch_size == 0 and width % patch_size == 0, \
            "Height and width must be divisible by patch size"

        self.num_patches_h = height // patch_size
        self.num_patches_w = width // patch_size

        # One learnable weight per patch
        self.raw_weights = nn.Parameter(torch.full(
            (self.num_patches_h, self.num_patches_w), init_value))

    def forward(self, corruption_mask, position_mask=None):
        """
        Args:
            corruption_mask: [B, H, W], values in [0, 1] indicating severity (x / 32)
        Returns:
            camera_weights: [B, 1, H, W]
        """
        B, H, W = corruption_mask.shape
        device = corruption_mask.device
        assert H == self.height and W == self.width, \
            f"Expected corruption mask of shape ({self.height}, {self.width}), got ({H}, {W})"

        # Step 1: Map per-patch raw weights to full resolution [H, W]
        scaled_weights = 0.5 + 0.5 * self.raw_weights.sigmoid()  # [num_patches_h, num_patches_w]
        full_weights = scaled_weights.repeat_interleave(self.patch_size, dim=0)\
                                      .repeat_interleave(self.patch_size, dim=1)  # [H, W]
        delta = full_weights - 0.5  # [H, W]

        # Step 2: Compute severity scale
        severity_scale = corruption_mask ** 2  # [B, H, W]
        severity_scale = severity_scale.unsqueeze(1)  # [B, 1, H, W]

        # Step 3: Initialize weights to 0.5 and add scaled delta only where corrupted
        camera_weights = torch.full((B, 1, H, W), 0.5, device=device)
        delta = delta.unsqueeze(0).unsqueeze(1)  # [1, 1, H, W]
        camera_weights += delta * severity_scale  # Only modifies where severity > 0

        return camera_weights'''

'''@DETECTORS.register_module()
class FusionWeightPredictor(nn.Module):
    def __init__(self, height=200, width=200, patch_size=8, init_value=0.0):
        """
        Initializes a learnable fusion weight map, with one value per patch.

        Args:
            height (int): Total height of BEV map.
            width (int): Total width of BEV map.
            patch_size (int): Size of each patch (e.g., 20x20).
            init_value (float): Initial value in raw (logit) space. 
                                Use 0.0 for sigmoid(0.0) = 0.5.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.patch_size = patch_size
        assert height % patch_size == 0 and width % patch_size == 0, \
            "Height and width must be divisible by patch size"

        self.num_patches_h = height // patch_size
        self.num_patches_w = width // patch_size

        # One learnable raw weight per patch (in logit space)
        self.raw_weights = nn.Parameter(torch.full(
            (self.num_patches_h, self.num_patches_w), init_value))  # sigmoid(0.0) = 0.5

    def forward(self, corruption_mask, position_mask=None):
        """
        Args:
            corruption_mask: [B, H, W] tensor (ignored in this version)
        Returns:
            camera_weights: [B, 1, H, W] tensor with values in [0, 1]
        """
        B, H, W = corruption_mask.shape
        device = corruption_mask.device
        assert H == self.height and W == self.width, \
            f"Expected corruption mask of shape ({self.height}, {self.width}), got ({H}, {W})"

        # Map patch-wise raw weights through sigmoid to [0, 1]
        patch_weights = self.raw_weights.sigmoid()  # [num_patches_h, num_patches_w]

        # Upsample to full resolution [H, W]
        full_weights = patch_weights.repeat_interleave(self.patch_size, dim=0)\
                                     .repeat_interleave(self.patch_size, dim=1)  # [H, W]

        # Expand to [B, 1, H, W] for batch-wise fusion weights
        camera_weights = full_weights.unsqueeze(0).unsqueeze(1).expand(B, 1, H, W).to(device)

        return camera_weights'''


@DETECTORS.register_module()
class FusionWeightPredictor(nn.Module):
    def __init__(self, height=200, width=200, patch_size=200, init_value=-6.0):
        super().__init__()
        self.height = height
        self.width = width
        self.patch_size = patch_size
        assert height % patch_size == 0 and width % patch_size == 0, \
            "Height and width must be divisible by patch size"

        self.num_patches_h = height // patch_size
        self.num_patches_w = width // patch_size

        # One learnable weight per patch
        self.raw_weights = nn.Parameter(torch.full(
            (self.num_patches_h, self.num_patches_w), init_value))

    def forward(self, corruption_mask, img_metas, position_mask=None):
        """
        Args:
            corruption_mask: [B, H, W], values in [0, 1] indicating severity (x / 32)
        Returns:
            camera_weights: [B, 1, H, W]
        """
        B, H, W = corruption_mask.shape
        device = corruption_mask.device
        assert H == self.height and W == self.width, \
            f"Expected corruption mask of shape ({self.height}, {self.width}), got ({H}, {W})"

        # Step 1: Map per-patch raw weights to full resolution [H, W]
        scaled_weights = 0.5 + 0.5 * self.raw_weights.sigmoid()  # [num_patches_h, num_patches_w]
        full_weights = scaled_weights.repeat_interleave(self.patch_size, dim=0)\
                                      .repeat_interleave(self.patch_size, dim=1)  # [H, W]
        delta = full_weights - 0.5  # [H, W]

        # Load saved FOV masks dictionary (camera_name -> 200x200 tensor mask)
        save_path = 'fov_masks_tensors/fov_masks.pt'
        fov_masks = torch.load(save_path, map_location=device)  # dict: cam_name -> tensor mask (bool or 0/1)

        # The camera order you used in the masks
        cam_channels = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        combined_masks = []

        for meta in img_metas:  # loop over batch
            zero_indices = meta['zero_cams']  # list/array of camera indices (in your order)

            selected_masks = [fov_masks[cam_channels[i]] for i in zero_indices]

            combined_mask = torch.zeros_like(selected_masks[0], dtype=torch.bool)
            for m in selected_masks:
                combined_mask |= m.bool()

            combined_masks.append(combined_mask.float())

        # Stack into (B, H, W) then unsqueeze to (B, 1, H, W)
        batched_masks = torch.stack(combined_masks, dim=0).unsqueeze(1)

        # Step 3: Initialize weights to 0.5 and add scaled delta only where corrupted
        camera_weights = torch.full((B, 1, H, W), 0.5, device=device)
        delta = delta.unsqueeze(0).unsqueeze(1)  # [1, 1, H, W]
        camera_weights += delta * batched_masks

        return camera_weights

