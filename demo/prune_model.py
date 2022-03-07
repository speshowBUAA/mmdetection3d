from curses import raw
from multiprocessing.spawn import old_main_modules
from pyexpat import features
from black import out
import mmcv
from mmcv.runner import force_fp32
from argparse import ArgumentParser
import torch
import numpy as np
from torch import nn as nn
from mmdet3d.models.voxel_encoders.utils import VFELayer, get_paddings_indicator
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.ops import DynamicScatter
from mmcv.parallel import collate, scatter
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox import get_box_type
from copy import deepcopy
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models.middle_encoders.pillar_scatter import PointPillarsScatter
from mmdet3d.models.backbones import SECOND
from mmdet3d.models.necks import SECONDFPN
import cv2
from mmdet.models.builder import (HEADS)
from mmdet.core import build_bbox_coder
from mmcv.runner import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer

class HardVFE(nn.Module):
    def __init__(self,
                 cfg,
                 return_point_feats=False,
                 fusion_layer=None,
                 prune_index = None,
                 test=False):
        super().__init__()
        if prune_index:
            self.feat_channels = prune_index
        else:
            self.feat_channels = cfg.model['pts_voxel_encoder']['feat_channels']
        assert len(self.feat_channels) > 0
        self.in_channels = cfg.model['pts_voxel_encoder']['in_channels']
        self._with_distance = cfg.model['pts_voxel_encoder']['with_distance']
        self._with_cluster_center = cfg.model['pts_voxel_encoder']['with_cluster_center']
        self._with_voxel_center = cfg.model['pts_voxel_encoder']['with_voxel_center']
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False
        if self._with_cluster_center:
            self.in_channels += 3
        if self._with_voxel_center:
            self.in_channels += 3
        if self._with_distance:
            self.in_channels += 1

        # Need pillar (voxel) size and x/y offset to calculate pillar offset
        self.vx = cfg['voxel_size'][0]
        self.vy = cfg['voxel_size'][1]
        self.vz = cfg['voxel_size'][2]
        self.point_cloud_range = cfg.model['pts_voxel_encoder']['point_cloud_range']
        self.x_offset = self.vx / 2 + self.point_cloud_range[0]
        self.y_offset = self.vy / 2 + self.point_cloud_range[1]
        self.z_offset = self.vz / 2 + self.point_cloud_range[2]
        self.norm_cfg = cfg.model['pts_voxel_encoder']['norm_cfg']
        self.scatter = DynamicScatter(cfg['voxel_size'], self.point_cloud_range, True)
        self.fusion_layer = None
        self.test = test

        feat_channels = [self.in_channels] + list(self.feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            print("in_filter: {:d}, out_filter: {:d}.".format(in_filters, out_filters))
            if i > 0:
                in_filters *= 2
            # TODO: pass norm_cfg to VFE
            # norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            if i == (len(feat_channels) - 2):
                cat_max = False
                max_out = True
                if fusion_layer:
                    max_out = False
            else:
                max_out = True
                cat_max = True
            vfe_layers.append(
                VFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg=self.norm_cfg,
                    max_out=max_out,
                    cat_max=cat_max))
            self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
    
    @force_fp32(out_fp16=True)
    def forward(self, *args):
        if self.training:
            return self.forward_train(*args)
        elif self.test:
            return self.forward_train(*args)
        else:
            return self.forward_test(*args)
                       
            
    @force_fp32(out_fp16=True)
    def forward_test(self,voxel_feats):
        for i, vfe in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)
        return voxel_feats

    @force_fp32(out_fp16=True)
    def forward_train(self,
                    features,
                    num_points,
                    coors,
                    img_feats=None,
                    img_metas=None):
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is MxNxC.
            num_points (torch.Tensor): Number of points in each voxel.
            coors (torch.Tensor): Coordinates of voxels, shape is Mx(1+NDim).
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = (
                features[:, :, :3].sum(dim=1, keepdim=True) /
                num_points.type_as(features).view(-1, 1, 1))
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(
                size=(features.size(0), features.size(1), 3))
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                self.y_offset)
            f_center[:, :, 2] = features[:, :, 2] - (
                coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        voxel_feats = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty.
        # Need to ensure that empty voxels remain set to zeros.
        voxel_count = voxel_feats.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        voxel_feats *= mask.unsqueeze(-1).type_as(voxel_feats)
        raw_feats = voxel_feats
        for i, vfe in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)
        
        if (self.fusion_layer is not None and img_feats is not None):
            voxel_feats = self.fusion_with_mask(features, mask, voxel_feats,
                                                coors, img_feats, img_metas)
        return voxel_feats, raw_feats

    def fusion_with_mask(self, features, mask, voxel_feats, coors, img_feats,
                         img_metas):
        """Fuse image and point features with mask.

        Args:
            features (torch.Tensor): Features of voxel, usually it is the
                values of points in voxels.
            mask (torch.Tensor): Mask indicates valid features in each voxel.
            voxel_feats (torch.Tensor): Features of voxels.
            coors (torch.Tensor): Coordinates of each single voxel.
            img_feats (list[torch.Tensor]): Multi-scale feature maps of image.
            img_metas (list(dict)): Meta information of image and points.

        Returns:
            torch.Tensor: Fused features of each voxel.
        """
        # the features is consist of a batch of points
        batch_size = coors[-1, 0] + 1
        points = []
        for i in range(batch_size):
            single_mask = (coors[:, 0] == i)
            points.append(features[single_mask][mask[single_mask]])

        point_feats = voxel_feats[mask]
        point_feats = self.fusion_layer(img_feats, points, point_feats,
                                        img_metas)

        voxel_canvas = voxel_feats.new_zeros(
            size=(voxel_feats.size(0), voxel_feats.size(1),
                  point_feats.size(-1)))
        voxel_canvas[mask] = point_feats
        out = torch.max(voxel_canvas, dim=1)[0]

        return out

class Backbone(nn.Module):
    def __init__(self, cfg, pts_backbone, pts_neck, pts_bbox_head):
        super().__init__()
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck
        self.cfg = cfg
        self.pts_bbox_head = pts_bbox_head
        self.bbox_coder = build_bbox_coder(cfg['pts_bbox_head']['bbox_coder'])
        self.box_code_size = self.bbox_coder.code_size
        loss_cls = cfg['pts_bbox_head']['loss_cls']
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.dir_limit_offset = cfg['pts_bbox_head']['dir_limit_offset']
        self.dir_offset = cfg['pts_bbox_head']['dir_offset']

    def forward(self, input):
        print(input.shape)
        x = input[:22*200*200]
        x = x.reshape(-1, 22, 200, 200)
        anchors = input[22*200*200:]
        anchors = anchors.reshape(-1, 9)

        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        x = self.pts_bbox_head(x)
        bboxes, scores, labels = self.get_boxes(*x, anchors)
        return bboxes, scores, labels
    
    def decode(self, anchors, deltas):
        """Apply transformation `deltas` (dx, dy, dz, dw, dh, dl, dr, dv*) to
        `boxes`.

        Args:
            anchors (torch.Tensor): Parameters of anchors with shape (N, 7).
            deltas (torch.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, w, l, h, r, velo*].

        Returns:
            torch.Tensor: Decoded boxes.
        """
        cas, cts = [], []
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)
        za = za + ha / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)

    def get_boxes(self, cls_scores, bbox_preds, dir_cls_preds, anchors):
        cls_score = cls_scores[0].detach()
        bbox_pred = bbox_preds[0].detach()
        dir_cls_pred = dir_cls_preds[0].detach()
        cls_score = cls_scores[0].detach()
        bbox_pred = bbox_preds[0].detach()
        dir_cls_pred = dir_cls_preds[0].detach()
        bboxes, scores, labels = self.get_bboxes_single(cls_score, bbox_pred,
                                            dir_cls_pred, anchors)
        return bboxes, scores, labels

    def get_bboxes_single(self, cls_scores, bbox_preds, dir_cls_preds, anchors):
        cls_score = cls_scores[0]
        bbox_pred = bbox_preds[0]
        dir_cls_pred = dir_cls_preds[0]
        dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.pts_bbox_head.num_classes)
        if self.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            scores = cls_score.softmax(-1)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.box_code_size)

        nms_pre = 1000
        if self.use_sigmoid_cls:
            max_scores, _ = scores.max(dim=1)
        else:
            max_scores, _ = scores[:, :-1].max(dim=1)
        _, topk_inds = max_scores.topk(nms_pre)
        topk_inds = topk_inds.long()
        anchors = anchors[topk_inds, :]
        bbox_pred = bbox_pred[topk_inds, :]
        scores = scores[topk_inds, :]
        dir_cls_score = dir_cls_score[topk_inds]
        bboxes = self.decode(anchors, bbox_pred)
        return bboxes, scores, dir_cls_score

class PRUNED_SECOND(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 init_cfg=None,
                 pretrained=None):
        super().__init__()
        self.layer_nums = layer_nums

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        count = 0
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[count],
                    out_channels[count],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                build_norm_layer(norm_cfg, out_channels[count])[1],
                nn.ReLU(inplace=True),
            ]
            count += 1
            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[count-1],
                        out_channels[count],
                        3,
                        padding=1))
                block.append(build_norm_layer(norm_cfg, out_channels[count])[1])
                block.append(nn.ReLU(inplace=True))
                count += 1

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)

def parse_model(model):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

def visualize_feature(feature, i=0, file = 'visualize.jpg'):
    if isinstance(feature, torch.Tensor):
        feature = feature.cpu().detach().numpy()
    feature = feature.squeeze(0)
    img = feature[i,:,:] * 255
    cv2.imwrite(file, img)

def get_thre(model, module = nn.BatchNorm2d, ratio = 0.7):
    total = 0
    for m in model.modules():
        if isinstance(m, module):
            total += m.weight.data.shape[0]
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, module):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size
    y, i = torch.sort(bn)
    thre_index = int(total * ratio)
    thre = y[thre_index]
    return thre

def prune(model, thre, module = nn.BatchNorm2d):
    pruned = 0
    cfg_index = []
    cfg_mask = []
    index = 0
    for m in model.modules():
        if isinstance(m, module):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.cpu().ge(thre).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg_index.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(index, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg_index.append('M')
        index += 1
    return cfg_index, cfg_mask

def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])

def build_and_prune_vfe_model(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    convert_SyncBN(config.model)
    config.model.train_cfg = None
    # original model
    model = build_model(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint_load = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint_load['meta']:
            model.CLASSES = checkpoint_load['meta']['CLASSES']
        else:
            model.CLASSES = config.class_names
        if 'PALETTE' in checkpoint_load['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint_load['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    torch.cuda.set_device(device)
    model.to(device)
    model.eval()

    cfg_index, cfg_mask = prune(model, get_thre(model, module=nn.BatchNorm1d, ratio=0.7), nn.BatchNorm1d)
    # parse_model(model.pts_voxel_encoder)

    # VFE model
    pts_voxel_encoder = HardVFE(config, prune_index=cfg_index)
    pts_voxel_encoder.to(device).eval()
    ori_modules = list(model.pts_voxel_encoder.modules())
    prune_modules = list(pts_voxel_encoder.modules())
    layer_id_in_cfg = 0

    start_mask = torch.ones(pts_voxel_encoder.in_channels)
    end_mask = cfg_mask[layer_id_in_cfg]
    linear_count = 0

    for layer_id in range(len(prune_modules)):
        m0 = ori_modules[layer_id]
        m1 = prune_modules[layer_id]
        if isinstance(m0, nn.BatchNorm1d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
        elif isinstance(m0, nn.Linear):
            if isinstance(ori_modules[layer_id-1], nn.BatchNorm1d):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                if linear_count == 1:   ## VFElayer 的repeat操作
                    repeated = np.array(idx0 + int(m0.weight.data.size()[1] / 2))
                    idx0 = np.concatenate((idx0, repeated))
                w1 = m0.weight.data[:, idx0.tolist()].clone()
                w1 = w1[idx1.tolist(),:].clone()
                m1.weight.data = w1.clone()
                linear_count += 1
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]

    # print('-----------------------')
    # parse_model(pts_voxel_encoder)
    return model, pts_voxel_encoder, cfg_mask

def build_backbone_model(model, in_channels, cfg_mask, device='cuda:0'):
    pts_backbone = SECOND(in_channels, out_channels=[64, 128, 256])
    pts_backbone.to(device).eval()
    ori_modules = list(model.pts_backbone.modules())
    prune_modules = list(pts_backbone.modules())
    flag = False
    for layer_id in range(len(prune_modules)):
        m0 = ori_modules[layer_id]
        m1 = prune_modules[layer_id]
        if not flag :
            if isinstance(m0, nn.Conv2d):       #only the first conv layer is different
                idx = np.squeeze(np.argwhere(np.asarray(cfg_mask.cpu().numpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w1 = m0.weight.data[:,idx.tolist(),:,:].clone()
                m1.weight.data = w1.clone()
                flag = True
                continue
        
        if isinstance(m0, nn.Conv2d):
            m1.weight.data = m0.weight.data.clone()
        if isinstance(m0, nn.BatchNorm2d):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

    return pts_backbone

def prune_backbone_model(model, in_channel, ratio=0.1, device='cuda:0'):
    cfg_index, cfg_mask = prune(model, get_thre(model, ratio=ratio))    # bn2d
    pruned_backbone = PRUNED_SECOND(in_channel, cfg_index)
    pruned_backbone.to(device).eval()
    # parse_model(pruned_backbone)
    layer_id_in_cfg = 0

    output_layer_mask = []
    id = -1
    for layer_num in pruned_backbone.layer_nums:
        id += 1 + layer_num
        output_layer_mask.append(cfg_mask[id])
        
    start_mask = torch.ones(in_channel)
    end_mask = cfg_mask[layer_id_in_cfg]

    ori_modules = list(model.modules())
    prune_modules = list(pruned_backbone.modules())
    for layer_id in range(len(ori_modules)):
        m0 = ori_modules[layer_id]
        m1 = prune_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('shape {:d}.'.format(idx1.size))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if isinstance(ori_modules[layer_id+1], nn.BatchNorm2d):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
    return pruned_backbone, output_layer_mask

def build_neck_model(model, in_channels, cfg_mask, device='cuda:0'):
    pts_neck = SECONDFPN(in_channels, out_channels=[128, 128, 128])
    pts_neck.to(device).eval()
    ori_modules = list(model.pts_neck.modules())
    prune_modules = list(pts_neck.modules())
    cfg_mask_id = 0
    for layer_id in range(len(prune_modules)):
        m0 = ori_modules[layer_id]
        m1 = prune_modules[layer_id]
        if isinstance(m0, nn.ConvTranspose2d):       # 3 channels , the first convtranspose layer in every channel is different
            idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[cfg_mask_id].cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w1 = m0.weight.data[idx.tolist(),:,:,:].clone()
            m1.weight.data = w1.clone()
            cfg_mask_id += 1
        if isinstance(m0, nn.BatchNorm2d):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
    return pts_neck

def preprocess(model, pcd):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    data = dict(
        pts_filename=pcd,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # for ScanNet demo we need axis_align_matrix
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[],
        # set timestamp = 0
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
    return data

# get api
def get_prune_model(cfg):
    device = "cuda:0"
    checkpoint = "checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth"
    # pcd = "/data/testdata/QT_js_for_nuscenes_10Hz/1606813512902872600.bin"
    # checkpoint = "work_dirs/hv_pointpillars_secfpn_sbn-all_4x8_2x_custom-3d_0302/epoch_24.pth"
    pcd = "data/2022-02-18/testing/velodyne/1645148902202493613.bin"
    model, prune_pts_vfe, cfg_mask = build_and_prune_vfe_model(cfg, checkpoint, device=device)

    data = preprocess(model, pcd)
    pts = data['points'][0]
    voxels, num_points, coors = model.voxelize(pts)
    _, raw_feats = model.pts_voxel_encoder(voxels, num_points, coors)
    v_features = prune_pts_vfe(raw_feats)
    batch_size = coors[-1, 0] + 1
    pts_middle_encoder = PointPillarsScatter(v_features.shape[1], output_shape=[400, 400])
    feature_map = pts_middle_encoder(v_features, coors, batch_size)
    pts_backbone = build_backbone_model(model, v_features.shape[1], cfg_mask[-1], device=device)
    prune_pts_backbone, backbone_cfg_mask = prune_backbone_model(pts_backbone, v_features.shape[1], ratio=0.1, device=device)
    b_features = prune_pts_backbone(feature_map)
    pts_neck = build_neck_model(model, [b_features[0].shape[1], b_features[1].shape[1], b_features[2].shape[1]], backbone_cfg_mask, device=device)
    return prune_pts_vfe, pts_middle_encoder, prune_pts_backbone, pts_neck

def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model, prune_model, cfg_mask = build_and_prune_vfe_model(args.config, args.checkpoint, device=args.device)

    data = preprocess(model, args.pcd)
    img_metas = data['img_metas'][0]
    pts = data['points'][0]
    np_pts = pts[0].cpu().numpy()
    np_pts.tofile('pts.bin')
    voxels, num_points, coors = model.voxelize(pts)
    voxel_features, raw_feats = model.pts_voxel_encoder(voxels, num_points, coors)

    batch_size = coors[-1, 0] + 1
    feature_map = model.pts_middle_encoder(voxel_features, coors, batch_size)

    r_np_raw_feats = np.loadtxt('0_Model_pfe_input_gather_feature.txt',  dtype=np.float32).reshape(-1, 64, 10)
    r_np_raw_feats = r_np_raw_feats[:3687, :, :]
    
    # np_raw_feats = raw_feats.cpu().numpy()
    # sort_ind = []
    # for i in range(r_np_raw_feats.shape[0]):
    #     sort_ind.append([r_np_raw_feats[i,0,0], r_np_raw_feats[i,0,1]])
    # sort_ind = np.array(sort_ind)
    # modify_r_np_raw_feats = np.empty_like(r_np_raw_feats)
    # np_coors = coors.cpu().numpy()
    # check_coors = np.zeros_like(np_coors)
    # for i in range(np_raw_feats.shape[0]):
    #     value = np.array([np_raw_feats[i,0,0], np_raw_feats[i,0,1]])
    #     idx = (np.sqrt((sort_ind[:,0] - value[0])*(sort_ind[:,0] - value[0]) + (sort_ind[:,1] - value[1])*(sort_ind[:,1] - value[1]))).argmin()
    #     modify_r_np_raw_feats[i,:,:] = r_np_raw_feats[idx,:,:]
    #     check_coors[i,2] = int((value[1] + 25) * 4)
    #     check_coors[i,3] = int((value[0] + 25) * 4)

    # coors_range = np.array([-25, -25, -5, 25, 25, 3])
    # voxel_size = np.array([0.25, 0.25, 8])
    # voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    # voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    # voxelmap_shape = voxelmap_shape[::-1]
    # print(voxelmap_shape)
    # coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    # coor = np.zeros(shape=(3, ), dtype=np.int32)
    # ndim = 3
    # ndim_minus_1 = ndim - 1
    # voxel_num = 0
    # for i in range(np_pts.shape[0]):
    #     for j in range(ndim):
    #         c = np.floor((np_pts[i, j] - coors_range[j]) / voxel_size[j])
    #         coor[ndim_minus_1 - j] = c
    #     voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
    #     if voxelidx == -1:
    #         voxelidx = voxel_num
    #         voxel_num += 1
    #         coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx

    # swap_r_np_raw_feats = np.empty_like(r_np_raw_feats)
    # for i in range(r_np_raw_feats.shape[0]):
    #     for j in range(ndim):
    #         c = np.floor((r_np_raw_feats[i,0,j] - coors_range[j]) / voxel_size[j]).astype(np.int32)
    #         coor[ndim_minus_1 - j] = c
    #     voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
    #     swap_r_np_raw_feats[voxelidx,:,:] = r_np_raw_feats[i,:,:]

    r_raw_feats = torch.from_numpy(r_np_raw_feats).cuda(args.device)

    # vfe prune
    v_features = prune_model(r_raw_feats)
    # # vfe onnx export
    # if isinstance(args.config, str):
    #     config = mmcv.Config.fromfile(args.config)
    # dummy_input = torch.ones(config.model['pts_voxel_layer']['max_voxels'][1], config.model['pts_voxel_layer']['max_num_points'] , prune_model.in_channels).cuda()
    # export_onnx_file = './pts_pfe_prune.onnx'
    # torch.onnx.export(prune_model,
    #                 dummy_input,
    #                 export_onnx_file,
    #                 opset_version=12,
    #                 verbose=True,
    #                 do_constant_folding=True) # 输出名

    r_np_v_features = np.loadtxt('1_Model_pfe_output_buffers_[1].txt',  dtype=np.float32).reshape(-1, 22)
    r_np_v_features = r_np_v_features[:3687, :]
    r_v_features = torch.from_numpy(r_np_v_features).cuda(args.device)
    v_features = r_v_features

    pts_middle_encoder = PointPillarsScatter(v_features.shape[1], output_shape=[200, 200])
    rfeature_map = pts_middle_encoder(v_features, coors, batch_size)
    print(rfeature_map.shape)
    pts_backbone = build_backbone_model(model, v_features.shape[1], cfg_mask[-1], device=args.device)

    r_np_feature_map = np.loadtxt('2_Model_backbone_input_dev_scattered_feature.txt',  dtype=np.float32).reshape(-1, 200, 200)
    r_feature_map = torch.from_numpy(r_np_feature_map).cuda(args.device).unsqueeze(0)
    print(r_feature_map.shape)
    # debug 
    x = model.pts_backbone(feature_map)
    rx = pts_backbone(r_feature_map)
    x = model.pts_neck(rx)
    bbox_list = [dict() for i in range(len(img_metas))]
    outs = model.pts_bbox_head(x)
    bbox_list = model.pts_bbox_head.get_bboxes(
        *outs, img_metas, rescale=False)
    bbox_results = [
        bbox3d2result(bboxes, scores, labels)
        for bboxes, scores, labels in bbox_list
    ]
    boxes = bbox_results[0]['boxes_3d'].tensor.numpy()
    np.savetxt('./boxes.txt', boxes[:,:7], fmt="%f", delimiter=" ")
    exit()
    # # backbone onnx export
    # if isinstance(args.config, str):
    #     config = mmcv.Config.fromfile(args.config)
    # pts_backbone = Backbone(config.model, pts_backbone, model.pts_neck, model.pts_bbox_head)
    # pts_backbone.to(args.device).eval()
    # anchors = np.load('./np_anchors.npy')
    # anchors = torch.from_numpy(anchors).cuda(args.device)
    # rfeature_map = rfeature_map.reshape(-1)
    # ranchors = anchors.reshape(-1)
    # input = torch.cat((rfeature_map, ranchors))
    # # export to onnx
    # export_onnx_file = './pts_backbone_prune.onnx'
    # torch.onnx.export(pts_backbone,
    #                 input,
    #                 export_onnx_file,
    #                 opset_version=12,
    #                 verbose=True,
    #                 do_constant_folding=True) # 输出名

    prune_pts_backbone, backbone_cfg_mask = prune_backbone_model(pts_backbone, v_features.shape[1], ratio=0.1, device=args.device)
    v_features = prune_pts_backbone(rfeature_map)
    pts_neck = build_neck_model(model, [v_features[0].shape[1], v_features[1].shape[1], v_features[2].shape[1]], backbone_cfg_mask, device=args.device)
    
    # debug
    x = model.pts_backbone(feature_map)
    x = model.pts_neck(x)
    rx = pts_neck(v_features)
    bbox_list = [dict() for i in range(len(img_metas))]
    outs = model.pts_bbox_head(rx)
    bbox_list = model.pts_bbox_head.get_bboxes(
        *outs, img_metas, rescale=False)
    bbox_results = [
        bbox3d2result(bboxes, scores, labels)
        for bboxes, scores, labels in bbox_list
    ]
    boxes = bbox_results[0]['boxes_3d'].tensor.numpy()
    np.savetxt('./boxes.txt', boxes[:,:7], fmt="%f", delimiter=" ")

    # backbone onnx export
    if isinstance(args.config, str):
        config = mmcv.Config.fromfile(args.config)
    pts_backbone_output = Backbone(config.model, prune_pts_backbone, pts_neck, model.pts_bbox_head)
    pts_backbone_output.to(args.device).eval()
    anchors = np.load('./np_anchors.npy')
    anchors = torch.from_numpy(anchors).cuda(args.device)
    rfeature_map = rfeature_map.reshape(-1)
    np_rfeature_map = rfeature_map.detach().cpu().numpy()
    np.save('./np_rfeature_map.npy', np_rfeature_map)
    ranchors = anchors.reshape(-1)
    input = torch.cat((rfeature_map, ranchors))

    # export to onnx
    export_onnx_file = './pts_backbone_prune.onnx'
    torch.onnx.export(pts_backbone_output,
                    input,
                    export_onnx_file,
                    opset_version=12,
                    verbose=True,
                    do_constant_folding=True) # 输出名

import onnxruntime
def test_onnx():
    # test onnx
    input_data = np.loadtxt('2_Model_backbone_input_dev_scattered_feature.txt',  dtype=np.float32).reshape(-1, 22, 200, 200)
    # input_data = np.load('./rfeature_map.npy')
    anchors = np.load('./np_anchors.npy')
    # np.savetxt('generate_anchors.txt', anchors, fmt="%f", delimiter=" ")
    # anchors = np.zeros(anchors.shape, dtype=anchors.dtype)

    input_data = input_data.reshape(-1)
    print(input_data.shape)
    anchors = anchors.reshape(-1)
    print(anchors.shape)
    input_data = np.concatenate([input_data, anchors])
    print(input_data.shape)

    sess = onnxruntime.InferenceSession('./pts_backbone_prune.onnx')
    outputs = sess.run(["271", "222", "223"], {"0":input_data})
    # outputs = sess.run(["134", "193", "194", "195"], {"0":input_data})
    np_preds = outputs[0]
    np_scores = outputs[1]
    np_clss = outputs[2]
    # np_anchor = outputs[0]

    print(np_preds[:1])
    print(np_scores[:1])
    print(np_clss[:1])
    # print(np_anchor[:1])
    print(np_preds.shape)
    print(np_scores.shape)
    print(np_clss.shape)
    # print(np_anchor.shape)

    filter_pred = []
    for i in range(np_scores.shape[0]):
        score = np_scores[i,:][0]
        if score > 0.1:
            filter_pred.append(np_preds[i,:])
    filter_pred = np.array(filter_pred)
    print(filter_pred.shape)
    np.savetxt('./boxes_onnx.txt', filter_pred[:,:7], fmt="%f", delimiter=" ")

if __name__ == '__main__':
    # test_onnx()
    main()