from argparse import ArgumentParser
from curses import raw

import torch
from torch import nn
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
import numpy as np
import onnx
import onnxruntime
from copy import deepcopy
from mmcv.parallel import collate, scatter
import mmcv
from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16,force_fp32
from mmcv.runner import load_checkpoint
from mmdet3d.core import anchor, bbox3d2result
from mmdet3d.core.bbox import get_box_type
from mmdet3d.apis import init_model
from simplifier_onnx import simplify_onnx as simplify_onnx
from torch.nn import functional as F
import onnx
from onnxsim import simplify
from mmdet.core import build_bbox_coder, build_prior_generator

class VFELayer(nn.Module):
    """Voxel Feature Encoder layer.

    The voxel encoder is composed of a series of these layers.
    This module do not support average pooling and only support to use
    max pooling to gather features inside a VFE.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        max_out (bool): Whether aggregate the features of points inside
            each voxel and only return voxel features.
        cat_max (bool): Whether concatenate the aggregated features
            and pointwise features.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 max_out=True,
                 cat_max=True):
        super(VFELayer, self).__init__()
        self.fp16_enabled = False
        self.cat_max = cat_max
        self.max_out = max_out
        # self.units = int(out_channels / 2)

        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    @auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (torch.Tensor): Voxels features of shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.

        Returns:
            torch.Tensor: Voxel features. There are three mode under which the
                features have different meaning.
                - `max_out=False`: Return point-wise features in
                    shape (N, M, C).
                - `max_out=True` and `cat_max=False`: Return aggregated
                    voxel features in shape (N, C)
                - `max_out=True` and `cat_max=True`: Return concatenated
                    point-wise features in shape (N, M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]
        if self.max_out:
            aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        else:
            # this is for fusion layer
            return pointwise

        if not self.cat_max:
            return aggregated.squeeze(1)
        else:
            # [K, 1, units]
            repeated = aggregated.repeat(1, voxel_count, 1)
            concatenated = torch.cat([pointwise, repeated], dim=2)
            # [K, T, 2 * units]
            return concatenated

class VFE(nn.Module):
    def __init__(self,
                 cfg,
                 return_point_feats=False,
                 fusion_layer=None,
                 ):
        super().__init__()
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

        feat_channels = [self.in_channels] + list(self.feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
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
    def forward(self,voxel_feats):
        for i, vfe in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)
        return voxel_feats

class PointPillar(nn.Module):
    def __init__(self,
                 cfg,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_metas=None):
        super().__init__()
        self.cfg = cfg
        self.pts_voxel_encoder = pts_voxel_encoder
        self.pts_middle_encoder = pts_middle_encoder
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck
        self.img_metas = img_metas
        self.pts_bbox_head = pts_bbox_head
        self.bbox_coder = build_bbox_coder(cfg['pts_bbox_head']['bbox_coder'])
        self.anchor_generator = build_prior_generator(cfg['pts_bbox_head']['anchor_generator'])
        self.box_code_size = self.bbox_coder.code_size
        
    def forward(self, voxel_feats, coors):
        voxel_features = self.pts_voxel_encoder(voxel_feats)
        x = self.pts_middle_encoder(voxel_features, coors, 1)
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        x = self.pts_bbox_head(x)
        bboxes, scores, labels = self.get_boxes(*x)
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

    def get_boxes(self, cls_scores, bbox_preds, dir_cls_preds):
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        cls_score = cls_scores[0].detach()
        bbox_pred = bbox_preds[0].detach()
        dir_cls_pred = dir_cls_preds[0].detach()
        anchors = mlvl_anchors[0].reshape(-1, self.box_code_size)
        cls_score = cls_scores[0].detach()
        bbox_pred = bbox_preds[0].detach()
        dir_cls_pred = dir_cls_preds[0].detach()
        bboxes, scores, labels = self.get_bboxes_single(cls_score, bbox_pred,
                                            dir_cls_pred, anchors)
        return bboxes, scores, labels

    def get_bboxes_single(self, cls_scores, bbox_preds, dir_cls_preds, anchors):
        cfg = self.cfg['test_cfg']['pts']
        # assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        cls_score = cls_scores[0]
        bbox_pred = bbox_preds[0]
        dir_cls_pred = dir_cls_preds[0]
       
        dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        # dir_cls_score = dir_cls_pred
        dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.pts_bbox_head.num_classes)
        scores = cls_score.sigmoid()
        bbox_pred = bbox_pred.permute(1, 2,
                                        0).reshape(-1, self.box_code_size)
        # return bbox_pred, scores, dir_cls_score

        nms_pre = 1000
        max_scores, _ = scores.max(dim=1)
        _, topk_inds = max_scores.topk(nms_pre)
        topk_inds = topk_inds.long()
        anchors = anchors[topk_inds, :]
        bbox_pred = bbox_pred[topk_inds, :]
        scores = scores[topk_inds, :]
        dir_cls_score = dir_cls_score[topk_inds]
        bboxes = self.decode(anchors, bbox_pred)
        # bboxes = self.onnx_decode(bbox_pred)
        return bboxes, scores, dir_cls_score

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

def build_vfe_model(config, checkpoint=None, device='cuda:0'):
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

    # VFE model
    pts_voxel_encoder = VFE(config)
    pts_voxel_encoder.to(device).eval()
    checkpoint_pts_load = torch.load(checkpoint, map_location=device)
    dicts = {}
    for key in checkpoint_pts_load["state_dict"].keys():
        if "vfe" in key:
            dicts[key[18:]] = checkpoint_pts_load["state_dict"][key]
    pts_voxel_encoder.load_state_dict(dicts)
    print('-----------------------')
    return model, pts_voxel_encoder

def parse_model(model):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

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
        data['points'] = data['points'][0].data
    return data

def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--max-voxels', default=5000, help='max-voxels')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model, pts_voxel_encoder = build_vfe_model(args.config, args.checkpoint, device=args.device)
    # parse_model(model)
    data = preprocess(model, args.pcd)
    img_metas = data['img_metas'][0]
    pts = data['points'][0]

    if isinstance(args.config, str):
        config = mmcv.Config.fromfile(args.config)
    elif not isinstance(args.config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(args.config)}')

    PointPillarModel = PointPillar(config.model, pts_voxel_encoder, model.pts_middle_encoder, model.pts_backbone, model.pts_neck, model.pts_bbox_head, img_metas)
    PointPillarModel.to('cuda').eval()
    checkpoint = torch.load(args.checkpoint, map_location='cuda')
    dicts = {}
    for key in checkpoint["state_dict"].keys():
        if "vfe" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "backbone" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "neck" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "bbox_head" in key:
            dicts[key] = checkpoint["state_dict"][key]
    PointPillarModel.load_state_dict(dicts)

    torch.cuda.set_device(args.device)
    PointPillarModel.to(args.device)
    PointPillarModel.eval()

    # anchors = np.load('./np_anchors_400x400.npy')
    # anchors = torch.from_numpy(anchors).cuda(args.device)

    bbox_list = [dict() for i in range(len(img_metas))]
    voxels, num_points, coors = model.voxelize(pts)
    voxel_features, raw_feats = model.pts_voxel_encoder(voxels, num_points, coors)
    raw_feats = raw_feats[:args.max_voxels,:,:]
    coors = coors[:args.max_voxels,:]
    bboxes, scores, labels = PointPillarModel(raw_feats, coors)

    boxes = bboxes.cpu().numpy()
    scores = scores.cpu().numpy()
    np.savetxt('./boxes.txt', boxes[:,:7], fmt="%f", delimiter=" ")
    np.savetxt('./scores.txt', scores, fmt="%f", delimiter=" ")

    export_onnx_file = './pointpillar.onnx'
    torch.onnx.export(PointPillarModel,
                    (raw_feats, coors),
                    export_onnx_file,
                    export_params=True,
                    opset_version=11,
                    verbose=True,
                    do_constant_folding=True,
                    input_names = ['input', 'coords'],   # the model's input names
                    output_names = ['box_preds', 'cls_preds', 'dir_cls_preds']) # the model's output names
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(export_onnx_file)), export_onnx_file)
    
    # onnx_model = onnx.load("./pointpillar.onnx")  # load onnx model
    # model_simp, check = simplify(onnx_model)
    # assert check, "Simplified ONNX model could not be validated"
    
    # model_simp = simplify_onnx(onnx_model)
    # onnx.save(model_simp, "pointpillar_simp.onnx")
    # print("export pointpillar.onnx.")
    # print('finished exporting onnx')

    # # debug onnx
    # sess = onnxruntime.InferenceSession('./pointpillar_sim.onnx')
    # outputs = sess.run(['cls_preds', 'box_preds', 'dir_cls_preds'], {"input":raw_feats, "coords":coors, "anchors":anchors})[0]

    # np_preds = outputs[0]
    # np_scores = outputs[1]

    # filter_pred = []
    # for i in range(np_scores.shape[0]):
    #     score = np_scores[i,:][0]
    #     if score > 0.4:
    #         filter_pred.append(np_preds[i,:])
    # filter_pred = np.array(filter_pred)
    # print(filter_pred.shape)
    # np.savetxt('./boxes_onnx.txt', filter_pred[:,:7], fmt="%f", delimiter=" ")

if __name__ == '__main__':
    main()