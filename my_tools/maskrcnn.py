from __future__ import division

from functools import wraps
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg

from modeling.ResNet import ResNet50_conv4_body, ResNet_roi_conv5_head
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction

def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper

class MaskRCNN(nn.Module):
    def __init__(self):
        super(MaskRCNN, self).__init__()

        # For cache
        self.mapping_to_detectron = None
        # self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = ResNet50_conv4_body()

        print("RPN ON")
        self.RPN = rpn_heads.generic_rpn_outputs(self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        print("BBOX ON")
        self.Box_Head = ResNet_roi_conv5_head(self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
        self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(self.Box_Head.dim_out)

        print("MASK ON")
        self.Mask_Head = mask_rcnn_heads.mask_rcnn_fcn_head_v0upshare(
            self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
        self.Mask_Head.share_res5_module(self.Box_Head.res5)
        self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(self.Mask_Head.dim_out)

        self._init_modules()

    def _init_modules(self):
        from utils.resnet_weights_helper import load_pretrained_imagenet_weights

        # if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
        #     print("LOAD IMAGENET WEIGHTS")
        #     load_pretrained_imagenet_weights(self)
        #     # Check if shared weights are equaled
        #     if cfg.MODEL.MASK_ON and getattr(self.Mask_Head, 'SHARE_RES5', False):
        #         assert compare_state_dict(self.Mask_Head.res5.state_dict(), self.Box_Head.res5.state_dict())
        #     # if cfg.MODEL.KEYPOINTS_ON and getattr(self.Keypoint_Head, 'SHARE_RES5', False):
        #     #     assert compare_state_dict(self.Keypoint_Head.res5.state_dict(), self.Box_Head.res5.state_dict())
        
        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, im_info, roidb=None, **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info, roidb, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info, roidb, **rpn_kwargs)

    def _forward(self, data, im_info, roidb=None, **rpn_kwargs):
        im_data = data
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables

        blob_conv = self.Conv_Body(im_data)

        rpn_ret = self.RPN(blob_conv, im_info, roidb)

        # if not cfg.MODEL.RPN_ONLY:
        if cfg.MODEL.SHARE_RES5 and self.training:
            box_feat, res5_feat = self.Box_Head(blob_conv, rpn_ret)
        else:
            box_feat = self.Box_Head(blob_conv, rpn_ret)
        cls_score, bbox_pred = self.Box_Outs(box_feat)

        if self.training:
            raise NotImplentedError("Forward pass training code not implemented!")
        else:
            return_dict['blob_conv'] = blob_conv
            return_dict['rois'] = rpn_ret['rois']
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred

        return return_dict

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIAlign',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    xform_out = RoIAlignFunction(
                        resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            xform_out = RoIAlignFunction(
                resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv

    @check_inference
    def mask_net(self, blob_conv, rpn_blob):
        """For inference"""
        mask_feat = self.Mask_Head(blob_conv, rpn_blob)
        mask_pred = self.Mask_Outs(mask_feat)
        return mask_pred

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

            return self.mapping_to_detectron, self.orphans_in_detectron

