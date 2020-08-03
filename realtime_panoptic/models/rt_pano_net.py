# Copyright 2020 Toyota Research Institute.  All rights reserved.

# We adapted some FCOS related functions from official repository.
# https://github.com/tianzhi0549/FCOS

import math
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import apex
from realtime_panoptic.layers.scale import Scale
from realtime_panoptic.utils.bounding_box import BoxList
from realtime_panoptic.models.backbones import ResNetWithModifiedFPN
from realtime_panoptic.models.panoptic_from_dense_box import PanopticFromDenseBox
class RTPanoNet(torch.nn.Module):
    """Real-Time Panoptic Network
    This module takes the input from a FPN backbone and conducts feature extraction
    through a panoptic head, which can be then fed into post processing for final panoptic
    results including semantic segmentation and instance segmentation.
    NOTE: Currently only the inference functionality is supported.

    Parameters
    ----------
    backbone: str
        backbone type.

    num_classes: int
        Number of total classes, including 'things' and 'stuff'.

    things_num_classes: int
        number of thing classes

    pre_nms_thresh: float
        Acceptance class probability threshold for bounding box candidates before NMS.

    pre_nms_top_n: int
        Maximum number of accepted bounding box candidates before NMS.

    nms_thresh: float
        NMS threshold.

    fpn_post_nms_top_n: int
        Maximum number of detected object per image.

    instance_id_range: list of int
        [min_id, max_id] defines the range of id in 1:num_classes that corresponding to thing classes.
    """

    def __init__(
        self,
        backbone, 
        num_classes,
        things_num_classes,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        instance_id_range
        ):
        super(RTPanoNet, self).__init__()
        # TODO: adapt more backbone.
        if backbone == 'R-50-FPN-RETINANET':
            self.backbone = ResNetWithModifiedFPN('resnet50')
            backbone_out_channels = 256
            fpn_strides = [8, 16, 32, 64, 128]
            num_fpn_levels = 5
        else:
            raise NotImplementedError("Backbone type: {} is not supported yet.".format(backbone))
        # Global panoptic head that extracts features from each FPN output feature map
        self.panoptic_head = PanopticHead(
            num_classes,
            things_num_classes,
            num_fpn_levels,
            fpn_strides,
            backbone_out_channels
            )

        # Parameters
        self.fpn_strides = fpn_strides

        # Use dense bounding boxes to reconstruct panoptic segmentation results.
        self.panoptic_from_dense_bounding_box = PanopticFromDenseBox(
            pre_nms_thresh=pre_nms_thresh,
            pre_nms_top_n=pre_nms_top_n,
            nms_thresh=nms_thresh,
            fpn_post_nms_top_n=fpn_post_nms_top_n,
            min_size=0,
            num_classes=num_classes,
            mask_thresh=0.4,
            instance_id_range=instance_id_range,
            is_training=False)

    def forward(self, images, detection_targets=None, segmentation_targets=None):
        """ Forward function.

        Parameters
        ----------
        images: torchvision.models.detection.ImageList
            Images for which we want to compute the predictions

        detection_targets: list of BoxList
            Ground-truth boxes present in the image

        segmentation_targets: List of torch.Tensor
            semantic segmentation target for each image in the batch.

        Returns
        -------
        panoptic_result: Dict
            'instance_segmentation_result': list of BoxList
                The predicted boxes (including instance masks), one BoxList per image.
            'semantic_segmentation_result': torch.Tensor
                semantic logits interpolated to input data size. 
                NOTE: this might not be the original input image size due to paddings. 
        losses: dict of torch.ScalarTensor
            the losses for the model during training. During testing, it is an empty dict.
        """
        features = self.backbone(torch.stack(images.tensors))

        locations = self.compute_locations(list(features.values()))

        semantic_logits, box_cls, box_regression, centerness, levelness_logits = self.panoptic_head(list(features.values()))

        # Get full size semantic logits.
        downsampled_level = images.tensors[0].shape[-1] // semantic_logits.shape[-1]
        interpolated_semantic_logits_padded = F.interpolate(semantic_logits, scale_factor=downsampled_level, mode='bilinear')
        interpolated_semantic_logits = interpolated_semantic_logits_padded[:,:,:images.tensors[0].shape[-2], :images.tensors[0].shape[-1]]
        # Calculate levelness locations.
        h, w = levelness_logits.size()[-2:]
        levelness_location = self.compute_locations_per_level(h, w, self.fpn_strides[0] // 2, levelness_logits.device)
        locations.append(levelness_location)

        # Reconstruct mask from dense bounding box and semantic predictions
        panoptic_result = OrderedDict()
        boxes = self.panoptic_from_dense_bounding_box.process(
            locations, box_cls, box_regression, centerness, levelness_logits, semantic_logits, images.image_sizes
        )
        panoptic_result["instance_segmentation_result"] = boxes
        panoptic_result["semantic_segmentation_result"] = interpolated_semantic_logits
        return panoptic_result, {}

    def compute_locations(self, features):
        """Compute corresponding pixel location for feature maps.

        Parameters
        ----------
        features: list of torch.Tensor
            List of feature maps.

        Returns
        -------
        locations: list of torch.Tensor
            List of pixel location corresponding to the list of features.
        """
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(h, w, self.fpn_strides[level], feature.device)
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        """Compute corresponding pixel location for a feature map in pyramid space with certain stride.

        Parameters
        ----------
        h: int
            height of current feature map.

        w: int
            width of current feature map.

        stride: int
            stride level of current feature map with respect to original input.

        device: torch.device
            device to create return tensor.

        Returns
        -------
        locations: torch.Tensor
            pixel location map.
        """
        shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


class PanopticHead(torch.nn.Module):
    """Network module of Panoptic Head extracting features from FPN feature maps.

    Parameters
    ----------
    num_classes: int
        Number of total classes, including 'things' and 'stuff'.

    things_num_classes: int
        number of thing classes. 
    
    num_fpn_levels: int
        Number of FPN levels.

    fpn_strides: list 
        FPN strides at each FPN scale.

    in_channels: int
        Number of channels of the input features (output of FPN)

    norm_reg_targets: bool
        If true, train on normalized target.

    centerness_on_reg: bool
        If true, regress centerness on box tower of FCOS.

    fcos_num_convs: int
        number of convolution modules used in FCOS towers.

    fcos_norm: str
        Normalization layer type used in FCOS modules. 

    prior_prob: float
        Initial probability for focal loss. See `https://arxiv.org/pdf/1708.02002.pdf` for more details.
    """
    def __init__(
            self,
            num_classes,
            things_num_classes,
            num_fpn_levels,
            fpn_strides,
            in_channels,
            norm_reg_targets=False,
            centerness_on_reg=True,
            fcos_num_convs=4,
            fcos_norm='GN',
            prior_prob=0.01,
            ):
        super(PanopticHead, self).__init__()
        self.fpn_strides = fpn_strides
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg

        cls_tower = []
        bbox_tower = []

        mid_channels = in_channels // 2

        for i in range(fcos_num_convs):
            # Class branch
            if i == 0:
                cls_tower.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1))
            else:
                cls_tower.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1))
            if fcos_norm == "GN":
                cls_tower.append(nn.GroupNorm(mid_channels // 8, mid_channels))
            elif fcos_norm == "BN":
                cls_tower.append(nn.BatchNorm2d(mid_channels))
            elif fcos_norm == "SBN":
                cls_tower.append(apex.parallel.SyncBatchNorm(mid_channels))
            cls_tower.append(nn.ReLU())

            # Box regression branch
            bbox_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))

            if fcos_norm == "GN":
                bbox_tower.append(nn.GroupNorm(in_channels // 8, in_channels))
            elif fcos_norm == "BN":
                bbox_tower.append(nn.BatchNorm2d(in_channels))
            elif fcos_norm == "SBN":
                bbox_tower.append(apex.parallel.SyncBatchNorm(in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        self.cls_logits = nn.Conv2d(mid_channels * 5, num_classes, kernel_size=3, stride=1, padding=1)
        self.box_cls_logits = nn.Conv2d(mid_channels, things_num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        self.levelness = nn.Conv2d(in_channels * 5, num_fpn_levels + 1, kernel_size=3, stride=1, padding=1)

        # initialization
        to_initialize = [
            self.bbox_tower, self.cls_logits, self.cls_tower, self.bbox_pred, self.centerness, self.levelness,
            self.box_cls_logits
        ]

        for modules in to_initialize:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = prior_prob
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        box_cls = []
        logits = []
        bbox_reg = []
        centerness = []
        levelness = []

        downsampled_shape = x[0].shape[2:]
        box_feature_map_downsampled_shape = torch.Size((downsampled_shape[0] * 2, downsampled_shape[1] * 2))

        for l, feature in enumerate(x):
            # bbox
            box_tower = self.bbox_tower(feature)

            # class
            cls_tower = self.cls_tower(feature)
            box_cls.append(self.box_cls_logits(cls_tower))
            logits.append(F.interpolate(cls_tower, size=downsampled_shape, mode='bilinear'))

            # centerness
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            # bbox regression
            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred.clamp(max=math.log(10000))))

            # levelness prediction
            levelness.append(F.interpolate(box_tower, size=box_feature_map_downsampled_shape, mode='bilinear'))

        # predict levelness
        levelness = torch.cat(levelness, dim=1)
        # levelness = torch.stack(levelness, dim=0).sum(dim=0)
        levelness_logits = self.levelness(levelness)

        # level attention for semantic segmentation
        logits = torch.cat(logits, 1)
        semantic_logits = self.cls_logits(logits)

        return semantic_logits, box_cls, bbox_reg, centerness, levelness_logits


