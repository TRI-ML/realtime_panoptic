# Copyright 2020 Toyota Research Institute.  All rights reserved.
from torch import nn
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import (FeaturePyramidNetwork,
                                                     LastLevelP6P7)

# This class is adapted from "BackboneWithFPN" in torchvision.models.detection.backbone_utils

class ResNetWithModifiedFPN(nn.Module):
    """Adds a p67-FPN on top of a ResNet model with more options.

    We adopt this function from torchvision.models.detection.backbone_utils.
    Modification has been added to enable RetinaNet style FPN with P6 P7 as extra blocks.

    Parameters
    ----------
    backbone_name: string 
        Resnet architecture supported by torchvision. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
         'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'

    norm_layer: torchvision.ops
        It is recommended to use the default value. For details visit:
        (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)

    pretrained: bool
        If True, returns a model with backbone pre-trained on Imagenet. Default: False

    trainable_layers: int
        Number of trainable (not frozen) resnet layers starting from final block.
        Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.

    out_channels: int
        number of channels in the FPN.
    """

    def __init__(self, backbone_name, pretrained=False, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=3, out_channels = 256):
        super().__init__()
        # Get ResNet
        backbone = resnet.__dict__[backbone_name](pretrained=pretrained, norm_layer=norm_layer)
        # select layers that wont be frozen
        assert 0 <= trainable_layers <= 5
        layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
        # freeze layers only if pretrained backbone is used
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

        in_channels_stage2 = backbone.inplanes // 8
        self.in_channels_list = [
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list[1:],  # nonzero only
            out_channels=out_channels,
            extra_blocks=LastLevelP6P7(out_channels, out_channels),
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        keys = list(x.keys())
        for idx, key in enumerate(keys):
            if self.in_channels_list[idx] == 0:
                del x[key]
        x = self.fpn(x)
        return x
