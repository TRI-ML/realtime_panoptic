# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os

from yacs.config import CfgNode as CN
cfg = CN()
########################################################################################################################
### MODEL
########################################################################################################################
cfg.model = CN()
cfg.model.name = ''                             # Training model
cfg.model.backbone = ''                         # Backbone
cfg.model.checkpoint_path = ''                  # Checkpoint path for model saving

cfg.model.panoptic = CN()
cfg.model.panoptic.num_classes = 19                    # number of total classes
cfg.model.panoptic.num_thing_classes = 8               # number of thing classes
cfg.model.panoptic.pre_nms_thresh = 0.05               # objectness threshold before NMS
cfg.model.panoptic.pre_nms_top_n = 1000                # max num of accepted bboxes before NMS
cfg.model.panoptic.nms_thresh = 0.6                    # NMS threshold
cfg.model.panoptic.fpn_post_nms_top_n = 100          # Top detection post NMS     
# for cityscapes, it is (11,18), for COCO it is (0,79), for Vistas it is (0,36)
cfg.model.panoptic.instance_id_range = (11, 18)


########################################################################################################################
### INPUT
########################################################################################################################
cfg.input = CN()
cfg.input.pixel_mean = [102.9801, 115.9465, 122.7717]
cfg.input.pixel_std = [1., 1., 1.]
# Convert image to BGR format, in range 0-255
cfg.input.to_bgr255 = True
