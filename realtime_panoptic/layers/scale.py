# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from FCOS
# https://github.com/tianzhi0549/FCOS/blob/master/fcos_core/layers/scale.py

import torch
from torch import nn


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        """Scale layer with trainable scale factor.

        Parameters
        ----------
        init_value: float
            Initial value of the scale factor.
        """
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
