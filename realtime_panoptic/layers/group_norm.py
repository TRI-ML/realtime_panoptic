import torch
import torch.nn as nn


class ONNXAbleGN(nn.Module):
    r"""This is a drop-in reimplementation of GroupNorm. ONNX trace based
    exporting should work with this layer. TensorRT works as well.

    Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization`_ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494
    """
    __constants__ = [
        'num_groups', 'num_channels', 'eps', 'affine', 'weight', 'bias'
    ]

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(ONNXAbleGN, self).__init__()
        self.num_groups = int(min(num_groups, num_channels))
        self.num_channels = int(num_channels)
        self.grouped_channels = self.num_channels // self.num_groups
        self.affine = affine
        self.eps = eps
        self.norm = nn.InstanceNorm2d(self.num_groups, self.eps, 0.0, False,
                                      False)

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        N, C, H, W = input.shape
        N, C, H, W = int(N), int(C), int(H), int(W)
        x = torch.reshape(input,
                          [N, self.num_groups, self.grouped_channels, H * W])
        x = self.norm(x)
        x = torch.reshape(x, [N, C, H, W])

        if self.affine:
            return x * self.weight[:, None, None] + self.bias[:, None, None]

        else:
            return x

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)
