import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
from torch import Tensor

NOISE_SCALE = 1e-12

class LinearConv2D(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation
            ,groups, bias, padding_mode)

        self.linear_weight = Parameter(torch.zeros_like(self.weight, requires_grad=True) + NOISE_SCALE)
        if bias:
            self.linear_bias = Parameter(torch.zeros_like(self.bias, requires_grad=True) + NOISE_SCALE)
        else:
            self.linear_bias = None

    def _linear_forward(self, input: Tensor) -> Tensor:
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self.linear_weight, self.linear_bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, self.linear_weight, self.linear_bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def _conv_forward(self, input: Tensor, add_bias: bool = True) -> Tensor:
        bias = self.bias if add_bias else None
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self.weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, self.weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor, input_jvp: Tensor) -> Tensor:
        output = self._conv_forward(input)
        output_jvp = self._linear_forward(input) + self._conv_forward(input_jvp, add_bias=False)
        return output, output_jvp

class LinearLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        self.linear_weight = Parameter(torch.zeros_like(self.weight, requires_grad=True) + NOISE_SCALE)
        if bias:
            self.linear_bias = Parameter(torch.zeros_like(self.bias, requires_grad=True) + NOISE_SCALE)
        else:
            self.linear_bias = None

    def forward(self, input: Tensor, input_jvp: Tensor) -> Tensor:
        output = super().forward(input)
        output_jvp = F.linear(input, self.linear_weight, self.linear_bias) + F.linear(input_jvp, self.weight, None)
        return output, output_jvp

class LinearReLU(nn.ReLU):

    def forward(self, input: Tensor, input_jvp: Tensor) -> Tensor:
        output = super().forward(input)
        output_jvp = input_jvp * (output > 0).float()
        return output, output_jvp

class LinearSequential(nn.Sequential):

    def forward(self, *input):
        for module in self:
            input = module(*input)
        return input

class LinearBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine=True,
                 track_running_stats=True) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        if affine:
            self.linear_weight = Parameter(torch.zeros_like(self.weight, requires_grad=True) + NOISE_SCALE)
            self.linear_bias = Parameter(torch.zeros_like(self.bias, requires_grad=True) + NOISE_SCALE)

    def _linear_forward(self, input: Tensor) -> Tensor:
        return F.batch_norm(input, self.running_mean, self.running_var,
                            self.linear_weight, self.linear_bias, training=False, momentum=self.momentum, eps=self.eps)

    def _jvp_forward(self, input: Tensor) -> Tensor:
        return F.batch_norm(input,
                            torch.zeros_like(self.running_mean), self.running_var, self.weight, None, training=False,
                            momentum=self.momentum, eps=self.eps)

    def forward(self, input: Tensor, input_jvp:Tensor) -> Tensor:
        output = super().forward(input)
        output_jvp = self._linear_forward(input) + self._jvp_forward(input_jvp)
        return output, output_jvp

class LinearAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):

    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def forward(self, input, input_jvp):
        return super().forward(input), super().forward(input_jvp)

