# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn
from torch.optim import SGD

import numpy as np

import MinkowskiEngine as ME

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from .resfieldnet import ResNetBase
from .stylegan1_3d import FullyConnectedLayer, MappingNetwork, set_feature


class SparseFullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

        if self.activation == 'relu':
            self.actv = ME.MinkowskiReLU(inplace=True)
        elif self.activation == 'lrelu':
            self.actv = ME.MinkowskiLeakyReLU(negative_slope=0.2)
        else:
            raise Exception(f'Unsupported activation: {self.activation}.')

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        
        x = ME.MinkowskiFunctional.linear(x, weight=w, bias=b)
        x = self.actv(x)
        return x

class StyleUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, z_dim, w_dim, D=3):
        self.w_dim = w_dim
        self.z_dim = z_dim
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        # Decoder
        self.num_waffine = 0

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        # self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.waffine4 = FullyConnectedLayer(self.w_dim, self.PLANES[4], bias_init=1)
        self.num_waffine += 1
        self.instnorm4 = ME.MinkowskiInstanceNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        # self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.waffine5 = FullyConnectedLayer(self.w_dim, self.PLANES[5], bias_init=1)
        self.num_waffine += 1
        self.instnorm5 = ME.MinkowskiInstanceNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        # self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.waffine6 = FullyConnectedLayer(self.w_dim, self.PLANES[6], bias_init=1)
        self.num_waffine += 1
        self.instnorm6 = ME.MinkowskiInstanceNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        # self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])
        self.waffine7 = FullyConnectedLayer(self.w_dim, self.PLANES[7], bias_init=1)
        self.num_waffine += 1
        self.instnorm7 = ME.MinkowskiInstanceNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.out5 = self._make_out_layer(self.PLANES[4] * self.BLOCK.expansion, out_channels, bias=True, dimension=D)
        self.out6 = self._make_out_layer(self.PLANES[5] * self.BLOCK.expansion, out_channels, bias=True, dimension=D)
        self.out7 = self._make_out_layer(self.PLANES[6] * self.BLOCK.expansion, out_channels, bias=True, dimension=D)
        self.out8 = self._make_out_layer(self.PLANES[7] * self.BLOCK.expansion, out_channels, bias=True, dimension=D)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.leaky_relu = ME.MinkowskiLeakyReLU(negative_slope=0.2)

        self.latent_mapping = MappingNetwork(self.z_dim, c_dim=0, w_dim=self.w_dim, num_layers=2, num_ws=self.num_waffine)
        # self.out5 = ME.MinkowskiConvolution(
        #     self.PLANES[4] * self.BLOCK.expansion,
        #     out_channels,
        #     kernel_size=1,
        #     bias=True,
        #     dimension=D)

        # self.out6 = ME.MinkowskiConvolution(
        #     self.PLANES[5] * self.BLOCK.expansion,
        #     out_channels,
        #     kernel_size=1,
        #     bias=True,
        #     dimension=D)

        # self.out7 = ME.MinkowskiConvolution(
        #     self.PLANES[6] * self.BLOCK.expansion,
        #     out_channels,
        #     kernel_size=1,
        #     bias=True,
        #     dimension=D)

        # self.out8 = ME.MinkowskiConvolution(
        #     self.PLANES[7] * self.BLOCK.expansion,
        #     out_channels,
        #     kernel_size=1,
        #     bias=True,
        #     dimension=D)

    def _make_out_layer(self, in_channels, out_channels, bias, dimension):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
                dimension=dimension
            ),
            ME.MinkowskiInstanceNorm(out_channels),
            ME.MinkowskiLeakyReLU(negative_slope=0.2)
        )
        # return SparseFullyConnectedLayer(in_channels, out_channels, activation='lrelu')

    def forward(self, x, z):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        # print('after block1:', out_b1p2.coordinate_map_key)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        # print('after block2:', out_b2p4.coordinate_map_key)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        # print('after block3:', out_b3p8.coordinate_map_key)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)
        # print('after block4:', out.coordinate_map_key)

        # ============================================================= #
        # Decoder

        ws = self.latent_mapping(z=z, c=None)
        w_iter = iter(ws.unbind(dim=1))

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        # out = self.bntr4(out)
        # out = self.relu(out)
        out = self.instnorm4(out)
        s4 = self.waffine4(next(w_iter))
        out = out * s4
        # print(f's4: max: {s4.max().item()}, min: {s4.min().item()}, avg: {s4.mean().item()}')
        out = self.leaky_relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        y1 = self.out5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        # out = self.bntr5(out)
        # out = self.relu(out)
        out = self.instnorm5(out)
        s5 = self.waffine5(next(w_iter))
        out = out * s5
        # print(f's5: max: {s5.max().item()}, min: {s5.min().item()}, avg: {s5.mean().item()}')
        out = self.leaky_relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        y2 = self.out6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        # out = self.bntr6(out)
        # out = self.relu(out)
        out = self.instnorm6(out)
        s6 = self.waffine6(next(w_iter))
        # print(f's6: max: {s6.max().item()}, min: {s6.min().item()}, avg: {s6.mean().item()}')
        out = out * s6
        out = self.leaky_relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        y3 = self.out7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        # out = self.bntr7(out)
        # out = self.relu(out)
        out = self.instnorm7(out)
        s7 = self.waffine7(next(w_iter))
        # print(f's7: max: {s7.max().item()}, min: {s7.min().item()}, avg: {s7.mean().item()}')
        out = out * s7
        out = self.leaky_relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)

        y4 = self.out8(out)

        # y = torch.cat([y1.F, y2.F, y3.F, y4.F], dim=0)
        # std = torch.std(y, dim=0)
        # mean = torch.mean(y, dim=0)
        # print(f'std: {std.tolist()}')
        # print(f'mean: {mean.tolist()}')
        # input('aaa')

        return y1, y2, y3, y4   # resolution: low to high


class StyleNet14(StyleUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class StyleNet18(StyleUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class StyleNet34(StyleUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class StyleNet50(StyleUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class StyleNet101(StyleUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class StyleNet14A(StyleNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class StyleNet14B(StyleNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class StyleNet14C(StyleNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class StyleNet14D(StyleNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class StyleNet18A(StyleNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class StyleNet18B(StyleNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class StyleNet18D(StyleNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class StyleNet34A(StyleNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class StyleNet34B(StyleNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class StyleNet34C(StyleNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

class StyleFieldUNetBase(StyleUNetBase):
    def network_initialization(self, in_channels, out_channels, D):
        field_ch = 32
        field_ch2 = 64
        self.field_network = nn.Sequential(
            ME.MinkowskiSinusoidal(in_channels, field_ch),
            ME.MinkowskiBatchNorm(field_ch, track_running_stats=False),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(field_ch, field_ch),
            ME.MinkowskiBatchNorm(field_ch, track_running_stats=False),
            ME.MinkowskiReLU(inplace=True),
            # ME.MinkowskiToSparseTensor(),
        )
        self.field_network2 = nn.Sequential(
            ME.MinkowskiSinusoidal(field_ch + in_channels, field_ch2),
            ME.MinkowskiBatchNorm(field_ch2, track_running_stats=False),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(field_ch2, field_ch2),
            ME.MinkowskiBatchNorm(field_ch2, track_running_stats=False),
            ME.MinkowskiReLU(inplace=True),
            # ME.MinkowskiToSparseTensor(),
        )

        StyleUNetBase.network_initialization(self, field_ch2, out_channels, D)

    def forward(self, x: ME.TensorField, z):
        # assert(x.C[:, 0].max().item() == 0) # so far only support batch size == 1
        # print('input:', x.coordinate_map_key)
        x = set_feature(x, torch.rand_like(x.F))
        otensor = self.field_network(x)
        otensor = ME.cat(otensor, x)
        otensor2 = self.field_network2(otensor)
        otensor2 = otensor2.sparse()
        return StyleUNetBase.forward(self, otensor2, z)

class StyleFieldUNet14(StyleFieldUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)

class StyleFieldUNet14A(StyleFieldUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)

class StyleFieldUNet50(StyleFieldUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

class StyleFieldUNet50Small(StyleFieldUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256//4, 128//4, 96//4, 96//4)