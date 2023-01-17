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

import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck


class BasicBlockInstanceNorm(BasicBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = ME.MinkowskiInstanceNorm(args[1])
        self.norm2 = ME.MinkowskiInstanceNorm(args[1])


class BottleneckInstanceNorm(Bottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = ME.MinkowskiInstanceNorm(args[1])
        self.norm2 = ME.MinkowskiInstanceNorm(args[1])
        self.norm3 = ME.MinkowskiInstanceNorm(args[1] * self.expansion)


class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):

        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, self.inplanes, kernel_size=7, dilation=2, stride=2, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),
        )

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2
        )

        self.out1 = ME.MinkowskiLinear(self.BLOCK.expansion * self.PLANES[0], out_channels)
        self.out2 = ME.MinkowskiLinear(self.BLOCK.expansion * self.PLANES[1], out_channels)
        self.out3 = ME.MinkowskiLinear(self.BLOCK.expansion * self.PLANES[2], out_channels)
        self.out4 = ME.MinkowskiLinear(self.BLOCK.expansion * self.PLANES[3], out_channels)

        # self.conv5 = nn.Sequential(
        #     ME.MinkowskiConvolution(
        #         self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D
        #     ),
        #     ME.MinkowskiInstanceNorm(self.inplanes),
        #     ME.MinkowskiGELU(),
        # )

        # self.glob_pool = ME.MinkowskiGlobalMaxPooling()

        # self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            
            if isinstance(m, ME.MinkowskiInstanceNorm):
                m.reset_parameters()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                # ME.MinkowskiBatchNorm(planes * block.expansion),
                ME.MinkowskiInstanceNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: ME.SparseTensor):
        x = self.conv1(x)

        x = self.layer1(x)
        y1 = self.out1(x)

        x = self.layer2(x)
        y2 = self.out2(x)

        x = self.layer3(x)
        y3 = self.out3(x)

        x = self.layer4(x)
        y4 = self.out4(x)

        # x = self.conv5(x)
        # x = self.glob_pool(x)
        # return self.final(x)

        return y1, y2, y3, y4



class ResNet14(ResNetBase):
    BLOCK = BasicBlockInstanceNorm
    LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
    BLOCK = BasicBlockInstanceNorm
    LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
    BLOCK = BasicBlockInstanceNorm
    LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
    BLOCK = BottleneckInstanceNorm
    LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
    BLOCK = BottleneckInstanceNorm
    LAYERS = (3, 4, 23, 3)


class ResFieldNetBase(ResNetBase):
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
            ME.MinkowskiToSparseTensor(),
        )
        self.field_network2 = nn.Sequential(
            ME.MinkowskiSinusoidal(field_ch + in_channels, field_ch2),
            ME.MinkowskiBatchNorm(field_ch2, track_running_stats=False),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(field_ch2, field_ch2),
            ME.MinkowskiBatchNorm(field_ch2, track_running_stats=False),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiToSparseTensor(),
        )

        ResNetBase.network_initialization(self, field_ch2, out_channels, D)

    def forward(self, x: ME.TensorField):
        # assert(x.C[:, 0].max().item() == 0) # so far only support batch size == 1
        otensor = self.field_network(x)
        otensor2 = self.field_network2(otensor.cat_slice(x))
        return ResNetBase.forward(self, otensor2)


class ResFieldNet14(ResFieldNetBase):
    BLOCK = BasicBlockInstanceNorm
    LAYERS = (1, 1, 1, 1)


class ResFieldNet18(ResFieldNetBase):
    BLOCK = BasicBlockInstanceNorm
    LAYERS = (2, 2, 2, 2)


class ResFieldNet34(ResFieldNetBase):
    BLOCK = BasicBlockInstanceNorm
    LAYERS = (3, 4, 6, 3)


class ResFieldNet50(ResFieldNetBase):
    BLOCK = BottleneckInstanceNorm
    LAYERS = (3, 4, 6, 3)


class ResFieldNet101(ResFieldNetBase):
    BLOCK = BottleneckInstanceNorm
    LAYERS = (3, 4, 23, 3)





class ResFieldNet50Hierarchical(ResFieldNetBase):
    BLOCK = BottleneckInstanceNorm
    LAYERS = (3, 4, 6, 3)
    PLANES = (32, 64, 128, 128)

