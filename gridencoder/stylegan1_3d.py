# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Modified by Katja Schwarz for VoxGRAF: Fast 3D-Aware Image Synthesis with Sparse Voxel Grids
#

"""Network architectures from the paper
"Analyzing and Improving the Image Quality of StyleGAN".
Matches the original implementation of configs E-F by Karras et al. at
https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import MinkowskiEngine as ME
from MinkowskiEngineBackend._C import ConvolutionMode, PoolingMode
from MinkowskiEngine.MinkowskiKernelGenerator import KernelGenerator
# from MinkowskiEngine.MinkowskiSparseTensor import _get_coordinate_map_key

# ------------------------------------------------------------------------ #
# Latent Mapping

class FullyConnectedLayer(torch.nn.Module):
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

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = torch.matmul(w.t())

        if self.activation == 'linear':
            x = x
        elif self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'lrelu':
            x = F.leaky_relu(x)
        else:
            raise Exception(f'Unsupported activation: {self.activation}.')
        return x

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    # StyleGAN3: truncation_psi=0.5, truncation_cutoff=8
    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # Embed, normalize, and concat inputs.
        x = None
        if self.z_dim > 0:
            # misc.assert_shape(z, [None, self.z_dim])
            x = normalize_2nd_moment(z.to(torch.float32))
        if self.c_dim > 0:
            # misc.assert_shape(c, [None, self.c_dim])
            y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            assert self.w_avg_beta is not None
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

# ------------------------------------------------------------------------ #
# Generation

def sparseconv3d(x, w, f=None, up=1, down=1, padding=0):
    # params
    is_transpose = (up == 2)
    stride = 2 if is_transpose else 1
    kernel_size = w.shape[-1]

    # generate kernel
    w = w.flatten(2).permute(2, 1, 0).contiguous()
    kernel_generator = KernelGenerator(
        kernel_size=kernel_size,
        stride=stride,
        dilation=1,
        expand_coordinates=is_transpose,
        dimension=3,
    )

    # params
    conv = ME.MinkowskiConvolutionTransposeFunction if is_transpose else ME.MinkowskiConvolutionFunction

    # Get a new coordinate_map_key or extract one from the coords
    # out_coordinate_map_key = _get_coordinate_map_key(x, None, kernel_generator.expand_coordinates)
    x_stride = np.array(x.coordinate_map_key.get_tensor_stride())
    out_stride = x_stride // 2 if is_transpose else x_stride
    out_coordinate_map_key = x.coordinate_manager._get_coordinate_map_key(out_stride)

    outfeat = conv.apply(
        x.F.float(),
        w.float(),
        kernel_generator,
        ConvolutionMode.DEFAULT,
        x.coordinate_map_key,
        out_coordinate_map_key,
        x.coordinate_manager,
    )

    outfeatsparse = ME.SparseTensor(outfeat, coordinate_map_key=out_coordinate_map_key, coordinate_manager=x.coordinate_manager)

    # if is_transpose:
    #     # kick out the max values, equivalent to cropping: out = out[..., :-1, :-1, :-1]
    #     _, max_coord = get_cur_min_max_coordinate(outfeatsparse, final_resolution=final_resolution, start_resolution=start_resolution)

    #     # prune (don't calculate on batch dimension!)
    #     max_idcs = torch.any(outfeatsparse.C[:, 1:] == max_coord, dim=1)
    #     outfeatsparse = ME.MinkowskiPruning()(outfeatsparse, ~max_idcs)

    return outfeatsparse

'''
init: x, weight, style
if not demodulate:
    x = x * style
    x = conv(x, w=weight)
    return x
else:
    w = weight * style
    decoef = sqrt(w.square())
    if not fused_modconv:
        x = x * style
        x = conv(x, w=weight)
        x = x * decoef
    else:
        w = w * decoef
        x = conv(x, w)
'''
def modulated_sparseconv3d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    demodulate      = False,     # Apply weight demodulation?
    fused_modconv   = False,    # Perform modulation, convolution, and demodulation as a single fused operation?
    input_gain      = None,     # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    batch_size = styles.shape[0]
    out_channels, in_channels, kh, kw, kd = weight.shape

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw * kd) / weight.norm(float('inf'), dim=[1,2,3,4], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # weight is at 10^(-1) level
    # x = sparseconv3d(x=x, w=weight.to(x.dtype), up=up, down=down, padding=padding)

    # Pre-normalize inputs.
    weight = weight * weight.square().mean([1,2,3,4], keepdim=True).rsqrt() # [OIkkk]
    styles = styles * styles.square().mean().rsqrt()    # [NI]
    # Calculate per-sample weights and demodulation coefficients.
    if demodulate or fused_modconv:
        styled_weight = weight.unsqueeze(0) * styles.reshape(batch_size, 1, -1, 1, 1, 1) # [NOIkkk]
    if demodulate:
        dcoefs = (styled_weight.square().sum(dim=[2,3,4,5]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        styled_weight = styled_weight * dcoefs.reshape(batch_size, -1, 1, 1, 1, 1) # [NOIkkk]
        styled_weight = styled_weight[0]    # Since batch_size=1, using group conv if batch_size > 1

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        styles = torch.index_select(styles, 0, x.C[:, 0])
        x = ME.SparseTensor(
            x.F * styles,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key
        )
        x = sparseconv3d(x=x, w=weight.to(x.dtype), up=up, down=down, padding=padding)
        # demodulate
        if demodulate:
            dcoefs = torch.index_select(dcoefs, 0, x.C[:, 0])
            x = ME.SparseTensor(
                x.F * dcoefs,
                coordinate_manager=x.coordinate_manager,
                coordinate_map_key=x.coordinate_map_key,
            )
    else:
        x = sparseconv3d(x=x, w=styled_weight.to(x.dtype), up=up, down=down, padding=padding)

    if input_gain != None:
        input_gain = input_gain.expand(*x.F.shape)
        x = ME.SparseTensor(
            x.F * input_gain,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key,
        )

    if noise is not None:
        x += noise

    return x

# Add bias b to the activation
def bias_lrelu_sparse(x, b=None, dim=1, alpha=None, gain=None, clamp=None):
    assert clamp is None or clamp >= 0

    def_alpha, def_gain, def_clamp = 0.2, np.sqrt(2), -1
    alpha = float(alpha if alpha is not None else def_alpha)
    gain = float(gain if gain is not None else def_gain)
    clamp = float(clamp if clamp is not None else def_clamp)

    # Add bias.
    if b is not None:
        b = b.unsqueeze(0).repeat(x.shape[0], 1)
        b = ME.SparseTensor(
            features=b,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key,
        )
        x += b

    x = ME.MinkowskiFunctional.leaky_relu(x, alpha) # negative_slope = alpha = 0.2? default is 0.01

    # Scale by gain.
    gain = torch.tensor(gain, dtype=float)  # why scale by gain?
    if gain != 1:
        x = x * gain

    # Clamp.
    if clamp >= 0:
        x = ME.MinkowskiFunctional._wrap_tensor(x, torch.clamp(x.F, min=-clamp, max=clamp))
    return x

class StyleGAN2_3D_Layer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        conv_clamp      = 256,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        magnitude_ema_beta  = 0.999,    # Decay rate for the moving average of input magnitudes.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.up = up
        self.conv_clamp = conv_clamp
        # self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2

        self.magnitude_ema_beta = magnitude_ema_beta
        self.register_buffer('magnitude_ema', torch.ones([]))

        self.affine = FullyConnectedLayer(w_dim, out_channels, bias_init=1)
        self.inst_norm = ME.MinkowskiInstanceNorm(out_channels)

        if up == 2:
            self.conv = ME.MinkowskiConvolutionTranspose(in_channels, out_channels, kernel_size, 
                            stride=2, bias=True, dimension=3)
        elif up == 1:
            self.conv = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size, 
                            stride=1, bias=True, dimension=3)

        self.actv = ME.MinkowskiLeakyReLU(negative_slope=0.2)

    def forward(self, x, w, update_emas=False):
        # Track input magnitude.
        if update_emas:
            magnitude_cur = x.F.detach().square().mean()
            self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
        input_gain = self.magnitude_ema.rsqrt()

        x = self.conv(x)    # [NOkkk]
        x = self.inst_norm(x)   # [NIkkk]

        styles = self.affine(w) # [NI]
        x = x * styles  # [NIkkk]

        x = self.actv(x)
        return x

class StyleGAN2_3D_Block(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        w_dim,
        is_first,
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.is_first = is_first

        self.num_conv = 0

        if not is_first:
            self.conv0 = StyleGAN2_3D_Layer(in_channels, out_channels, w_dim=w_dim, up=1, 
                                            conv_clamp=conv_clamp)
            self.num_conv += 1

            self.conv1 = StyleGAN2_3D_Layer(out_channels, out_channels, w_dim=w_dim, up=2,
                                            conv_clamp=conv_clamp)
            self.num_conv += 1

            # self.conv2 = StyleGAN2_3D_Layer(out_channels, out_channels, w_dim=w_dim, up=1,
            #                                 conv_clamp=conv_clamp)
            # self.num_conv += 1
        else:
            self.conv0 = StyleGAN2_3D_Layer(in_channels, out_channels, w_dim=w_dim, up=1,
                                            conv_clamp=conv_clamp)
            self.num_conv += 1
            
            self.conv1 = StyleGAN2_3D_Layer(out_channels, out_channels, w_dim=w_dim, up=2,
                                            conv_clamp=conv_clamp)
            self.num_conv += 1

    def forward(self, x, ws, update_emas):
        w_iter = iter(ws.unbind(dim=1))

        dtype = torch.float32

        # Main layers.
        if self.is_first:
            x = self.conv0(x, next(w_iter))
            x = self.conv1(x, next(w_iter))
        else:
            x = self.conv0(x, next(w_iter))
            x = self.conv1(x, next(w_iter))
            # x = self.conv2(x, next(w_iter))
        return x
    
def set_feature(x, features):
    if isinstance(x, ME.TensorField):
        return ME.TensorField(
            features=features.to(x.device),
            coordinate_field_map_key=x.coordinate_field_map_key,
            coordinate_manager=x.coordinate_manager,
            quantization_mode=x.quantization_mode,
            device=x.device,
        )
    elif isinstance(x, ME.SparseTensor):
        return ME.SparseTensor(
            features=features.to(x.device),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device,
        )
    else:
        pass
    raise ValueError("Input tensor is not ME.TensorField nor ME.SparseTensor.")
    return None

class StyleGAN2_3D_Generator(torch.nn.Module):
    def __init__(self, 
        z_dim,
        w_dim,
        num_blocks=5,
        num_latent_mapping_layers=4,
        feature_out_channels=8,
        init_seed_channels=8
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_blocks = num_blocks
        self.num_latent_mapping_layers = num_latent_mapping_layers
        self.feature_out_channels = feature_out_channels

        self.init_seed_channels = init_seed_channels
        # init range from (-64m, 64m), (-64m, 64m), (-4.8m, 27.2m)
        self.init_stride = 32 # 1.6m
        self.init_lower = torch.Tensor([-40, -40, -3]).int()
        self.init_upper = torch.Tensor([40, 40, 17]).int()
        self.init_shape = self.init_upper - self.init_lower # block shape is (80, 80, 20)
        self.to_init_index = torch.Tensor([self.init_shape[1] * self.init_shape[2], self.init_shape[2], 1]).int()
        # self.init_coords = torch.stack(torch.meshgrid(
        #     torch.arange(self.init_lower[0], self.init_upper[0]) * self.init_stride,
        #     torch.arange(self.init_lower[1], self.init_upper[1]) * self.init_stride,
        #     torch.arange(self.init_lower[2], self.init_upper[2]) * self.init_stride,
        # )).reshape((3, -1)).transpose(0, 1).int()
        # self.init_coords_batch = lambda b: torch.cat([torch.ones(self.init_coords.shape[0], 1) * int(b), self.init_coords], dim=1)
        self.init_seed_features = torch.nn.Parameter(torch.randn(torch.prod(self.init_shape).int(), init_seed_channels))

        self.block_out_channels = [256, 256, 128, 64, 32]
        self.block_in_channels = [init_seed_channels] + self.block_out_channels[:-1]

        # SynthesisBlocks
        self.num_ws = 0
        for block_idx in range(1, num_blocks+1):
            in_channels = self.block_in_channels[block_idx-1]
            out_channels = self.block_out_channels[block_idx-1]
            block = StyleGAN2_3D_Block(in_channels, out_channels, w_dim, is_first=(block_idx==1), conv_clamp=256)
            self.num_ws += block.num_conv
            setattr(self, f'b{block_idx}', block)

        # Mapping Network
        self.latent_mapping = MappingNetwork(z_dim=z_dim, c_dim=0, w_dim=w_dim, 
                                        num_layers=self.num_latent_mapping_layers, num_ws=self.num_ws)

        self.pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        # Out feature blocks
        for block_idx in range(num_blocks-3, num_blocks+1):
            # out_block = ME.MinkowskiConvolution(self.block_out_channels[block_idx-1], feature_out_channels, 
            #                                     kernel_size=1, bias=True, dimension=3)
            out_block = self._make_out_layer(self.block_out_channels[block_idx-1], feature_out_channels, bias=False)
            setattr(self, f'out_b{block_idx}', out_block)

    def forward(self, pts, z, update_emas):
        x = set_feature(pts, torch.ones_like(pts.F))

        # print(f'before pooling: {torch.cuda.memory_allocated(device=0)/(1024**3)}, #pts: {x.shape}')
        # Pool the input clouds to generate the coordinate map key for the feature of each stylegan generative layer
        with torch.no_grad():
            for block_idx in range(1, self.num_blocks+1):
                x = self.pool(x)
        # print(f'after pooling: {torch.cuda.memory_allocated(device=0)/(1024**3)}.')

        # Sample init input from self.init_seed
        idx = torch.round(x.C[:, 1:] / self.init_stride).int()
        idx = torch.clamp(idx, self.init_lower.to(x.F.device), self.init_upper.to(x.F.device) - 1)
        idx -= self.init_lower.to(x.F.device)
        # print('idx:', idx.dtype, idx.shape)
        # print('to_init_index:', self.to_init_index.dtype)
        idx *= self.to_init_index.to(x.F.device)
        idx = idx.sum(dim=1).long()
        x = set_feature(x, self.init_seed_features[idx])

        # split ws and block forwarding
        ws = self.latent_mapping(z=z, c=None)
        # ws = torch.ones_like(ws)

        w_idx = 0
        features_out = []
        for block_idx in range(1, self.num_blocks+1):
            block = getattr(self, f'b{block_idx}')
            # if block_idx == 2:
            #     print(block.conv0.weight)
            #     input()
            block_ws = ws.narrow(1, w_idx, block.num_conv) # extract block_ws from the 1st dim of ws, with indices of [w_dix: w_idx+block.num_conv]
            w_idx += block.num_conv
            x = block(x, block_ws, update_emas)
            # print(f'after block{block_idx}: {torch.cuda.memory_allocated(device=0)/(1024**3)}.')
            # print(f'block{block_idx}: in: {block.in_channels}, out: {block.out_channels}, #pts: {x.shape}')

            if block_idx in range(self.num_blocks-3, self.num_blocks+1):
                out_block = getattr(self, f'out_b{block_idx}')
                feat = out_block(x)
                # print(f'{block_idx}: {feat.C.shape}, {feat.C.max().item()}, {feat.C.min().item()}, {feat.coordinate_map_key.get_tensor_stride()}')
                # print(feat)
                # feat = set_feature(feat, 10 * F.tanh(feat.F / 5.0))
                # feat = ME.MinkowskiFunctional._wrap_tensor(feat, torch.clamp(feat.F, min=-15, max=15))
                features_out.append(feat)

        return features_out # resolution: low to high

    def _make_out_layer(self, in_channels, out_channels, bias, dimension=3):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
                dimension=dimension
            ),
            # ME.MinkowskiInstanceNorm(out_channels),
            # ME.MinkowskiLeakyReLU(negative_slope=1.0)
        )

# ------------------------------------------------------------------------ #
# Discrimination

class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        padding,
        stride,
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'lrelu',     # Activation function: 'relu', 'lrelu', etc.
        use_batch_norm = True,
        use_spectral_norm = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.use_batch_norm = use_batch_norm

        # memory_format = torch.contiguous_format
        # weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        # bias = torch.zeros([out_channels]) if bias else None
        # self.weight = torch.nn.Parameter(weight)
        # self.bias = torch.nn.Parameter(bias) if bias else None
        # self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                            stride=stride, padding=padding, bias=bias)
        if use_spectral_norm:
            self.conv = nn.utils.parametrizations.spectral_norm(self.conv)

        if use_batch_norm:
            self.inst_norm = nn.InstanceNorm2d(out_channels)

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, gain=1):
        # w = self.weight * self.weight_gain
        # b = self.bias.to(x.dtype) if self.bias else None
        # x = F.conv2d(x, weight=w, bias=b, stride=self.stride, padding=self.padding)
        x = self.conv(x)

        if self.use_batch_norm:
            x = self.inst_norm(x)

        x = self.act(x)
        if gain != 1:
            x *= gain
        return x

class StyleGAN2_2D_Disc_Block(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        mid_channles,
        architecture = 'resnet',
        activation = 'lrelu',
        use_batch_norm = True,
        use_spectral_norm = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channles = mid_channles

        assert architecture in ['orig', 'resnet']   # do not support skip architecture as StyleGAN2
        self.architecture = architecture
        
        assert in_channels in [3, mid_channles]
        if in_channels == 3:
            self.fromrgb = Conv2dLayer(in_channels, mid_channles, kernel_size=1, padding=0, stride=1, 
                activation=activation, use_batch_norm = use_batch_norm, use_spectral_norm=use_spectral_norm)

        self.conv0 = Conv2dLayer(mid_channles, mid_channles, kernel_size=3, padding=1, stride=1,
            activation=activation, use_batch_norm = use_batch_norm, use_spectral_norm=use_spectral_norm)
        self.conv1 = Conv2dLayer(mid_channles, out_channels, kernel_size=3, padding=1, stride=2, 
            activation=activation, use_batch_norm = use_batch_norm, use_spectral_norm=use_spectral_norm)

        if architecture == 'resnet':
            self.bypass = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                Conv2dLayer(mid_channles, out_channels, kernel_size=1, padding=0, stride=1, bias=False, 
                            activation=activation, use_batch_norm = use_batch_norm, use_spectral_norm=use_spectral_norm)
            )

    def forward(self, x):
        if self.in_channels == 3:
            x = self.fromrgb(x)
        
        if self.architecture == 'resnet':
            residual = self.bypass(x)
            x = self.conv0(x)
            x = self.conv1(x)
            x += residual
        else:
            x = self.conv0(x)
            x = self.conv1(x)
        return x

class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

class StyleGAN2_2D_Disc_Final(torch.nn.Module):
    def __init__(self,
        in_channels,
        activation = 'lrelu',
        mbstd_num_channels = 0,
        mbstd_group_size = None,
        use_batch_norm = True,
        use_spectral_norm = False
    ):
        super().__init__()

        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None

        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, padding=1, stride=1, 
                                activation=activation, use_batch_norm=use_batch_norm, use_spectral_norm=use_spectral_norm)
        self.out = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0, stride=1)
        if use_spectral_norm:
            self.out = nn.utils.parametrizations.spectral_norm(self.out)

    def forward(self, x):
        if self.mbstd != None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.out(x)
        return x

class StyleGAN2_2D_Discriminator(torch.nn.Module):
    def __init__(self,
        num_blocks = 4,
        use_batch_norm = True,
        use_spectral_norm = False,
    ):
        super().__init__()
        self.num_blocks = num_blocks

        self.block_out_channels = [min(2**(5+i), 256) for i in range(num_blocks)] # [32, 64, 128, 256]
        self.block_in_channels = [16] + self.block_out_channels[:-1]    # [16, 32, 64, 128]

        for block_idx in range(1, num_blocks+1):
            in_channels = self.block_in_channels[block_idx-1] if block_idx != 1 else 3
            mid_channles = self.block_in_channels[block_idx-1]
            out_channels = self.block_out_channels[block_idx-1]
            block = StyleGAN2_2D_Disc_Block(in_channels, out_channels, mid_channles, 
                        activation='lrelu', use_batch_norm=use_batch_norm, 
                        use_spectral_norm=use_spectral_norm)
            setattr(self, f'b{block_idx}', block)

        self.out = StyleGAN2_2D_Disc_Final(out_channels, activation='lrelu', 
                        use_batch_norm=use_batch_norm, 
                        use_spectral_norm=use_spectral_norm)

    def forward(self, img):
        x = img
        for block_idx in range(1, self.num_blocks+1):
            block = getattr(self, f'b{block_idx}')
            x = block(x)

        x = self.out(x)
        return x
    
    def get_params(self, lr):
        return [
            {'params': self.parameters(), 'lr': lr},
        ]

from torchvision.models import resnet18

class Latent_2D_Encoder(torch.nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        resnet18_model = resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AdaptiveAvgPool2d(output_ize=1)
        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar

    def get_params(self, lr):
        return [
            {'params': self.parameters(), 'lr': lr},
        ]