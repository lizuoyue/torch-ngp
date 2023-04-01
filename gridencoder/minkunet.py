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
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
import sys

import MinkowskiEngine as ME

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck


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
                in_channels, self.inplanes, kernel_size=3, stride=2, dimension=D
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

        self.conv5 = nn.Sequential(
            ME.MinkowskiDropout(),
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiGELU(),
        )

        self.glob_pool = ME.MinkowskiGlobalMaxPooling()

        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

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
                ME.MinkowskiBatchNorm(planes * block.expansion),
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
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.glob_pool(x)
        return self.final(x)



class MinkUNetBase(ResNetBase):
    BLOCK = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
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

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return self.final(out)


class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

















class GaussianDiffusion(object):
    """Gaussian diffusion utility.

    Args:
        beta_start: Start value of the scheduled variance
        beta_end: End value of the scheduled variance
        timesteps: Number of time steps in the forward process
    """

    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
        device=None,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device
        assert(device is not None)

        # Define the linear variance schedule
        self.betas = betas = np.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=np.float64,  # Using float64 for better precision
        )
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = torch.from_numpy(betas).float().to(device)
        self.alphas_cumprod = torch.from_numpy(alphas_cumprod).float().to(device)

        self.alphas_cumprod_prev = torch.from_numpy(alphas_cumprod_prev).float().to(device)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.from_numpy(np.sqrt(alphas_cumprod)).float().to(device)

        self.sqrt_one_minus_alphas_cumprod = torch.from_numpy(
            np.sqrt(1.0 - alphas_cumprod)
        ).float().to(device)

        self.log_one_minus_alphas_cumprod = torch.from_numpy(
            np.log(1.0 - alphas_cumprod)
        ).float().to(device)

        self.sqrt_recip_alphas_cumprod = torch.from_numpy(
            np.sqrt(1.0 / alphas_cumprod)
        ).float().to(device)

        self.sqrt_recipm1_alphas_cumprod = torch.from_numpy(
            np.sqrt(1.0 / alphas_cumprod - 1)
        ).float().to(device)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = torch.from_numpy(posterior_variance).float().to(device)

        # Log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.from_numpy(
            np.log(np.maximum(posterior_variance, 1e-20))
        ).float().to(device)

        self.posterior_mean_coef1 = torch.from_numpy(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        ).float().to(device)

        self.posterior_mean_coef2 = torch.from_numpy(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        ).float().to(device)

        self.is_nonzero_time = torch.from_numpy(
            np.arange(self.timesteps) > 0,
        ).float().to(device)

    def _extract(self, a, t, x):
        """Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

        Args:
            a: Tensor to extract from
            t: Timestep for which the coefficients are to be extracted
            x: Current batched samples
        """
        return ME.SparseTensor(
            features=a[t.long()][x.C[:, 0].long()][:, None],
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key,
        )

    # def q_mean_variance(self, x_start, t):
    #     """Extracts the mean, and the variance at current timestep.

    #     Args:
    #         x_start: Initial sample (before the first diffusion step)
    #         t: Current timestep
    #     """
    #     x_start_shape = tf.shape(x_start)
    #     mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
    #     variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
    #     log_variance = self._extract(
    #         self.log_one_minus_alphas_cumprod, t, x_start_shape
    #     )
    #     return mean, variance, log_variance

    def q_sample(self, x_start, t, noise):
        """Diffuse the data.

        Args:
            x_start: Embedding
            t: Current timestep (length of batch size)
            noise: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """
        assert(x_start.coordinate_map_key == noise.coordinate_map_key)
        coeff_x_start = self._extract(self.sqrt_alphas_cumprod, t, x_start)
        coeff_noise = self._extract(self.sqrt_alphas_cumprod, t, noise)
        return coeff_x_start * x_start + coeff_noise * noise

    def q_sample_list(self, x_start_list, t, noise_list):
        """Diffuse the data.

        Args:
            x_start_list: List of embeddings
            t: Current timestep (length of batch size)
            noise_list: Gaussian noise to be added at the current timestep
        Returns:
            Diffused samples at timestep `t`
        """
        return [self.q_sample(x_start, t, noise) for x_start, noise in zip(x_start_list, noise_list)]








    def predict_start_from_noise(self, x_t, t, noise):
        assert(x_t.coordinate_map_key == noise.coordinate_map_key)
        coeff_x_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t)
        coeff_noise = self._extract(self.sqrt_recipm1_alphas_cumprod, t, noise)
        return coeff_x_t * x_t - coeff_noise * noise
    
    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion
        posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """
        assert(x_start.coordinate_map_key == x_t.coordinate_map_key)

        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_start) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = ME.SparseTensor(
                features=torch.clamp(x_recon.F, self.clip_min, self.clip_max),
                coordinate_manager=x.coordinate_manager,
                coordinate_map_key=x.coordinate_map_key,
            )

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Sample from the diffuison model.

        Args:
            pred_noise: Noise predicted by the diffusion model
            x: Samples at a given timestep for which the noise was predicted
            t: Current timestep
            clip_denoised (bool): Whether to clip the predicted noise
                within the specified range or not.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = ME.SparseTensor(
            features=torch.randn_like(x.F),
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key,
        )
        # No noise when t == 0
        nonzero_mask = self._extract(self.is_nonzero_time, t, x)
        var = ME.SparseTensor(
            features=torch.exp(0.5 * model_log_variance.F),
            coordinate_manager=model_log_variance.coordinate_manager,
            coordinate_map_key=model_log_variance.coordinate_map_key,
        )
        return model_mean + nonzero_mask * var * noise

    def p_sample_list(self, pred_noise_list, x_list, t, clip_denoised=True):
        return [self.p_sample(pred_noise, x, t, clip_denoised) for pred_noise, x in zip(pred_noise_list, x_list)]
    
    # For reconstruction loss
    def predict_start(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = ME.SparseTensor(
                features=torch.clamp(x_recon.F, self.clip_min, self.clip_max),
                coordinate_manager=x.coordinate_manager,
                coordinate_map_key=x.coordinate_map_key,
            )
        return x_recon
    
    def predict_start_list(self, pred_noise_list, x_list, t, clip_denoised=True):
        return [self.predict_start(pred_noise, x, t, clip_denoised) for pred_noise, x in zip(pred_noise_list, x_list)]




class DiffusionModel(nn.Module):
    def __init__(self, network, ema_network, gdf_util, batch_size, optimizer, ema=0.999):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.gdf_util = gdf_util
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.ema = ema
        self.recon_loss = False

    def loss(self, noise_list, pred_noise_list):
        loss = 0
        num = 0
        for pred_noise, noise in zip(pred_noise_list, noise_list):
            n = np.sqrt(noise.F.shape[0])
            loss += F.smooth_l1_loss(pred_noise.F, noise.F) * n
            num += n
        return loss / num

    def train_step(self, embedding_list):
        self.optimizer.zero_grad()

        # 1. Get the batch size
        batch_size = self.batch_size

        # 2. Sample timesteps uniformly
        t = torch.randint(self.gdf_util.timesteps, size=(batch_size,)).to(embedding_list[0].F.device).long()

        # 3. Sample random noise to be added to the images in the batch
        noise_list = [
            ME.SparseTensor(
                features=torch.randn_like(embedding.F),
                coordinate_manager=embedding.coordinate_manager,
                coordinate_map_key=embedding.coordinate_map_key,
            ) for embedding in embedding_list
        ]

        # 4. Diffuse the images with noise
        x_t_list = self.gdf_util.q_sample_list(embedding_list, t, noise_list)

        # 5. Pass the diffused images and time steps to the network
        pred_noise_list = self.network(x_t_list, t)

        # 6. Calculate the loss
        loss = self.loss(noise_list, pred_noise_list)
        recon_loss = loss * 0
        if self.recon_loss:
            x_recon_list = self.gdf_util.predict_start_list(pred_noise_list, x_t_list, t)
            recon_loss += self.loss(embedding_list, x_recon_list)
            loss += recon_loss

        # 7. Get the gradients
        loss.backward()

        # 8. Update the weights of the network
        self.optimizer.step()

        # 9. Updates the weight values for the network with EMA weights
        state_dict = self.network.state_dict()
        ema_state_dict = self.ema_network.state_dict()
        for key in ema_state_dict:
            ema_state_dict[key] = self.ema * ema_state_dict[key] + (1 - self.ema) * state_dict[key]
        self.ema_network.load_state_dict(ema_state_dict)

        # 10. Return loss values
        return {"loss": loss, "recon_loss": recon_loss}
    
    def train_step_1stlayer(self, embedding_list):
        self.optimizer.zero_grad()

        # 1. Get the batch size
        batch_size = self.batch_size

        # 2. Sample timesteps uniformly
        t = torch.randint(self.gdf_util.timesteps, size=(batch_size,)).to(embedding_list[0].F.device).long()

        # 3. Sample random noise to be added to the images in the batch
        noise_list = [
            ME.SparseTensor(
                features=torch.randn_like(embedding.F),
                coordinate_manager=embedding.coordinate_manager,
                coordinate_map_key=embedding.coordinate_map_key,
            ) for embedding in embedding_list[:1]
        ]

        # 4. Diffuse the images with noise
        x_t_list = [self.gdf_util.q_sample(embedding_list[0], t, noise_list[0])]
        x_t_list.extend([
            ME.SparseTensor(
                features=embedding.F * 0,
                coordinate_manager=embedding.coordinate_manager,
                coordinate_map_key=embedding.coordinate_map_key,
            ) for embedding in embedding_list[1:]
        ])

        # 5. Pass the diffused images and time steps to the network
        pred_noise_list = self.network(x_t_list, t)

        # 6. Calculate the loss
        loss = self.loss(noise_list[:1], pred_noise_list[:1])
        recon_loss = loss * 0
        if self.recon_loss:
            x_recon_list = self.gdf_util.predict_start_list(pred_noise_list, x_t_list, t)
            recon_loss += self.loss(embedding_list, x_recon_list)
            loss += recon_loss

        # 7. Get the gradients
        loss.backward()

        # 8. Update the weights of the network
        self.optimizer.step()

        # 9. Updates the weight values for the network with EMA weights
        state_dict = self.network.state_dict()
        ema_state_dict = self.ema_network.state_dict()
        for key in ema_state_dict:
            ema_state_dict[key] = self.ema * ema_state_dict[key] + (1 - self.ema) * state_dict[key]
        self.ema_network.load_state_dict(ema_state_dict)

        # 10. Return loss values
        return {"loss": loss, "recon_loss": recon_loss}
    

    def test_step(self, embedding_list, t=None):
        with torch.no_grad():
            # 1. Get the batch size
            batch_size = self.batch_size

            # 2. Sample timesteps uniformly
            if t is None:
                t = torch.randint(self.gdf_util.timesteps, size=(batch_size,)).to(embedding_list[0].F.device).long()

            # 3. Sample random noise to be added to the images in the batch
            noise_list = [
                ME.SparseTensor(
                    features=torch.randn_like(embedding.F),
                    coordinate_manager=embedding.coordinate_manager,
                    coordinate_map_key=embedding.coordinate_map_key,
                ) for embedding in embedding_list
            ]

            # 4. Diffuse the images with noise
            x_t_list = self.gdf_util.q_sample_list(embedding_list, t, noise_list)

            # 5. Pass the diffused images and time steps to the network
            pred_noise_list = self.network(x_t_list, t)

            # 6. Calculate the loss
            loss = self.loss(noise_list, pred_noise_list)
            recon_loss = loss * 0
            if self.recon_loss:
                x_recon_list = self.predict_start_list(pred_noise_list, x_t_list, t)
                recon_loss += self.loss(embedding_list, x_recon_list)
                loss += recon_loss

        # 7. Return loss values
        return {"loss": loss, "recon_loss": recon_loss}


    def generate_embeddings_1stlayer(self, embedding_list, clip_denoised=True, every_n_times=None):
        # 1. Randomly sample noise (starting point for reverse process)
        sample_list = [
            ME.SparseTensor(
                features=torch.randn_like(embedding.F) * int(idx == 0),
                coordinate_manager=embedding.coordinate_manager,
                coordinate_map_key=embedding.coordinate_map_key,
            ) for idx, embedding in enumerate(embedding_list)
        ]
        # 2. Sample from the model iteratively
        with torch.no_grad():
            for t in reversed(range(0, self.gdf_util.timesteps)):
                tt = torch.Tensor([t] * self.batch_size).to(embedding_list[0].F.device).long()
                pred_noise_list = self.ema_network(sample_list, tt)
                # pred_noise_list = self.network(sample_list, tt)
                sample_list[0] = self.gdf_util.p_sample(
                    pred_noise_list[0], sample_list[0], tt, clip_denoised=clip_denoised
                )
                if every_n_times is not None and ((t % every_n_times == 0) or (t == 1)):
                    yield sample_list, t
        
        # 3. Return generated samples
        if every_n_times is None:  
            return sample_list

    
    def generate_embeddings(self, embedding_list, clip_denoised=True, every_n_times=None):
        # 1. Randomly sample noise (starting point for reverse process)
        sample_list = [
            ME.SparseTensor(
                features=torch.randn_like(embedding.F),
                coordinate_manager=embedding.coordinate_manager,
                coordinate_map_key=embedding.coordinate_map_key,
            ) for embedding in embedding_list
        ]
        # 2. Sample from the model iteratively
        with torch.no_grad():
            for t in reversed(range(0, self.gdf_util.timesteps)):
                tt = torch.Tensor([t] * self.batch_size).to(embedding_list[0].F.device).long()
                # pred_noise_list = self.ema_network(sample_list, tt)
                pred_noise_list = self.network(sample_list, tt)
                sample_list = self.gdf_util.p_sample_list(
                    pred_noise_list, sample_list, tt, clip_denoised=clip_denoised
                )
                if every_n_times is not None and ((t % every_n_times == 0) or (t == 1)):
                    yield sample_list, t
        
        # 3. Return generated samples
        if every_n_times is None:  
            return sample_list

    # def plot_images(
    #     self, epoch=None, logs=None, num_rows=2, num_cols=8, figsize=(12, 5)
    # ):
    #     """Utility to plot images using the diffusion model during training."""
    #     generated_samples = self.generate_images(num_images=num_rows * num_cols)
    #     generated_samples = (
    #         tf.clip_by_value(generated_samples * 127.5 + 127.5, 0.0, 255.0)
    #         .numpy()
    #         .astype(np.uint8)
    #     )

    #     _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    #     for i, image in enumerate(generated_samples):
    #         if num_rows == 1:
    #             ax[i].imshow(image)
    #             ax[i].axis("off")
    #         else:
    #             ax[i // num_cols, i % num_cols].imshow(image)
    #             ax[i // num_cols, i % num_cols].axis("off")

    #     plt.tight_layout()
    #     plt.show()

















class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, max_time=1000):
        super().__init__()
        self.dim = dim
        self.max_time = max_time
        self.embeddings = self.get_embeddings()
        # from PIL import Image
        # Image.fromarray(((self.embeddings.cpu().numpy() / 2.0 + 0.5) * 255.0).astype(np.uint8)).save('a.png')
        # input()

    def get_embeddings(self):
        import math
        half_dim = self.dim // 2
        embeddings = math.log(1000) / half_dim
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = torch.arange(self.max_time)[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # print(embeddings[0])
        return embeddings
    
    def forward(self, time):
        # print(time)
        # print(self.embeddings.shape)
        # input()
        return self.embeddings[time.long()]


class MinkUNetMultiResolutionDiffusion(MinkUNetBase):
    BLOCK = Bottleneck
    PLANES = (16, 32, 64, 128, 256, 256, 128, 64, 32)
    LAYERS = (1, 2, 3, 4, 6, 2, 2, 2, 2)

    def __init__(self, emb_channels, time_channels, D=3, has_mid_unet=False):
        self.has_mid_unet = has_mid_unet
        ResNetBase.__init__(self, emb_channels, time_channels, D)
        assert(len(self.PLANES) == 9)
        assert(len(self.LAYERS) == 9)
    
    def cat_time(self, x, time_embeddings):
        return ME.SparseTensor(
            features=torch.cat([x.F, time_embeddings[x.C[:, 0].long()]], dim=-1),
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key,
        )

    def network_initialization(self, emb_channels, time_channels, D):

        self.pos_emb = SinusoidalPositionEmbeddings(time_channels)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )

        # Activation
        self.relu = ME.MinkowskiReLU(inplace=True)

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM

        def init_conv():
            return nn.Sequential(
                ME.MinkowskiConvolution(
                    emb_channels, self.inplanes, kernel_size=5, dimension=D),
                ME.MinkowskiBatchNorm(self.inplanes),
                ME.MinkowskiReLU(inplace=True),
            )

        # Input layers
        self.init_conv1 = init_conv()
        self.init_conv2 = init_conv()
        self.init_conv3 = init_conv()
        self.init_conv4 = init_conv()

        # Encoder / downsamples
        self.inplanes += time_channels
        self.block0 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.inplanes += (time_channels + self.INIT_DIM)
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.inplanes += (time_channels + self.INIT_DIM)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)

        self.inplanes += (time_channels + self.INIT_DIM)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        # Mid
        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.inplanes += time_channels
        if self.has_mid_unet:
            self.mid_unet_inplanes = self.inplanes
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        
        if self.has_mid_unet: # overide
            self.block4 = MinkUNet14A(self.mid_unet_inplanes, self.inplanes)

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[5])


        # Decoder / upsamples
        self.inplanes = self.PLANES[5] + self.PLANES[3] * self.BLOCK.expansion + time_channels
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[2] * self.BLOCK.expansion + time_channels
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.PLANES[1] * self.BLOCK.expansion + time_channels
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[8], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[8])

        self.inplanes = self.PLANES[8] + self.PLANES[0] * self.BLOCK.expansion + time_channels
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[8],
                                       self.LAYERS[8])

        # Output layers
        self.out_conv8 = ME.MinkowskiConvolution(
            self.PLANES[8] * self.BLOCK.expansion, emb_channels,
            kernel_size=1, bias=True, dimension=D,
        )

        self.out_conv7 = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion, emb_channels,
            kernel_size=1, bias=True, dimension=D,
        )

        self.out_conv6 = ME.MinkowskiConvolution(
            self.PLANES[6] * self.BLOCK.expansion, emb_channels,
            kernel_size=1, bias=True, dimension=D,
        )

        self.out_conv5 = ME.MinkowskiConvolution(
            self.PLANES[5] * self.BLOCK.expansion, emb_channels,
            kernel_size=1, bias=True, dimension=D,
        )


    def forward(self, xs, t):

        assert(len(xs) == 4)
        x1, x2, x3, x4 = xs
        time_emb = self.time_mlp(self.pos_emb(t))

        x1 = self.cat_time(self.init_conv1(x1), time_emb)
        x2 = self.cat_time(self.init_conv2(x2), time_emb)
        x3 = self.cat_time(self.init_conv3(x3), time_emb)
        x4 = self.cat_time(self.init_conv4(x4), time_emb)

        out_b0p1 = self.block0(x1)

        out = self.conv1p1s2(out_b0p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(ME.cat(out, x2))

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(ME.cat(out, x3))

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(ME.cat(out, x4))

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(self.cat_time(out, time_emb))

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out_b5 = self.block5(self.cat_time(out, time_emb))

        # tensor_stride=4
        out = self.convtr5p8s2(out_b5)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out_b6 = self.block6(self.cat_time(out, time_emb))

        # tensor_stride=2
        out = self.convtr6p4s2(out_b6)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out_b7 = self.block7(self.cat_time(out, time_emb))

        # tensor_stride=1
        out = self.convtr7p2s2(out_b7)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_b0p1)
        out_b8 = self.block8(self.cat_time(out, time_emb))

        y1 = self.out_conv8(out_b8)
        y2 = self.out_conv7(out_b7)
        y3 = self.out_conv6(out_b6)
        y4 = self.out_conv5(out_b5)

        return y1, y2, y3, y4





def get_data(filenames, normalize=False):
    features = [[], [], [], []]
    coordinates = [[], [], [], []]

    for batch, filename in enumerate(filenames):
        d = torch.load(filename)
        idx = 0
        for fts, coords in zip(d['features'], d['coordinates']): # levels
            coords[:, 0] *= 0
            coords[:, 0] += batch
            coordinates[idx].append(coords)
            features[idx].append(fts)
            idx += 1

    embs = []
    mus = []
    sigmas = []
    stride = 8
    # import matplotlib; matplotlib.use("agg")
    # import matplotlib.pyplot as plt
    for fts, coords in zip(features, coordinates): # levels from low to high
        all_fts = torch.cat(fts, dim=0)
        # a, b = np.histogram(all_fts.cpu().numpy().flatten(), bins=200)
        # plt.plot(b[:200], a[:200]/a.sum(), label=f"stride{stride}")
        
        if normalize:
            mus.append(all_fts.mean(dim=0))
            sigmas.append(all_fts.std(dim=0))
        else:
            mus.append(all_fts.mean(dim=0) * 0)
            sigmas.append(all_fts.std(dim=0) * 0 + 3)
        embs.append(
            ME.SparseTensor(
                (all_fts - mus[-1]) / sigmas[-1], coordinates=torch.cat(coords, dim=0), device=fts[0].device, tensor_stride=stride,
                coordinate_manager=embs[0].coordinate_manager if len(embs) else None,
            )
        )
        # print(embs[-1].F.min(dim=0))
        # print(embs[-1].F.max(dim=0))
        # print(embs[-1].F.shape)
        # input()
        stride //= 2
    # plt.legend()
    # plt.savefig("aaaaa.png")
    # input()

    return embs[::-1], mus[::-1], sigmas[::-1] # levels from high to low


if __name__ == '__main__':

    batch_size = 1
    overfit5 = False
    feature_gau = False
    recon_loss = False
    test_idx = np.array([10, 20, 34, 50, 65]).astype(np.int32) - 1 + 1520
    # test_idx = np.array([1, 9, 17, 25, 33]).astype(np.int32)

    if overfit5:
        log_name = "log_overfit5"
        ckpt_name = "ckpt_overfit5"
        idx_list = np.array([10, 20, 34, 50, 65]).astype(np.int32) - 1
        pred_postfix = "_overfit5"
    else:
        log_name = "log_1stlayer"#"log"#
        ckpt_name = "ckpt_1stlayer"#"ckpt"#
        idx_list = np.arange(1520).astype(np.int32)
        pred_postfix = "1stlayer"
    
    if recon_loss:
        log_name += "_reconloss"
        ckpt_name += "_reconloss"
        pred_postfix += "reconloss"

    if feature_gau:
        clip_range = 1.0
        normalize = False
        folder = "embeddings+minkunet14_gau_20"
    else:
        clip_range = 3.0
        normalize = True
        folder = "embeddings+minkunet14_20"

    
    log_name = "log_overfitoverfit"
    ckpt_name = "ckpt_overfitoverfit"
    pred_postfix = ""
    clip_range = 1.0
    normalize = False
    folder = "embeddings+overfitoverfit"



    has_mid_unet = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MinkUNetMultiResolutionDiffusion(emb_channels=3, time_channels=32, D=3, has_mid_unet=has_mid_unet).to(device)
    net.pos_emb.embeddings = net.pos_emb.embeddings.to(device)
    ema_net = MinkUNetMultiResolutionDiffusion(emb_channels=3, time_channels=32, D=3, has_mid_unet=has_mid_unet).to(device)
    ema_net.pos_emb.embeddings = net.pos_emb.embeddings.to(device)

    gdf_util = GaussianDiffusion(
        beta_start=0.0015,
        beta_end=0.05,
        timesteps=1000,
        clip_min=-clip_range,
        clip_max=clip_range,
        device=device,
    )

    optimizer = Adam(net.parameters(), lr=2e-4)

    diff_model = DiffusionModel(net, ema_net, gdf_util, batch_size, optimizer)
    diff_model.recon_loss = recon_loss

    if False: # See loss by time
        batch_size = 5
        for ttt in range(0, 1000, 50):
            diff_model.network.load_state_dict(torch.load(f"{folder}/{ckpt_name}.pt"))
            diff_model.ema_network.load_state_dict(torch.load(f"{folder}/{ckpt_name}_ema.pt"))
            # idx = np.random.randint(80, size=batch_size)
            t = torch.Tensor([ttt] * batch_size).to(device).long()
            filename = f"{folder}/" + "%d.pt"
            embs, mus, sigmas = get_data([(filename % i) for i in test_idx], normalize=normalize)
            loss = diff_model.test_step(embs, t)["loss"]
            print(ttt, loss.item())
        quit()

    
    if False:
        diff_model.batch_size = 5
        diff_model.network.load_state_dict(torch.load('embeddings+holicity+fineres+fieldnet/overfit0/ckpt.pt'))
        diff_model.ema_network.load_state_dict(torch.load('embeddings+holicity+fineres+fieldnet/overfit0/ckpt_ema.pt'))
        filename = "embeddings+holicity+fineres+fieldnet/%d.pt"
        embs, mus, sigmas = get_data([(filename % i) for i in range(5)])
        gen_embs, gen_embs_by_times = diff_model.generate_embeddings(embs, clip_denoised=True, every_n_times=100)

        time = 900
        for gen_embs in gen_embs_by_times:
            for gen_emb, mu, sigma in zip(gen_embs, mus, sigmas):
                emb = gen_emb * sigma + mu
                d = {"features": [], "coordinates": []}
                d["features"].append(emb.F)
                d["coordinates"].append(emb.C)
            torch.save(d, f"embeddings+holicity+fineres+fieldnet/overfit0/pred_0_clip8_{time}.pt")
            time -= 100
    
    if True:
        epoch = 114999
        diff_model.batch_size = 1
        diff_model.network.load_state_dict(torch.load(f'{folder}/{ckpt_name}_{epoch}.pt'))
        diff_model.ema_network.load_state_dict(torch.load(f'{folder}/{ckpt_name}_{epoch}_ema.pt'))
        filename = f"{folder}/" + "%d.pt"

        # embs, mus, sigmas = get_data([(filename % i) for i in test_idx], normalize=normalize)
        # gen_embs_by_times = diff_model.generate_embeddings(embs, clip_denoised=True, every_n_times=100)
        data_dict = torch.load("/home/lzq/lzy/denoising-diffusion-pytorch/test_point_cloud_scale_20.pt")
        embs = [ME.SparseTensor(
            features=data_dict["features"].cuda(),
            coordinates=data_dict["coordinates"].cuda(),
        )]
        pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
        for _ in range(3):
            embs.append(pool(embs[-1]))
        gen_embs_by_times = diff_model.generate_embeddings_1stlayer(embs, clip_denoised=True, every_n_times=100)
        with open(f"{folder}/pc_gt.txt", "w") as f:
            for (x, y, z), (r, g, b) in zip(
                embs[0].C[:, 1:].cpu().numpy(),
                (embs[0].F * 127.5 + 127.5).cpu().numpy().astype(np.uint8),
            ):
                f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        quit()
        while True:
            gen_embs, time = next(gen_embs_by_times, (None, None))
            if gen_embs is None:
                break
            
            with open(f"{folder}/pc_{time}.txt", "w") as f:
                for (x, y, z), (r, g, b) in zip(
                    gen_embs[0].C[:, 1:].cpu().numpy(),
                    (gen_embs[0].F * 127.5 + 127.5).cpu().numpy().astype(np.uint8),
                ):
                    f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
            
        # while True:
        #     gen_embs, time = next(gen_embs_by_times, (None, None))
        #     if gen_embs is None:
        #         break
        #     ds = [{"features": [], "coordinates": []} for _ in range(diff_model.batch_size)]
        #     for gen_emb, mu, sigma in zip(gen_embs, mus, sigmas): # each level
        #         for batch in range(diff_model.batch_size):
        #             coords = gen_emb.coordinates_at(batch)
        #             coords = torch.cat([torch.zeros(coords.shape[0], 1).int().to(coords.device), coords], dim=1)
        #             ft = torch.clamp(gen_emb.features_at(batch), gdf_util.clip_min, gdf_util.clip_max) * sigma + mu
        #             # ft = gen_emb.features_at(batch) * sigma + mu
        #             ds[batch]["features"].append(ft)
        #             ds[batch]["coordinates"].append(coords)
        #     for batch in range(diff_model.batch_size):
        #         torch.save(ds[batch], f"{folder}/pred_{batch}_{time}{pred_postfix}.pt")

        quit()

    # diff_model.network.load_state_dict(torch.load('embeddings+holicity+fineres+fieldnet/ckpt.pt'))
    # diff_model.ema_network.load_state_dict(torch.load('embeddings+holicity+fineres+fieldnet/ckpt_ema.pt'))
    data_dict = torch.load("/home/lzq/lzy/denoising-diffusion-pytorch/test_point_cloud_scale_20.pt")
    embs = [ME.SparseTensor(
        features=data_dict["features"].cuda(),
        coordinates=data_dict["coordinates"].cuda(),
    )]
    pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
    for _ in range(3):
        embs.append(pool(embs[-1]))


    with open(f"{folder}/{log_name}.txt", "w") as f:

        for epoch in range(500000):

            idx = np.random.choice(idx_list, batch_size)
            t = torch.from_numpy(np.random.randint(gdf_util.timesteps, size=batch_size)).to(device).long()
            filename = f"{folder}/" + "%d.pt"
            # embs, mus, sigmas = get_data([(filename % i) for i in idx.tolist()], normalize=normalize)

            loss = diff_model.train_step_1stlayer(embs)#train_step
            recon_loss = loss["recon_loss"].item()
            loss = loss["loss"].item()
            
            print(f"{epoch}\t{loss}\t{recon_loss}")
            f.write(f"{epoch}\t{loss}\t{recon_loss}\n")
            f.flush()

            if epoch % 1000 == 999:
                torch.save(diff_model.network.state_dict(), f'/home/lzq/lzy/torch-ngp/{folder}/{ckpt_name}_{epoch}.pt')
                torch.save(diff_model.ema_network.state_dict(), f'/home/lzq/lzy/torch-ngp/{folder}/{ckpt_name}_{epoch}_ema.pt')
