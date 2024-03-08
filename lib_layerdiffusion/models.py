import torch.nn as nn
import torch
import cv2
import numpy as np

from tqdm import tqdm
from typing import Optional, Tuple
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
import ldm_patched.modules.model_management as model_management
from ldm_patched.modules.model_patcher import ModelPatcher


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class LatentTransparencyOffsetEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            zero_module(torch.nn.Conv2d(256, 4, kernel_size=3, padding=1, stride=1)),
        )

    def __call__(self, x):
        return self.blocks(x)


# 1024 * 1024 * 3 -> 16 * 16 * 512 -> 1024 * 1024 * 3
class UNet1024(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (32, 32, 64, 128, 256, 512, 512),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 4,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        self.latent_conv_in = zero_module(nn.Conv2d(4, block_out_channels[2], kernel_size=1))

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift="default",
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            resnet_groups=norm_num_groups,
            attn_groups=None,
            add_attention=True,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=None,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift="default",
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, latent):
        sample_latent = self.latent_conv_in(latent)
        sample = self.conv_in(x)
        emb = None

        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if i == 3:
                sample = sample + sample_latent

            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        sample = self.mid_block(sample, emb)

        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            sample = upsample_block(sample, res_samples, emb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2


class TransparentVAEDecoder:
    def __init__(self, sd, mod_number=1):
        self.load_device = model_management.get_torch_device()
        self.offload_device = model_management.unet_offload_device()
        self.dtype = torch.float16 if model_management.should_use_fp16(self.load_device) else torch.float32

        model = UNet1024(in_channels=3, out_channels=4)
        model.load_state_dict(sd, strict=True)
        model.to(device=self.offload_device, dtype=self.dtype)
        model.eval()

        self.model = ModelPatcher(model, load_device=self.load_device, offload_device=self.offload_device)
        self.mod_number = mod_number
        return

    @torch.no_grad()
    def estimate_single_pass(self, pixel, latent):
        y = self.model.model(pixel, latent)
        return y

    @torch.no_grad()
    def estimate_augmented(self, pixel, latent):
        args = [
            [False, 0], [False, 1], [False, 2], [False, 3], [True, 0], [True, 1], [True, 2], [True, 3],
        ]

        result = []

        for flip, rok in tqdm(args):
            feed_pixel = pixel.clone()
            feed_latent = latent.clone()

            if flip:
                feed_pixel = torch.flip(feed_pixel, dims=(3,))
                feed_latent = torch.flip(feed_latent, dims=(3,))

            feed_pixel = torch.rot90(feed_pixel, k=rok, dims=(2, 3))
            feed_latent = torch.rot90(feed_latent, k=rok, dims=(2, 3))

            eps = self.estimate_single_pass(feed_pixel, feed_latent).clip(0, 1)
            eps = torch.rot90(eps, k=-rok, dims=(2, 3))

            if flip:
                eps = torch.flip(eps, dims=(3,))

            result += [eps]

        result = torch.stack(result, dim=0)
        median = torch.median(result, dim=0).values
        return median

    def patch(self, p, vae_patcher, output_origin):
        @torch.no_grad()
        def wrapper(func, latent):
            pixel = func(latent).movedim(-1, 1).to(device=self.load_device, dtype=self.dtype)

            if output_origin:
                origin_outputs = (pixel.movedim(1, -1) * 255.0).detach().cpu().float().numpy().clip(0, 255).astype(np.uint8)
                for png in origin_outputs:
                    p.extra_result_images.append(png)

            latent = latent.to(device=self.load_device, dtype=self.dtype)
            model_management.load_model_gpu(self.model)
            vis_list = []

            for i in range(int(latent.shape[0])):
                if self.mod_number != 1 and i % self.mod_number != 0:
                    vis_list.append(pixel[i:i+1].movedim(1, -1))
                    continue

                y = self.estimate_augmented(pixel[i:i+1], latent[i:i+1])

                y = y.clip(0, 1).movedim(1, -1)
                alpha = y[..., :1]
                fg = y[..., 1:]

                B, H, W, C = fg.shape
                cb = checkerboard(shape=(H // 64, W // 64))
                cb = cv2.resize(cb, (W, H), interpolation=cv2.INTER_NEAREST)
                cb = (0.5 + (cb - 0.5) * 0.1)[None, ..., None]
                cb = torch.from_numpy(cb).to(fg)

                vis = fg * alpha + cb * (1 - alpha)
                vis_list.append(vis)

                png = torch.cat([fg, alpha], dim=3)[0]
                png = (png * 255.0).detach().cpu().float().numpy().clip(0, 255).astype(np.uint8)
                p.extra_result_images.append(png)

            vis_list = torch.cat(vis_list, dim=0)
            return vis_list

        vae_patcher.set_model_vae_decode_wrapper(wrapper)
        return


class TransparentVAEEncoder:
    def __init__(self, sd):
        self.load_device = model_management.get_torch_device()
        self.offload_device = model_management.unet_offload_device()
        self.dtype = torch.float16 if model_management.should_use_fp16(self.load_device) else torch.float32

        model = LatentTransparencyOffsetEncoder()
        model.load_state_dict(sd, strict=True)
        model.to(device=self.offload_device, dtype=self.dtype)
        model.eval()

        self.model = ModelPatcher(model, load_device=self.load_device, offload_device=self.offload_device)
        return

    def patch(self, p, vae_patcher):
        @torch.no_grad()
        def wrapper(func, latent):
            print('VAE Encode with Latent Transparency Offset is under construction now.')
            print('This should not be important if denoising strength is high.')
            print('But this will influence results with low denoising strength.')
            print('But results should be better if you come back (and update) several days later when we finish this part.')
            return func(latent)

        vae_patcher.set_model_vae_encode_wrapper(wrapper)
        return

