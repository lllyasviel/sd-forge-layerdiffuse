import gradio as gr
import os
import functools
import torch
import numpy as np
import copy

from modules import scripts, script_callbacks, shared
from modules.processing import StableDiffusionProcessing
from lib_layerdiffusion.enums import ResizeMode
from lib_layerdiffusion.utils import rgba2rgbfp32, to255unit8, crop_and_resize_image
from enum import Enum
from modules.paths import models_path
from ldm_patched.modules.utils import load_torch_file
from lib_layerdiffusion.models import TransparentVAEDecoder, TransparentVAEEncoder
from ldm_patched.modules.model_management import current_loaded_models
from modules_forge.forge_sampler import sampling_prepare
from modules.modelloader import load_file_from_url


def is_model_loaded(model):
    return any(model == m.model for m in current_loaded_models)


layer_model_root = os.path.join(models_path, 'layer_model')
os.makedirs(layer_model_root, exist_ok=True)


vae_transparent_encoder = None
vae_transparent_decoder = None


class LayerMethod(Enum):
    FG_ONLY_ATTN = "Only Generate Transparent Image (Attention Injection)"
    FG_ONLY_CONV = "Only Generate Transparent Image (Conv Injection)"
    FG_TO_BLEND = "From Foreground to Blending"
    FG_BLEND_TO_BG = "From Foreground and Blending to Background"
    BG_TO_BLEND = "From Background to Blending"
    BG_BLEND_TO_FG = "From Background and Blending to Foreground"


@functools.lru_cache(maxsize=2)
def load_layer_model_state_dict(filename):
    return load_torch_file(filename, safe_load=True)


class LayerDiffusionForForge(scripts.Script):
    def title(self):
        return "LayerDiffusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enabled', value=False)
            method = gr.Dropdown(choices=[e.value for e in LayerMethod], value=LayerMethod.FG_ONLY_ATTN.value, label="Method", type='value')
            gr.HTML('</br>')  # some strange gradio problems

            with gr.Row():
                fg_image = gr.Image(label='Foreground', source='upload', image_mode='RGBA', visible=False)
                bg_image = gr.Image(label='Background', source='upload', image_mode='RGBA', visible=False)
                blend_image = gr.Image(label='Blending', source='upload', image_mode='RGBA', visible=False)

            with gr.Row():
                weight = gr.Slider(label=f"Weight", value=1.0, minimum=0.0, maximum=2.0, step=0.001)
                ending_step = gr.Slider(label="Stop At", value=1.0, minimum=0.0, maximum=1.0)

            resize_mode = gr.Radio(choices=[e.value for e in ResizeMode], value=ResizeMode.CROP_AND_RESIZE.value, label="Resize Mode", type='value', visible=False)
            output_origin = gr.Checkbox(label='Output original mat for img2img', value=False, visible=False)

        def method_changed(m):
            m = LayerMethod(m)

            if m == LayerMethod.FG_TO_BLEND:
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

            if m == LayerMethod.BG_TO_BLEND:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)

            if m == LayerMethod.BG_BLEND_TO_FG:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

            if m == LayerMethod.FG_BLEND_TO_BG:
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        method.change(method_changed, inputs=method, outputs=[fg_image, bg_image, blend_image, resize_mode], show_progress=False, queue=False)

        return enabled, method, weight, ending_step, fg_image, bg_image, blend_image, resize_mode, output_origin

    def process_before_every_sampling(self, p: StableDiffusionProcessing, *script_args, **kwargs):
        global vae_transparent_decoder, vae_transparent_encoder

        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        enabled, method, weight, ending_step, fg_image, bg_image, blend_image, resize_mode, output_origin = script_args

        if not enabled:
            return

        p.extra_generation_params.update(dict(
            layerdiffusion_enabled=enabled,
            layerdiffusion_method=method,
            layerdiffusion_weight=weight,
            layerdiffusion_ending_step=ending_step,
            layerdiffusion_fg_image=fg_image is not None,
            layerdiffusion_bg_image=bg_image is not None,
            layerdiffusion_blend_image=blend_image is not None,
            layerdiffusion_resize_mode=resize_mode,
        ))

        B, C, H, W = kwargs['noise'].shape  # latent_shape
        height = H * 8
        width = W * 8
        batch_size = p.batch_size

        method = LayerMethod(method)
        print(f'[Layer Diffusion] {method}')

        resize_mode = ResizeMode(resize_mode)
        fg_image = crop_and_resize_image(rgba2rgbfp32(fg_image), resize_mode, height, width) if fg_image is not None else None
        bg_image = crop_and_resize_image(rgba2rgbfp32(bg_image), resize_mode, height, width) if bg_image is not None else None
        blend_image = crop_and_resize_image(rgba2rgbfp32(blend_image), resize_mode, height, width) if blend_image is not None else None

        original_unet = p.sd_model.forge_objects.unet.clone()
        unet = p.sd_model.forge_objects.unet.clone()
        vae = p.sd_model.forge_objects.vae.clone()

        if method in [LayerMethod.FG_ONLY_ATTN, LayerMethod.FG_ONLY_CONV, LayerMethod.BG_BLEND_TO_FG]:
            if vae_transparent_decoder is None:
                model_path = load_file_from_url(
                    url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors',
                    model_dir=layer_model_root,
                    file_name='vae_transparent_decoder.safetensors'
                )
                vae_transparent_decoder = TransparentVAEDecoder(load_torch_file(model_path))
            vae_transparent_decoder.patch(p, vae.patcher, output_origin)

            if vae_transparent_encoder is None:
                model_path = load_file_from_url(
                    url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_encoder.safetensors',
                    model_dir=layer_model_root,
                    file_name='vae_transparent_encoder.safetensors'
                )
                vae_transparent_encoder = TransparentVAEEncoder(load_torch_file(model_path))
            vae_transparent_encoder.patch(p, vae.patcher)

        if fg_image is not None:
            fg_image = vae.encode(torch.from_numpy(np.ascontiguousarray(fg_image[None].copy())))
            fg_image = unet.model.latent_format.process_in(fg_image)

        if bg_image is not None:
            bg_image = vae.encode(torch.from_numpy(np.ascontiguousarray(bg_image[None].copy())))
            bg_image = unet.model.latent_format.process_in(bg_image)

        if blend_image is not None:
            blend_image = vae.encode(torch.from_numpy(np.ascontiguousarray(blend_image[None].copy())))
            blend_image = unet.model.latent_format.process_in(blend_image)

        if method == LayerMethod.FG_ONLY_ATTN:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_transparent_attn.safetensors'
            )
            layer_lora_model = load_layer_model_state_dict(model_path)
            unet.load_frozen_patcher(layer_lora_model, weight)

        if method == LayerMethod.FG_ONLY_CONV:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_conv.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_transparent_conv.safetensors'
            )
            layer_lora_model = load_layer_model_state_dict(model_path)
            unet.load_frozen_patcher(layer_lora_model, weight)

        if method == LayerMethod.BG_TO_BLEND:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bg2ble.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_bg2ble.safetensors'
            )
            unet.extra_concat_condition = bg_image
            layer_lora_model = load_layer_model_state_dict(model_path)
            unet.load_frozen_patcher(layer_lora_model, weight)

        if method == LayerMethod.FG_TO_BLEND:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fg2ble.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_fg2ble.safetensors'
            )
            unet.extra_concat_condition = fg_image
            layer_lora_model = load_layer_model_state_dict(model_path)
            unet.load_frozen_patcher(layer_lora_model, weight)

        if method == LayerMethod.BG_BLEND_TO_FG:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bgble2fg.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_bgble2fg.safetensors'
            )
            unet.extra_concat_condition = torch.cat([bg_image, blend_image], dim=1)
            layer_lora_model = load_layer_model_state_dict(model_path)
            unet.load_frozen_patcher(layer_lora_model, weight)

        if method == LayerMethod.FG_BLEND_TO_BG:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fgble2bg.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_fgble2bg.safetensors'
            )
            unet.extra_concat_condition = torch.cat([fg_image, blend_image], dim=1)
            layer_lora_model = load_layer_model_state_dict(model_path)
            unet.load_frozen_patcher(layer_lora_model, weight)

        sigma_end = unet.model.model_sampling.percent_to_sigma(ending_step)

        def remove_concat(cond):
            cond = copy.deepcopy(cond)
            for i in range(len(cond)):
                try:
                    del cond[i]['model_conds']['c_concat']
                except:
                    pass
            return cond

        def conditioning_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
            if timestep < sigma_end:
                if not is_model_loaded(original_unet):
                    sampling_prepare(original_unet, x)
                target_model = original_unet.model
                cond = remove_concat(cond)
                uncond = remove_concat(uncond)
            else:
                target_model = model

            return target_model, x, timestep, uncond, cond, cond_scale, model_options, seed

        unet.add_conditioning_modifier(conditioning_modifier)

        p.sd_model.forge_objects.unet = unet
        p.sd_model.forge_objects.vae = vae
        return

def on_ui_settings():
    section = ("Layer Diffusion", "Layer Diffusion")

    shared.opts.add_option(
        "layerdiffusion_save_transparent_images",
        shared.OptionInfo(False, "Save images with transparency", section=section),
    )

script_callbacks.on_ui_settings(on_ui_settings)
