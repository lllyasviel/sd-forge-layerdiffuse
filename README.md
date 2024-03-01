# sd-forge-layerdiffusion

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/36598904-ae5f-4578-87d3-4b496e11dcc5)

This is a WIP extension for SD WebUI [(via Forge)](https://github.com/lllyasviel/stable-diffusion-webui-forge) to generate transparent images and layers.

The image generating and basic layer functionality is working now, but the transparent img2img is not finished yet (will finish in about one week).

This code base is highly dynamic and may change a lot in the next month. If you are from professional content creation studio and need all previous results to be strictly reproduced, you may consider backup files during each update.

# Before You Start

Because many people may be curious about how the latent preview looks like during a transparent diffusion process, I recorded a video so that you can see it before you download the models and extensions:

https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/e93b71d1-3560-48e2-a970-0b8efbfebb42

You can see that the native transparent diffusion can process transparent glass, semi-transparent glowing effects, etc, that are not possible with simple background removal methods. Native transparent diffusion also gives you detailed fur, hair, whiskers, and detailed structure like that skeleton.

# Model Notes

Note that all currently released models are for SDXL. Models for SD1.5 may be provided later if demanded.

**Note that in this extension, all model downloads/selections are fully automatic. In fact most users can just skip this section.**

Below models are released:

1. `layer_xl_transparent_attn.safetensors` This is a rank-256 LoRA to turn a SDXL into a transparent image generator. It will change the latent distribution of the model to a "transparent latent space" that can be decoded by the special VAE pipeline.
2. `layer_xl_transparent_conv.safetensors` This is an alternative model to turn your SDXL into a transparent image generator. This safetensors file includes an offset of all conv layers (and actually, all layers that are not q,k,v of any attention layers). These offsets can be merged to any XL model to change the latent distribution to transparent images. Because we excluded the offset training of any q,k,v layers, the prompt understanding of SDXL should be perfectly preserved. However, in practice, I find the `layer_xl_transparent_attn.safetensors` will lead to better results. This `layer_xl_transparent_conv.safetensors` is still included for some special use cases that needs special prompt understanding. Also, this model may introduce a strong style influence to the base model.
3. `layer_xl_fg2ble.safetensors` This is a safetensors file includes offsets to turn a SDXL into a layer generating model, that is conditioned on foregrounds, and generates blended compositions.
4. `layer_xl_fgble2bg.safetensors` This is a safetensors file includes offsets to turn a SDXL into a layer generating model, that is conditioned on foregrounds and blended compositions, and generates backgrounds.
5. `layer_xl_bg2ble.safetensors` This is a safetensors file includes offsets to turn a SDXL into a layer generating model, that is conditioned on backgrounds, and generates blended compositions.
6. `layer_xl_bgble2fg.safetensors` This is a safetensors file includes offsets to turn a SDXL into a layer generating model, that is conditioned on backgrounds and blended compositions, and generates foregrounds.
7. `vae_transparent_encoder.safetensors` This is an image encoder to extract a latent offset from pixel space. The offset can be added to latent images to help the diffusion of transparency. Note that in the paper we used a relatively heavy model with exactly same amount of parameters as the SD VAE. The released model is more light weighted, requires much less vram, and does not influence result quality in my tests.
8. `vae_transparent_decoder.safetensors` This is an image decoder that takes SD VAE outputs and latent image as inputs, and outputs a real PNG image. The model architecture is also more lightweight than the paper version to reduce VRAM requirement. I have made sure that the reduced parameters does not influence result quality.

Below models may be released soon (if necessary):

1. A model that can generate foreground and background together (using attention sharing similar to AnimateDiff). I put this model on hold because of these reasons: (1) the other released models can already achieve all functionalities and this model does not bring more functionalities. (2) the inference speed of this model is 3x slower than others and requires 4x more VRAM than other released model, and I am working on reducing the VRAM of this model if necessary. (3) This model will involve more hyperparameters and if demanded, I will investigate the best practice for inference/training before release it.
2. The current background-conditioned foreground model may be a bit too lightweight. I will probably release a heavier one with more parameters and different behaviors (see also the discussions later).
3. Because the difference between diffusers training and k-diffusion inference, I can observe some mystical problems like sometimes DPM++ will give artifacts but Euler A will fix it. I am looking into it and may provide some revised model that works better with all A1111 samplers.


# Sanity Check

We highly encourage you to go through the sanity check and get exactly same results (so that if any problem occurs, we will know if the problem is on our side).

The two used models are:

1. https://civitai.com/models/133005?modelVersionId=198530 Juggernaut XL V6 (note that the used one is **V6**, not v7 or v8 or V9)
2. https://civitai.com/models/261336?modelVersionId=295158 anima_pencil-XL 1.0.0 (note that the used one is **1.0.0**, not 1.5.0)
