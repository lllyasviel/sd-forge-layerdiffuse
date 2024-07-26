# sd-forge-layerdiffuse

Transparent Image Layer Diffusion using Latent Transparency

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/36598904-ae5f-4578-87d3-4b496e11dcc5)

This is a WIP extension for SD WebUI [(via Forge)](https://github.com/lllyasviel/stable-diffusion-webui-forge) to generate transparent images and layers.

# Updates

1. img2img is finished! See also [here](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/854)

# Before You Start

Because many people may be curious about how the latent preview looks like during a transparent diffusion process, I recorded a video so that you can see it before you download the models and extensions:

https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/e93b71d1-3560-48e2-a970-0b8efbfebb42

You can see that the native transparent diffusion can process transparent glass, semi-transparent glowing effects, etc, that are not possible with simple background removal methods. Native transparent diffusion also gives you detailed fur, hair, whiskers, and detailed structure like that skeleton.

# Model Notes

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
9. `layer_sd15_vae_transparent_encoder.safetensors` Same as above VAE encoder, but fine-tuned for SD1.5.
10. `layer_sd15_vae_transparent_decoder.safetensors` Same as above VAE decoder, but fine-tuned for SD1.5.
11. `layer_sd15_transparent_attn.safetensors` This is a rank-256 LoRA to turn a SD1.5 into a transparent image generator. It will change the latent distribution of the model to a "transparent latent space" that can be decoded by the special VAE pipeline.
12. `layer_sd15_joint.safetensors` This model file allows for generating all layers together with SD1.5. It includes two rank-256 loras (foreground lora and background lora), and an attention sharing module to share attention between multiple diffusion processes on par. Note that different from paper, this model file includes an additional "blended lora", and it actually can generate three images together (fg, bg, and blended image). Generating blended images together with fg and bg is helpful for structural understanding in our very recent tests.
13. `layer_sd15_fg2bg.safetensors` This model file allows for generating background from foreground with SD1.5. It includes a rank-256 lora and an attention sharing module to share attention between multiple diffusion processes on par. This model file includes an additional "blended lora", and it actually can generate two images together (bg and blended image). Generating blended images together with bg is helpful for structural understanding in our very recent tests. Besides, to save VRAM, the fg is directly feed into all attention layers as control signal, rather than creating another diffusion pass.
14. `layer_sd15_bg2fg.safetensors` This model file allows for generating foreground from background with SD1.5. It includes a rank-256 lora and an attention sharing module to share attention between multiple diffusion processes on par. This model file includes an additional "blended lora", and it actually can generate two images together (fg and blended image). Generating blended images together with fg is helpful for structural understanding in our very recent tests. Besides, to save VRAM, the bg is directly feed into all attention layers as control signal, rather than creating another diffusion pass.

Below models may be released soon (if necessary):

1. SDXL models that can generate foreground and background together and SDXL's one step conditional model. (Note that all joint models for SD1.5 are already released) I put this model on hold because of these reasons: (1) the other released models can already achieve all functionalities and this model does not bring more functionalities. (2) the inference speed of this model is 3x slower than others and requires 4x more VRAM than other released model, and I am working on reducing the VRAM of this model and speed up the inference. (3) This model will involve more hyperparameters and if demanded, I will investigate the best practice for inference/training before release it.
2. The current background-conditioned foreground model for SDXL may be a bit too lightweight. I will probably release a heavier one with more parameters and different behaviors (see also the discussions later).
3. Because the difference between diffusers training and k-diffusion inference, I can observe some mystical problems like sometimes DPM++ will give artifacts but Euler A will fix it. I am looking into it and may provide some revised model that works better with all A1111 samplers.
4. Two-step foreground and background conditional models for SD1.5. (Note that one-step conditional/joint models are already released.)

# Sanity Check

### SDXL

We highly encourage you to go through the sanity check and get exactly same results (so that if any problem occurs, we will know if the problem is on our side).

The two used models are:

1. https://civitai.com/models/133005?modelVersionId=198530 Juggernaut XL V6 (note that the used one is **V6**, not v7 or v8 or V9)
2. https://civitai.com/models/261336?modelVersionId=295158 anima_pencil-XL 1.0.0 (note that the used one is **1.0.0**, not 1.5.0)

We will first test transparent image generating. Set your extension to this:

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/5b85b383-89c0-403e-aa07-d6e43ff3b8ae)

an apple, high quality

Negative prompt: bad, ugly

Steps: 20, Sampler: DPM++ 2M SDE Karras, CFG scale: 5, Seed: 12345, Size: 1024x1024, Model hash: 1fe6c7ec54, Model: juggernautXL_version6Rundiffusion, layerdiffusion_enabled: True, layerdiffusion_method: Only Generate Transparent Image (Attention Injection), layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: False, layerdiffusion_bg_image: False, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, Version: f0.0.17v1.8.0rc-latest-269-gef35383b

Make sure that you get this apple

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/376fa8bc-547e-4cd7-b658-7d60f2e37f1d)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/16efc57b-4da8-4227-a257-f45f3dfeaddc)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/38ace070-6530-43c9-9ca1-c98aa5b7a0ed)

woman, messy hair, high quality

Negative prompt: bad, ugly

Steps: 20, Sampler: DPM++ 2M SDE Karras, CFG scale: 5, Seed: 12345, Size: 1024x1024, Model hash: 1fe6c7ec54, Model: juggernautXL_version6Rundiffusion, layerdiffusion_enabled: True, layerdiffusion_method: Only Generate Transparent Image (Attention Injection), layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: False, layerdiffusion_bg_image: False, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, Version: f0.0.17v1.8.0rc-latest-269-gef35383b

Make sure that you get the woman with hair as messy as this

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/17c86ba5-eb29-45d4-b708-caf7e836b509)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/6f1ef595-255c-4162-bdf9-c8e4eb321f31)

a cup made of glass, high quality

Negative prompt: bad, ugly

Steps: 20, Sampler: DPM++ 2M SDE Karras, CFG scale: 5, Seed: 12345, Size: 1024x1024, Model hash: 1fe6c7ec54, Model: juggernautXL_version6Rundiffusion, layerdiffusion_enabled: True, layerdiffusion_method: Only Generate Transparent Image (Attention Injection), layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: False, layerdiffusion_bg_image: False, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, Version: f0.0.17v1.8.0rc-latest-269-gef35383b

Make sure that you get this cup

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/a99177e6-72ed-447b-b2a5-6ca0fe1dc105)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/3b7df3f3-c6c1-401d-afa8-5a1c404165c9)

glowing effect, book of magic, high quality

Negative prompt: bad, ugly

Steps: 20, Sampler: DPM++ 2M SDE Karras, CFG scale: 7, Seed: 12345, Size: 1024x1024, Model hash: 1fe6c7ec54, Model: juggernautXL_version6Rundiffusion, layerdiffusion_enabled: True, layerdiffusion_method: Only Generate Transparent Image (Attention Injection), layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: True, layerdiffusion_bg_image: False, layerdiffusion_blend_image: True, layerdiffusion_resize_mode: Crop and Resize, Version: f0.0.17v1.8.0rc-latest-269-gef35383b

make sure that you get this glowing book

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/c093c862-17a3-4604-8e23-6c7f3a0eb4b3)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/fa0b02b0-b530-48ed-a8ec-17bd9cccfc87)

OK then lets move on to a bit longer prompt:

(this prompt is from https://civitai.com/images/3160575)

photograph close up portrait of Female boxer training, serious, stoic cinematic 4k epic detailed 4k epic detailed photograph shot on kodak detailed bokeh cinematic hbo dark moody

Negative prompt: (worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)

Steps: 20, Sampler: DPM++ 2M SDE Karras, CFG scale: 7, Seed: 12345, Size: 896x1152, Model hash: 1fe6c7ec54, Model: juggernautXL_version6Rundiffusion, layerdiffusion_enabled: True, layerdiffusion_method: Only Generate Transparent Image (Attention Injection), layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: False, layerdiffusion_bg_image: False, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, Version: f0.0.17v1.8.0rc-latest-269-gef35383b

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/845c0e35-0096-484b-be2c-d443b4dc63cd)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/47ee7ba1-7f64-4e27-857f-c82c9d2bbb14)

Anime model test:

girl in dress, high quality

Negative prompt: nsfw, bad, ugly, text, watermark

Steps: 20, Sampler: DPM++ 2M SDE Karras, CFG scale: 7, Seed: 12345, Size: 896x1152, Model hash: 7ed8da12d9, Model: animaPencilXL_v100, layerdiffusion_enabled: True, layerdiffusion_method: Only Generate Transparent Image (Attention Injection), layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: False, layerdiffusion_bg_image: False, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, Version: f0.0.17v1.8.0rc-latest-269-gef35383b

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/fcec8ea5-32de-44af-847a-d66dd62b95d1)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/53d84e56-4061-4d91-982f-8f1e927f68b7)

(I am not very good at writing prompts in the AnimagineXL format, and perhaps you can get better results with better prompts)

### SD1.5

The tested model is [realisticVisionV51_v51VAE](https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/realisticVisionV51_v51VAE.safetensors). We highly encourage you to go through the sanity check and get exactly same results (so that if any problem occurs, we will know if the problem is on our side).

an apple, 4k, high quality

Negative prompt: bad, ugly

Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 512x512, Model hash: 15012c538f, Model: realisticVisionV51_v51VAE, layerdiffusion_enabled: True, layerdiffusion_method: (SD1.5) Only Generate Transparent Image (Attention Injection), layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: False, layerdiffusion_bg_image: False, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, layerdiffusion_fg_additional_prompt: , layerdiffusion_bg_additional_prompt: , layerdiffusion_blend_additional_prompt: , Version: f0.0.17v1.8.0rc-latest-276-g29be1da7

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/d377d1b4-5c53-4bda-8639-5420ff7218ca)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/5e74c1ac-2940-47b3-b8c9-2e314c05c21e)

### Generating Foregrounds and Backgrounds Together (SD1.5)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/c4e9dab3-8038-4f45-a3c0-96e2ee0f46dc)

This will allow you to generate all layers together in one single diffusion process.

**Very important: Because this will generate 3 images together (the foreground, background, and blended image), your batchsize MUST be divided by 3. For example, you can use batch size 3 or 6 or 9 or 12 ... If you do not use batchsize number divided by 3, you will only get noise.**

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/4155bcd3-c33e-4a68-af14-087ac3df617a)

man walking, 4k, high quality

Negative prompt: bad, ugly

Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 512x640, Model hash: 15012c538f, Model: realisticVisionV51_v51VAE, layerdiffusion_enabled: True, layerdiffusion_method: (SD1.5) Generate Everything Together, layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: False, layerdiffusion_bg_image: False, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, layerdiffusion_fg_additional_prompt: , layerdiffusion_bg_additional_prompt: , layerdiffusion_blend_additional_prompt: , Version: f0.0.17v1.8.0rc-latest-276-g29be1da7

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/bc66ca16-6f7b-40ab-b7ea-265582913667)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/f5b45ffe-5275-47dd-9a77-2a6fac3a8b2d)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/e32a5a44-902b-4f2b-bdfa-2025bf07619a)

(Note that the third image is encoded/decoded by VAE and diffusion process so it may be different to the fg/bg. To get perfectly same fg/bg, you can blend the real bf and fg with any other software, or wait us to provide a simple UI for simple blending of some png elements.)

(this image is SD1.5 with very simple prompts and results can be much better with more prompt with SD15 quality tags, or with high-res fix coming soon)

**Independent prompts for layers**

In some cases, you may find that the background is corrupted by the global prompt. For example:

an apple on table, high quality, 4k

Negative prompt: nsfw, bad, ugly

Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 512x512, Model hash: 15012c538f, Model: realisticVisionV51_v51VAE, layerdiffusion_enabled: True, layerdiffusion_method: (SD1.5) Generate Everything Together, layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: False, layerdiffusion_bg_image: False, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, layerdiffusion_fg_additional_prompt: , layerdiffusion_bg_additional_prompt: , layerdiffusion_blend_additional_prompt: , Version: f0.0.17v1.8.0rc-latest-276-g29be1da7

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/5f094232-68d7-458a-86ac-d28c2da506b2)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/c878d960-f034-46fb-a5f9-50a66eb18164)

(We somewhat do not want the apples in the background and only want foreground apples)

Then you can first remove all content part in the prompt

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/a62c9d49-a9e7-46d8-a857-2c29d6d4628c)

and then write them for different layers, like this

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/6861aee7-ea59-4265-9256-6048e90b2f59)

Then you will get

high quality, 4k

Negative prompt: nsfw, bad, ugly

Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 512x512, Model hash: 15012c538f, Model: realisticVisionV51_v51VAE, layerdiffusion_enabled: True, layerdiffusion_method: (SD1.5) Generate Everything Together, layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: False, layerdiffusion_bg_image: False, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, layerdiffusion_fg_additional_prompt: apple, layerdiffusion_bg_additional_prompt: floor in room, layerdiffusion_blend_additional_prompt: apple on floor in room, Version: f0.0.17v1.8.0rc-latest-276-g29be1da7

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/1a7daa07-b5c1-482e-b85d-bc72860931d6)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/3fffcb75-0141-4617-b1f0-b6e242b8af9d)

Some more examples

high quality, 4k
Negative prompt: nsfw, bad, ugly
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 512x640, Model hash: 15012c538f, Model: realisticVisionV51_v51VAE, layerdiffusion_enabled: True, layerdiffusion_method: (SD1.5) Generate Everything Together, layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: False, layerdiffusion_bg_image: False, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, layerdiffusion_fg_additional_prompt: dog running, layerdiffusion_bg_additional_prompt: street, layerdiffusion_blend_additional_prompt: dog running in street, Version: f0.0.17v1.8.0rc-latest-276-g29be1da7

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/e38672df-1333-4ece-82eb-26f579b62131)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/e1eeb4c2-179e-45b7-aa08-55c7d89ed1ee)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/e713d8d6-c10e-4a50-b040-22e25a608c55)

high quality, 4k
Negative prompt: nsfw, bad, ugly
Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 512x640, Model hash: 15012c538f, Model: realisticVisionV51_v51VAE, layerdiffusion_enabled: True, layerdiffusion_method: (SD1.5) Generate Everything Together, layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: False, layerdiffusion_bg_image: False, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, layerdiffusion_fg_additional_prompt: a man sitting, layerdiffusion_bg_additional_prompt: chair, layerdiffusion_blend_additional_prompt: a man sitting on chair, Version: f0.0.17v1.8.0rc-latest-276-g29be1da7

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/42cc92a4-67d0-4f30-b51c-9068f70750a0)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/7d3869d9-cfc5-448d-9932-85d422857231)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/bc3e7894-d3cb-4136-886d-8b4c0df4be21)

### Background Condition (SD1.5, one step workflow)

First download this image:

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/e7e2d80e-ffbe-4724-812a-5139a88027e3)

In most cases, bg-to-fg does not need additional layer prompts. But you can add it if you wish

**Very important: Because this will generate 2 images together (the foreground and blended image), your batchsize MUST be divided by 2. For example, you can use batch size 2 or 4 or 6 or 8 ... If you do not use batchsize number divided by 2, you will only get noise.**

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/3fc7af63-6f87-40fb-b1d7-6a021668da41)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/29352421-d5ce-4442-86f5-b35aee1d67cb)

an old man sitting, high quality, 4k

Negative prompt: bad, ugly

Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 512x640, Model hash: 15012c538f, Model: realisticVisionV51_v51VAE, layerdiffusion_enabled: True, layerdiffusion_method: (SD1.5) From Background to Foreground, layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: False, layerdiffusion_bg_image: True, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, layerdiffusion_fg_additional_prompt: , layerdiffusion_bg_additional_prompt: , layerdiffusion_blend_additional_prompt: , Version: f0.0.17v1.8.0rc-latest-276-g29be1da7

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/dba41ac5-07da-4036-a043-3605faf9d2e1)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/903c62ec-0f8e-49ef-b58a-1c098b200e0c)

Note that the second image is a visualization that will have color differences. To get perfectly same fg/bg, you can blend the real bg and fg with any other software, or wait us to provide a simple UI for simple blending of some png elements.

For example this is a real blending using photopea

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/baa0342d-55cb-49ca-887c-ab42d1873eb0)

Another example

Input:

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/c4110feb-7e70-455b-88a9-04e868dab0de)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/7d2e1351-d0af-4c5b-87f0-ac6331b9172d)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/cb7c7c82-d03d-460c-a634-39d6d5248510)

Note that the second image is a visualization that will have color differences. To get perfectly same fg/bg, you can blend the real bg and fg with any other software, or wait us to provide a simple UI for simple blending of some png elements.

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/8bf5d6bd-fcd5-41d6-b0fc-2f6f0e58cbb7)

### Foreground Condition (SD1.5, one step workflow)

We first generate a cat

a cat running, high quality, 4k

Negative prompt: nsfw, bad, ugly

Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 512x640, Model hash: 15012c538f, Model: realisticVisionV51_v51VAE, layerdiffusion_enabled: True, layerdiffusion_method: (SD1.5) Only Generate Transparent Image (Attention Injection), layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: True, layerdiffusion_bg_image: True, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, layerdiffusion_fg_additional_prompt: , layerdiffusion_bg_additional_prompt: , layerdiffusion_blend_additional_prompt: , Version: f0.0.17v1.8.0rc-latest-276-g29be1da7

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/4d39ab96-94d6-48eb-bd45-709f9ef25ec2)

Then drag the real transparent foreground to UI

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/671ecf9c-b2d4-488e-bcf9-ac5a05c0b3bf)

**Very important: Because this will generate 2 images together (the foreground and blended image), your batchsize MUST be divided by 2. For example, you can use batch size 2 or 4 or 6 or 8 ... If you do not use batchsize number divided by 2, you will only get noise.**

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/3fc7af63-6f87-40fb-b1d7-6a021668da41)

street, high quality, 4k

Negative prompt: nsfw, bad, ugly

Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 512x640, Model hash: 15012c538f, Model: realisticVisionV51_v51VAE, layerdiffusion_enabled: True, layerdiffusion_method: (SD1.5) From Foreground to Background, layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: True, layerdiffusion_bg_image: True, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, layerdiffusion_fg_additional_prompt: , layerdiffusion_bg_additional_prompt: , layerdiffusion_blend_additional_prompt: , Version: f0.0.17v1.8.0rc-latest-276-g29be1da7

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/30f446f8-cf2b-4040-94e0-ef01eb7bd577)

### Some More Complicated Examples for SD1.5

Lets travel a bit more.

First we get a man singing

a man singing, high quality, 4k

Negative prompt: bad, ugly

Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 512x640, Model hash: 15012c538f, Model: realisticVisionV51_v51VAE, layerdiffusion_enabled: True, layerdiffusion_method: (SD1.5) Only Generate Transparent Image (Attention Injection), layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: True, layerdiffusion_bg_image: True, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, layerdiffusion_fg_additional_prompt: , layerdiffusion_bg_additional_prompt: , layerdiffusion_blend_additional_prompt: , Version: f0.0.17v1.8.0rc-latest-276-g29be1da7

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/1cb18b88-fca7-4e2c-b7e7-993e460ce65f)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/1bbaeb76-6d36-4eaa-8ca8-d778d8a3e952)

(then get a concert stage)

concert stage, high quality, 4k

Negative prompt: bad, ugly

Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 512x640, Model hash: 15012c538f, Model: realisticVisionV51_v51VAE, layerdiffusion_enabled: True, layerdiffusion_method: (SD1.5) From Foreground to Background, layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: True, layerdiffusion_bg_image: True, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, layerdiffusion_fg_additional_prompt: , layerdiffusion_bg_additional_prompt: , layerdiffusion_blend_additional_prompt: , Version: f0.0.17v1.8.0rc-latest-276-g29be1da7

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/13d8e4dc-ce74-4db3-bf40-df2df83abac4)

then drag to background

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/e44d1033-462c-4c0c-b8b8-d6d10dfd20b8)

(Then get a portrait of michael)

michael jackson, portrait, high quality, 4k

Negative prompt: full body, nsfw, bad, ugly

Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 512x640, Model hash: 15012c538f, Model: realisticVisionV51_v51VAE, layerdiffusion_enabled: True, layerdiffusion_method: (SD1.5) From Background to Foreground, layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: True, layerdiffusion_bg_image: True, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, layerdiffusion_fg_additional_prompt: , layerdiffusion_bg_additional_prompt: , layerdiffusion_blend_additional_prompt: , Version: f0.0.17v1.8.0rc-latest-276-g29be1da7

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/4c09065b-45ab-491e-a1bb-86b8cc58ce1a)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffuse/assets/161511761/331e6027-3ab1-4c31-8730-517c45200b17)

### Background Condition (SDXL, two steps workflow)

First download this image:

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/e7e2d80e-ffbe-4724-812a-5139a88027e3)

then set the interface with

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/99a7e648-a83f-4ea5-bff6-66a1c624c0bd)

then set the parameters with

old man sitting, high quality

Negative prompt: bad, ugly

Steps: 20, Sampler: DPM++ 2M SDE Karras, CFG scale: 7, Seed: 12345, Size: 896x1152, Model hash: 1fe6c7ec54, Model: juggernautXL_version6Rundiffusion, layerdiffusion_enabled: True, layerdiffusion_method: From Background to Blending, layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: False, layerdiffusion_bg_image: True, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, Version: f0.0.17v1.8.0rc-latest-269-gef35383b

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/4dd5022a-d9fd-4436-83b8-775e2456bfc6)

Then set the interface with (you first change the mode and then drag the image from result to interface)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/8277399c-fc9b-43fd-a9bb-1c7a8dcebb3f)

Then change the sampler to Euler A or UniPC or some other sampler that is not dpm (This is probably because of some difference between diffusers training script and webui's k-diffusion. I am still looking into this and may revise my training script and model very soon so that this step will be removed.)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/2c7124c5-e5d4-40cf-b106-e55c33e40003)

FAQ:

*OK. But how can I get a background image like this?*

You can use the Foreground Condition to get a background like this. We will describe it in the next section.

Or you can use old inpainting tech to perform foreground removal on any image to get a background like this.

*Wait. Why you generate it with two steps? Can I generate it with one pass?*

Two steps allows for more flexible editing. We will release the one-step model soon for SDXL. Also, note that the one-step model for SD1.5 is already released.

Also you can see that the current model is about 680MB and in particular I think it is a bit too lightweight and will soon release a relatively heavier model for potential stronger structure understanding (but that is still under experiments).

# Foreground Condition (SDXL, two steps workflow)

First we generate a dog

a dog sitting, high quality

Negative prompt: bad, ugly

Steps: 20, Sampler: DPM++ 2M SDE Karras, CFG scale: 7, Seed: 12345, Size: 896x1152, Model hash: 1fe6c7ec54, Model: juggernautXL_version6Rundiffusion, layerdiffusion_enabled: True, layerdiffusion_method: Only Generate Transparent Image (Attention Injection), layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: True, layerdiffusion_bg_image: False, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, Version: f0.0.17v1.8.0rc-latest-269-gef35383b

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/dd515df4-cc58-47e0-8fe0-89e21e8320c4)

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/e2785fd4-c168-4062-ae2f-010540ff0991)

then change to `From Foreground to Blending` and drag the transparent image to foreground input.

Note that you drag the real transparent image, not the visualization with checkboard background. Make sure tou see this

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/b912e1e8-7511-4afc-aa61-4bb31d6949f7)

then do this

a dog sitting in room, high quality

Negative prompt: bad, ugly

Steps: 20, Sampler: DPM++ 2M SDE Karras, CFG scale: 7, Seed: 12345, Size: 896x1152, Model hash: 1fe6c7ec54, Model: juggernautXL_version6Rundiffusion, layerdiffusion_enabled: True, layerdiffusion_method: From Foreground to Blending, layerdiffusion_weight: 1, layerdiffusion_ending_step: 1, layerdiffusion_fg_image: True, layerdiffusion_bg_image: False, layerdiffusion_blend_image: False, layerdiffusion_resize_mode: Crop and Resize, Version: f0.0.17v1.8.0rc-latest-269-gef35383b

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/0b2abb76-56b9-448d-8f2a-8572a18c759b)

Then change mode, drag your image, so that

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/48667cbf-e460-4037-b059-a30580841bcd)

(Note that here I set stop at as 0.5 to get better results since I do not need the bg to be exactly same)

Then change the sampler to Euler A or UniPC or some other sampler that is not dpm (This is probably because of some difference between diffusers training script and webui's k-diffusion. I am still looking into this and may revise my training script and model very soon so that this step will be removed.)

then do this

room, high quality

Negative prompt: bad, ugly

Steps: 20, Sampler: UniPC, CFG scale: 7, Seed: 12345, Size: 896x1152, Model hash: 1fe6c7ec54, Model: juggernautXL_version6Rundiffusion, layerdiffusion_enabled: True, layerdiffusion_method: From Foreground and Blending to Background, layerdiffusion_weight: 1, layerdiffusion_ending_step: 0.5, layerdiffusion_fg_image: True, layerdiffusion_bg_image: False, layerdiffusion_blend_image: True, layerdiffusion_resize_mode: Crop and Resize, Version: f0.0.17v1.8.0rc-latest-269-gef35383b

![image](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/5f5a5b6a-7dd2-4e16-9571-1458a9ef465d)

Note that this is a two-step workflow. We will release the one-step model soon for SDXL. Also, note that the one-step model for SD1.5 is already released.
