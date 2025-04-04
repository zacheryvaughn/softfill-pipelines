# SoftFill Pipeline

SoftFill is a Diffusers Pipeline based on Differential Diffusion, implementing some input and preprocessing modifications which allow it to behave more like "soft inpainting" without requiring any extra inference steps. The small addition of a nondeterministic fluidlike noise fill is surprisingly good at guiding the img2img process toward producing an output in the masked region with a similar shape to the mask.

Differential Diffusion is proposed and developed by https://github.com/exx8 <br>
Standard differential img2img pipeline can be found at https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_stable_diffusion_xl_differential_img2img.py

#### Modifications from the original DiffDiff pipeline:
- Accepts an "image" and "mask" in PIL format.
- The mask should be a standard blurred mask, where white areas represent the inpaint area.
- The pipeline preprocesses the image to generate a randomly warped perlin noise in the >0.5 mask area and pastes that noise onto the image.
- The noise is applied by default and is optional; "noise_fill_image=True".
- The noise has an opacity of 0.75, allowing the img2img processes to still get some background context if needed.
- The noise has an edge fade which attempts to be half the mask's blur radius.

## Test the Pipeline

1. **Copy Structure:**
```
└── 📁any-folder
    └── pipeline_stable_diffusion_xl_softfill.py
    └── test_softfill.py
    └── setup.sh
    └── any-sdxl-model.safetensors
    └── image.png
    └── mask.png
```

2. **Run the Setup file:**
```bash
bash setup.sh
```
Note: You may need to use "python3" instead of "python" in 'setup.sh'.

3. **Add Model to 'test_softfill.py':**
```python
pipeline = StableDiffusionXLSoftfillPipeline.from_single_file(
--> "any-sdxl-model.safetensors", <--
    torch_dtype=torch_dtype,
    use_safetensors=True,
).to(device)
```

4. **Add your Prompt in 'test_softfill.py':**
```python
prompt = "describe what you want to see in the masked area"
negative_prompt = "describe what you do NOT want to see in the masked area"
```
Note: Only describe within masked area.

5. **Activate your Virtual Environment':**
```bash
source .venv/bin/activate
```

7. **Run the 'test_softfill.py':**
```bash
python test_softfill.py
```

## Example Image Process
**Model Used:** https://civitai.com/models/277058?modelVersionId=1522905 <br>
**prompt=** selfie photo of pretty girl, with blue bikini top, blue eyes, at beautiful white tropical bali villa pool <br>
**negative_prompt=** deformed, bad anatomy, from above, fat, lowres, lower body, nude, nsfw, nipples, mirror, holding phone <br>
**num_inference_steps=** 32 <br>
**guidance_scale=** 4, <br>
**strength=** 0.8, <br>
**noise_fill_image=** True,

| Image | Mask | Noised | Result |
|----------------|------|-------------------|---------------|
| ![original.png](image.png) | ![mask.png](mask.png) | ![noised_image.png](noised_image.png) | ![result.png](result.png) |
