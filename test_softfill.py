import torch
from PIL import Image
from pipeline_stable_diffusion_xl_softfill import StableDiffusionXLSoftFillPipeline

# SET DEVICE & PRECISION
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

# LOAD PIPELINE
print("Loading pipeline...")
pipeline = StableDiffusionXLSoftFillPipeline.from_single_file(
    "StableDiffusionXL.safetensors", # Add your model here.
    torch_dtype=torch_dtype,
    use_safetensors=True,
).to(device)

# PIL OPEN IMAGES
image = Image.open("image.png")
mask = Image.open("mask.png")

# INPUT PROMPTS
prompt = "selfie photo of pretty girl, with blue bikini top, blue eyes, at beautiful white tropical bali villa pool"
negative_prompt = "deformed, bad anatomy, from above, fat, lowres, lower body, nude, nsfw, nipples, mirror, holding phone"

# RUN PIPELINE
output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    mask=mask,
    noise_fill_image=True, # If you are attempting to modify existing content, this should be False. If you have drawn the desired object onto the image already, this should be False.
    num_inference_steps=32,
    guidance_scale=4,
    strength=0.8,
).images[0]

# SAVE OUTPUT
output.save("output.png")
print("Done!")