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
prompt = "photo of pretty girl, age 20, blonde hair, blue eyes and yellow tropical dress, at a bali villa with the pool and villa behind her"
negative_prompt = "closeup, nipples, nude, fat, lowres, worst quality, studio lighting"

# RUN PIPELINE
output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    # Both the image and mask should be PIL.
    image=image,
    mask=mask,
    # Defaults to true. Adds noise to the image at the mask's >0.8 area.
    noise_fill_image=True, # If your UI allows for drawing on the image, you probably want this to be False in that case.
    num_inference_steps=40,
    guidance_scale=3,
    strength=0.8,
).images[0]

# SAVE OUTPUT
output.save("output.png")
print("Done!")