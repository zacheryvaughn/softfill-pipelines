import torch
from PIL import Image
from pipeline_stable_diffusion_xl_softfill import StableDiffusionXLSoftfillPipeline

# Results currently largely depend on how good your model is at InPainting...
# Most models will cut off portions of the inpaint, and if it's too extreme then DiffDiff will assume that this is the desired result.

# Determine device and precision
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

# Load Pipeline
print("Loading pipeline...")
pipeline = StableDiffusionXLSoftfillPipeline.from_single_file(
    "SDXL_Anime.safetensors", # Add model here.
    torch_dtype=torch_dtype,
    use_safetensors=True,
).to(device)

# Load the image and mask
image = Image.open("image.png").convert("RGB")
mask_image = Image.open("mask_image.png").convert("RGB")

# Set generation parameters
prompt = "upclose, closeup, brunette girl with green eyes, wearing blue dress with white pattern, background forest"
negative_prompt = "nude, nsfw"
num_inference_steps = 30 # inpaint_steps are 1/3 of num_inference_steps
strength = 0.9 # effective range 0.8 - 1.0
guidance_scale = 6
seed = 123456
generator = torch.Generator(device=device).manual_seed(seed)

# Get dimensions of nput image
width, height = image.size

# Run the pipeline
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    mask_image=mask_image,
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,
    strength=strength,
    guidance_scale=guidance_scale,
    generator=generator
).images[0]

# Save the output
image.save("output.png")
print(f"Saved output.png")