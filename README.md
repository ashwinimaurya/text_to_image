# Text To Image Generation Using Diffusion Models and Hugging Face Transformers

##### These modela are trained using GPU, running on CPU machine may give runtime errors.

### 1. Using Stable Diffusion for Text to Image Generation

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from matplotlib import pyplot as plt

model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
plt.imshow(image)

### 2. Using StableDiffusionPipeline and CompVis/stable-diffusion-v1-4

import torch
from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

# Move the pipeline to the GPU for faster processing (optional)
pipe = pipe.to("cuda")

prompt = "grocery store on moon"

# Number of images to generate (you can change this)
num_images = 1

# Generate images based on the prompt
images = pipe(prompt)

# Access the generated image
generated_image = images.images[0]

plt.imshow(generated_image)
