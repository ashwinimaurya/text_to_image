# Text To Image Generation Using Diffusion Models and Hugging Face Transformers

##### These modela are trained using GPU, running on CPU machine may give runtime errors.

### 1. Using Stable Diffusion for Text to Image Generation

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from matplotlib import pyplot as plt

#### Load the Stable Diffusion pipeline

model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

#### Text prompt 
prompt = "a photo of an astronaut riding a horse on mars"

#### Generate the image
image = pipe(prompt).images[0]

#### Show image 

plt.imshow(image)
![image](https://github.com/ashwinimaurya/text_to_image/assets/13278692/d3e0e789-f0cf-4b4a-9749-e6905f846634)


### 2. Using StableDiffusionPipeline and CompVis/stable-diffusion-v1-4

import torch
from diffusers import StableDiffusionPipeline

#### Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

####  Move the pipeline to the GPU for faster processing (optional)
pipe = pipe.to("cuda")

prompt = "Man Cooking food on Mars"

####  Generate images based on the prompt
images = pipe(prompt)

####  Access the generated image
generated_image = images.images[0]

#### Show image 
plt.imshow(generated_image)
![image](https://github.com/ashwinimaurya/text_to_image/assets/13278692/a12a1a70-7cc8-412f-bf87-9de851d78ca0)
