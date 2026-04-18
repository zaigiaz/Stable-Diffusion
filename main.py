from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from datetime import datetime
from PIL import Image
import torch
import os
import sys
import argparse

# make the image directory where the images will be saved
os.makedirs("./img", exist_ok=True)

# folder where the stable diffusion safetensors and json config is at
model = "./stable_diffusion-1.5"

# if no gpu, then use cpu inference
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(10)

def main():
    """
    read the prompts from the command line parser and choose either text or image and text pipeline
    """
    # returns prompt and img if the path was inserted
    user_prompt, img_path = command_line()

    if img_path == None or img_path == "":
        txt_pipeline(user_prompt)
    else:
        img_pipeline(user_prompt, img_path)

    print("Program Has Ended")
    sys.exit(0)


def txt_pipeline(user_prompt):
    """
    Uses the basic Stable DIffusion Pipeline to get output of image from text prompt, we use
    this pipeline when no example image is given and we just rely on given text input
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        local_files_only=True,                         
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,                           
    )

    pipe = pipe.to(device)

    # can change interations and guidance scale for higher or lower quality images, although they would take more time
    image = pipe(user_prompt, num_inference_steps=25, guidance_scale=7.0).images[0]

    # gets current timestamp and names the file that
    file_name = "./img/" + time_stamp() + ".png"

    image.save(file_name)
    print("Saved File: ", file_name)
    

def img_pipeline(user_prompt, img_path):
    """
    Uses the img2img pipeline for stable diffusion to use an example image as input for prompt as well
    """
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
        safety_checker=None,
    )

    pipe = pipe.to(device)

    # load with PIL and ensure RGB
    init_image = Image.open(img_path).convert("RGB")

    # resize to multiples of 8
    w, h = init_image.size
    new_w, new_h = (w // 8) * 8, (h // 8) * 8
    if (new_w, new_h) != (w, h):
        init_image = init_image.resize((new_w, new_h), resample=Image.LANCZOS)

    result = pipe(
        prompt=user_prompt,
        image=init_image,
        strength=0.6,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]

    file_name = "./img/" + time_stamp() + ".png"
    result.save(file_name)
    print("Saved File:", file_name)


def command_line() -> tuple[str, str]:
    """
    Parse the command line and grab the user prompt
    Usage: -p or --prompt then your input
    Usage: -i or --image for the input image you want as guidance
    """
    parser = argparse.ArgumentParser(
        prog='diffusion',
        description='generate images')

    parser.add_argument('-p', '--prompt', type=str)
    parser.add_argument('-i', '--image', type=str)
    args = parser.parse_args()

    if args.prompt == None or args.prompt == "":
        print("no user prompt was added. \n usage: python3 main.py -p 'fantasy landscape'")
        sys.exit()

    if args.image and not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {img_path}")

    return args.prompt, args.image


def time_stamp() -> str:
    """
    gets timestamp signature and returns
    """
    now = datetime.now()
    timestring = f'{now.year}-{now.month}-{now.day}__{now.hour}-{now.minute}-{now.second}'
    return timestring


if __name__ == '__main__':
    main()
