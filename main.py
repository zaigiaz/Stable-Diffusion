from datetime import datetime
from PIL import Image
from json.decoder import JSONDecodeError
import torch
import os
import json
import sys
import argparse

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)

# Map between different seeds and schedulers
SCHEDULER_MAP = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "euler":   EulerDiscreteScheduler,
    "ddim":    DDIMScheduler,
    "dpm++_2m": DPMSolverMultistepScheduler,
}

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
    user_prompt, img_path, json_path = command_line()

    # read json file
    if json_path:
        read_json(json_path)

    # generates stable diffusion pipeline for our use
    pipe = pipeline(img_path)
    
    # main generation pipeline
    generate(pipe, user_prompt, img_path)

    print("Program Has Ended")
    sys.exit(0)


def pipeline(img):
    """
    Load the main pipeline for use in our system
    """
    MainPipeline = StableDiffusionImg2ImgPipeline if img else StableDiffusionPipeline

    pipe = MainPipeline.from_pretrained(
        model,
        local_files_only=True,                         
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,                           
    )

    pipe = pipe.to(device)
    return pipe


def generate(pipe, user_prompt, img_path):
    """
    Generate an image either with a text prompt or additional image
    """
    if not img_path:
        output = pipe(user_prompt, num_inference_steps=25, guidance_scale=7.0).images[0]
    else:
        # load with PIL and ensure RGB
        init_image = Image.open(img_path).convert("RGB")
        
        # resize to multiples of 8
        w, h = init_image.size
        new_w, new_h = (w // 8) * 8, (h // 8) * 8

        if (new_w, new_h) != (w, h):
            init_image = init_image.resize((new_w, new_h), resample=Image.LANCZOS)

        output = pipe(
            prompt=user_prompt,
            image=init_image,
            strength=0.6,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]

    # gets current timestamp and names the file that
    file_name = "./img/" + time_stamp() + ".png"

    output.save(file_name)
    print("Saved File: ", file_name)



# TODO :: finish this function, probably need to wait until we have better interface
def choose_scheduler():
    """
    Choose the scheduler you want to use with your image pipeline, if
    none then default to the basic stable-diffusion template
    """
    print("hello")



def read_json(json_path):
    """
    Reads from a json file with a specified schema
    """
    try:
        with open(json_path, mode='r') as json_file:
            data = json.load(json_file)
            # returns a dict
            # json is key-value seperated file with arrays and other features
            print(data["hello"])
            # TODO :: pull out each list of variables from json for each prompt and feed to pipeline
            
    except FileNotFoundError:
        print("File not found!")
    except JSONDecodeError:
        print("Invalid JSON format!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    sys.exit(0)


# TODO :: fill this out or another seperate function for error checking
def check_valid_path():
    """
    error checking function when opening files
    """

def command_line() -> tuple[str, str]:
    """
    Parse the command line and grab the user prompt
    Usage: -p or --prompt then your text input
    Usage: -i or --image for the input image you want as guidance
    Usage: -j or --json for the json file to read, for multiple prompts
    """
    parser = argparse.ArgumentParser(
        prog='diffusion',
        description='generate images')

    parser.add_argument('-p', '--prompt', type=str)
    parser.add_argument('-i', '--image', type=str)
    parser.add_argument('-j', '--json', type=str)
    args = parser.parse_args()

    if args.prompt == None or args.prompt == "":
        print("no user prompt was added. \n usage: python3 main.py -p 'fantasy landscape'")
        sys.exit()

    if args.json:
        if not os.path.exists(args.json):
            raise FileNotFoundError(f"not a valid file path {args.json}")
        if not os.path.isfile(args.json):
            raise FileNotFoundError(f"not a file: {args.json}")
    
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"not a valid file path {args.image}")
        if not os.path.isfile(args.image):
            raise FileNotFoundError(f"not a file: {args.image}")

    return args.prompt, args.image, args.json


def time_stamp() -> str:
    """
    gets timestamp signature and returns
    """
    now = datetime.now()
    timestring = f'{now.year}-{now.month}-{now.day}__{now.hour}-{now.minute}-{now.second}'
    return timestring


if __name__ == '__main__':
    main()
