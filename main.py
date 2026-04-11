from diffusers import StableDiffusionPipeline
from datetime import datetime
import torch
import sys
import argparse

# folder where the stable diffusion safetensors and json config is at
model = "./stable_diffusion-1.5"

# if no gpu, then use cpu inference
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(12)

def main():
    """
    Create the Stable Diffusion Pipeline then save the image created
    as with timestamp UUID in img/ folder, user prompts are from command line with -p or --p
    """
    user_prompt = command_line()

    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        local_files_only=True,                         # load from local repo
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,                           # optional for faster load
    )

    pipe = pipe.to(device)

    # gets current timestamp and names the file that
    file_name = "./img/" + time_stamp() + ".png"

    # can change interations and guidance scale for higher or lower quality images, although they would take more time
    image = pipe(user_prompt, num_inference_steps=25, guidance_scale=9).images[0]
    image.save(file_name)
    print("Saved File: ", file_name)
    print("Program Finished")


def command_line() -> str:
    """
    Parse the command line and grab the user prompt
    Usage: -p or --prompt then your input
    """
    parser = argparse.ArgumentParser(
        prog='diffusion',
        description='generate images')

    parser.add_argument('-p', '--prompt', type=str)
    args = parser.parse_args()

    if args.prompt == None or args.prompt == "":
        print("no user prompt was added. \n usage: python3 main.py -p 'fantasy landscape'")
        sys.exit()

    return args.prompt


def time_stamp() -> str:
    """
    gets timestamp signature and returns
    """
    now = datetime.now()
    timestring = f'{now.year}-{now.month}-{now.day}__{now.hour}:{now.minute}:{now.second}'
    return timestring


if __name__ == '__main__':
    main()
