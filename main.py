# run_sd15.py
from diffusers import StableDiffusionPipeline
import torch
import sys
import argparse
import hashlib

# folder where the stable diffusion safetensors and json config is at
model = "./stable_diffusion-1.5"

# if no gpu, then use cpu inference
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(12)

def main():
    """
    Create the Stable Diffusion Pipeline then save the image created
    as sha256 file in img/ folder, user prompts are from command line with -p or --p
    """
    user_prompt = command_line()

    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        local_files_only=True,                         # load from local repo
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,                           # optional for faster load
    )

    pipe = pipe.to(device)

    file_name = "./img/" + shortened_sha256(user_prompt) + ".png"

    image = pipe(user_prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
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


def shortened_sha256(input_str: str) -> str:
    """
    Compute SHAKE-256 of the UTF-8 encoding of input_str, request 20 bytes,
    then truncate to 16 bytes and return as a 32-char hex string.
    """
    h = hashlib.shake_256(input_str.encode('utf-8'))
    full_hex = h.hexdigest(20)           # 20 bytes -> 40 hex chars
    full_bytes = bytes.fromhex(full_hex) # length 20
    reduced_bytes = full_bytes[:16]      # length 16
    return reduced_bytes.hex()           # 32 hex chars


if __name__ == '__main__':
    main()
