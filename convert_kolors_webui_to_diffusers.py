import os
import argparse
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel


def convert_unet_key(state_dict):
    with open("./kolors/unet/unet_map_keys.txt", "r") as f:
        mapping = {v: k for k, v in [_.strip().split(",") for _ in f.readlines()]}

    unet_state_dict = {}
    for k, v in state_dict.items():
        if "model.diffusion_model." in k:
            k = k.replace("model.diffusion_model.", "")
            if k in mapping.keys():
                unet_state_dict[mapping[k]] = v
            else:
                print(f"miss key {k}")

    return unet_state_dict

def convert_and_save(input_webui_unet, output_diffusion_unet):
    state_dict = load_file(input_webui_unet)

    unet_state_dict = convert_unet_key(state_dict)

    print(f"Converted UNet keys: {list(unet_state_dict.keys())[:10]}")

    unet = UNet2DConditionModel.from_config("./kolors/unet/")
    unet.load_state_dict(unet_state_dict)
    unet.save_pretrained(output_diffusion_unet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert unet model in the Kohya format to diffusion model.")
    parser.add_argument(
        "--input_webui_unet",
        type=str,
        required=True,
        help="Path to the input unet model file in the Kohya format.",
    )
    parser.add_argument(
        "--output_diffusion_unet",
        type=str,
        required=True,
        help="Path for the converted diffusion model.",
    )

    args = parser.parse_args()

    convert_and_save(args.input_webui_unet, args.output_diffusion_unet)

