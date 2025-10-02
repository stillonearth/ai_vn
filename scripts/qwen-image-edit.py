import os
from PIL import Image
import torch
from diffusers import QwenImageEditPlusPipeline
import argparse

def main():
    parser = argparse.ArgumentParser(description='Edit images using QwenImageEditPlusPipeline')
    parser.add_argument('--image1', '-i1', type=str, required=True,
                       help='Path to first input image (background)')
    parser.add_argument('--image2', '-i2', type=str, required=False,
                       help='Path to second input image (source for editing)')
    parser.add_argument('--prompt', '-p', type=str, required=True,
                       help='Text prompt for image editing')
    parser.add_argument('--output', '-o', type=str, default='output_image_edit.png',
                       help='Output filename (default: output_image_edit.png)')
    parser.add_argument('--negative_prompt', '-n', type=str, default=' ',
                       help='Negative prompt (default: empty space)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--steps', type=int, default=30,
                       help='Number of inference steps (default: 30)')
    parser.add_argument('--cfg_scale', type=float, default=4.0,
                       help='CFG scale (default: 4.0)')
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA instead of CPU offload (requires 20GB VRAM)')

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.image1):
        raise FileNotFoundError(f"Input image 1 not found: {args.image1}")

    # Load model
    model_path = "ovedrive/Qwen-Image-Edit-2509-4bit"
    pipeline = QwenImageEditPlusPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    print("Pipeline loaded")

    # Configure device
    if args.use_cuda:
        pipeline.to("cuda")
        print("Using CUDA")
    else:
        pipeline.enable_model_cpu_offload()
        print("Using CPU offload")

    pipeline.set_progress_bar_config(disable=None)

    # Load images
    image1 = Image.open(args.image1).convert("RGB")

    images = [image1]

    if args.image2:
        image2 = Image.open(args.image2).convert("RGB")
        images.append(image2)

    # Prepare inputs
    inputs = {
        "image": images,
        "prompt": args.prompt,
        "generator": torch.manual_seed(args.seed),
        "true_cfg_scale": args.cfg_scale,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.steps,
    }

    # Generate output
    with torch.inference_mode():
        output = pipeline(**inputs)

    # Save output
    output_image = output.images[0]
    output_image.save(args.output)
    print(f"Image saved at {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
