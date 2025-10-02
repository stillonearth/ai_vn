import torch
from diffusers import ChromaPipeline
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate images using ChromaPipeline')
    parser.add_argument('--prompt', '-p', type=str, required=True,
                       help='Text prompt for image generation')
    parser.add_argument('--output', '-o', type=str, default='chroma.png',
                       help='Output filename (default: chroma.png)')
    parser.add_argument('--negative_prompt', '-n', type=str,
                       default='low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors',
                       help='Negative prompt for image generation')
    parser.add_argument('--seed', type=int, default=433,
                       help='Random seed (default: 433)')
    parser.add_argument('--steps', type=int, default=40,
                       help='Number of inference steps (default: 40)')
    parser.add_argument('--guidance_scale', '-g', type=float, default=3.0,
                       help='Guidance scale (default: 3.0)')

    args = parser.parse_args()

    # Initialize pipeline
    pipe = ChromaPipeline.from_pretrained("lodestones/Chroma1-HD", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    # Generate image
    image = pipe(
        prompt=[args.prompt],
        negative_prompt=[args.negative_prompt],
        generator=torch.Generator("cuda").manual_seed(args.seed),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=1,
    ).images[0]

    # Save output
    image.save(args.output)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()
