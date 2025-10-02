import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate audio using Stable Audio Tools')
    parser.add_argument('--prompt', '-p', type=str, required=True,
                       help='Text prompt for audio generation')
    parser.add_argument('--output', '-o', type=str, default='output.wav',
                       help='Output filename (default: output.wav)')
    parser.add_argument('--duration', '-d', type=float, default=30.0,
                       help='Audio duration in seconds (default: 30.0)')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of diffusion steps (default: 100)')
    parser.add_argument('--cfg_scale', type=float, default=7.0,
                       help='CFG scale (default: 7.0)')
    parser.add_argument('--sigma_min', type=float, default=0.3,
                       help='Minimum sigma value (default: 0.3)')
    parser.add_argument('--sigma_max', type=float, default=500.0,
                       help='Maximum sigma value (default: 500.0)')
    parser.add_argument('--sampler', type=str, default='dpmpp-3m-sde',
                       choices=['dpmpp-3m-sde'],  # Add other samplers if available
                       help='Sampler type (default: dpmpp-3m-sde)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu), auto-detects if not specified')
    parser.add_argument('--model', type=str, default='stabilityai/stable-audio-open-1.0',
                       help='Pretrained model name (default: stabilityai/stable-audio-open-1.0)')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Download model
    print("Loading model...")
    model, model_config = get_pretrained_model(args.model)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    model = model.to(device)
    print("Model loaded successfully")

    # Set up text and timing conditioning
    conditioning = [{
        "prompt": args.prompt,
        "seconds_start": 0,
        "seconds_total": args.duration
    }]

    print(f"Generating audio: '{args.prompt}' ({args.duration}s)")

    # Generate stereo audio
    output = generate_diffusion_cond(
        model,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sampler_type=args.sampler,
        device=device
    ).cpu()

    print("Generation completed")

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")

    # Peak normalize, clip, convert to int16, and save to file
    print("Processing audio...")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # Save to file
    torchaudio.save(args.output, output, sample_rate)
    print(f"Audio saved to: {args.output}")

if __name__ == "__main__":
    main()
