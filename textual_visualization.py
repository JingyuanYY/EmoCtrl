from utils import build_wrapper, text_gen

import torch
import os
from diffusers import DiffusionPipeline


def parse_args():
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--base_llm", type=str, default="", help="your base llm model")
    args.add_argument("--base_sd", type=str, default="", help="your base sd model")
    args.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="your textual emotion enhancement checkpoint",
    )
    args.add_argument("--num_samples", type=int, default=1000, help="number of samples")
    args.add_argument("--device", type=str, default="cuda:7", help="gpu device")
    args.add_argument("--bf16", action="store_true", help="use bf16")
    args.add_argument("--output_dir", type=str, default="", help="output dir")
    args.add_argument("--emotion", type=str, default="", help="emotion to visualize")
    return args.parse_args()


def main():
    args = parse_args()
    tokenizer, model = build_wrapper(
        base_llm=args.base_llm,
        ckpt_dir=args.checkpoint,
        device=args.device,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )
    pipe = DiffusionPipeline.from_pretrained(args.base_sd, torch_dtype=torch.bfloat16)
    pipe = pipe.to(args.device)
    emotion = args.emotion
    num_samples = args.num_samples
    output_dir = os.path.join(args.output_dir, emotion)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_samples):
        text = ""
        resp = text_gen(tokenizer, model, emotion, text)
        image = pipe(prompt=resp, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(f"{output_dir}/{i}-{emotion}-{resp}.png")


if __name__ == "__main__":
    main()
