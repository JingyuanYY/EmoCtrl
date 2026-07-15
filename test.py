from utils import build_wrapper, load_pipe_emoenhance, text_gen, image_gen
import argparse
import torch
import os
import json
from datetime import datetime
from peft import PeftModel


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--base_llm",
        type=str,
        default="/mnt/d/models/Qwen3-0.6B",
        help="Path to base llm model",
    )
    args.add_argument(
        "--base_sd",
        type=str,
        default="/mnt/d/models/stable-diffusion-xl-base-1.0",
        help="Path to base sd model",
    )
    args.add_argument(
        "--checkpoint",
        type=str,
        default="./results/stage1",
        help="Path to stage1 checkpoint",
    )
    args.add_argument(
        "--emoenhance_path",
        type=str,
        default="./results/stage2",
        help="Path to stage2 checkpoint",
    )
    args.add_argument(
        "--grpo_path",
        type=str,
        default="./results/stage3",
        help="Path to stage3 checkpoint",
    )
    args.add_argument(
        "--instruction", type=str, default="caption", help="element or caption or json"
    )
    args.add_argument("--seed", type=int, default=2025, help="random seed")
    args.add_argument("--intensity", type=float, default=0.3, help="emotion intensity")
    args.add_argument("--use_lora", type=bool, default=False, help="use lora weights")
    args.add_argument("--device", type=str, default="cuda:7", help="gpu device")
    args.add_argument("--bf16", action="store_true", help="use bf16")
    args.add_argument("--output_dir", type=str, default="./results", help="output dir")
    args.add_argument("--emotion", type=str, default="", help="emotion category")
    return args.parse_args()


def main():
    args = parse_args()
    tokenizer, model = build_wrapper(
        base_llm=args.base_llm,
        ckpt_dir=args.checkpoint,
        device=args.device,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )
    pipe, emoenhance = load_pipe_emoenhance(
        base_sd=args.base_sd,
        ckpt_dir=args.emoenhance_path,
        use_lora=args.use_lora,
        device=args.device,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, args.grpo_path)

    seed = args.seed
    generator = torch.Generator(device=args.device).manual_seed(seed)

    output_dir = args.output_dir
    emotion = args.emotion
    output_dir = os.path.join(output_dir, args.instruction, emotion)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    prompt_template = "Based on the <element>, give me a prompt to generate an image."

    element_list = ["dog", "cat", "tree", "sky"]

    caption_list = [
        "A dog is playing in the park.",
        "A tree in the park.",
        "A cat is sitting on the sofa.",
        "A sky with clouds.",
    ]

    date = datetime.now().strftime("%m%d%H")

    if args.instruction == "element":
        for text in element_list:
            prompt = prompt_template.replace("<element>", text)
            resp = text_gen(tokenizer, model, emotion, prompt)
            image_gen(
                pipe,
                emoenhance,
                generator,
                prompt=resp,
                emotion=emotion,
                device=args.device,
                intensity=0.3,
                sd_img=f"{output_dir}/sd-{text}-{emotion}.png",
                emo_img=f"{output_dir}/emo-{text}-{emotion}.png",
                is_ablation=False,
            )
    elif args.instruction == "caption":
        for text in caption_list:
            resp = text_gen(tokenizer, model, emotion, text)
            image_gen(
                pipe,
                emoenhance,
                generator,
                prompt=resp,
                emotion=emotion,
                device=args.device,
                intensity=0.3,
                sd_img=f"{output_dir}/sd-{text}-{emotion}.png",
                emo_img=f"{output_dir}/emo-{text}-{emotion}.png",
                is_ablation=False,
            )
    elif args.instruction == "json":
        with open("./data/test_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        infer_elem_list = data["element"]
        infer_elem_list = sorted(list(set(infer_elem_list)))
        infer_caption_list = data["caption"]
        infer_caption_list = sorted(list(set(infer_caption_list)))
        uni_content_list = infer_elem_list + infer_caption_list

        for i, text in enumerate(uni_content_list):
            if "." not in text:
                prompt = prompt_template.replace("<element>", text)
            else:
                prompt = text
            resp = text_gen(tokenizer, model, emotion, prompt)
            image_gen(
                pipe,
                emoenhance,
                generator,
                prompt=resp,
                emotion=emotion,
                device=args.device,
                intensity=0.3,
                sd_img=f"{output_dir}/sd-{text}-{emotion}-{date}-{seed}.png",
                emo_img=f"{output_dir}/{i}-{text}-{emotion}-{date}-{seed}.png",
                is_ablation=True,
            )


if __name__ == "__main__":
    main()
