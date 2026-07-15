import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from utils import build_wrapper, text_gen_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="/mnt/d/models/Qwen3-0.6B")
    parser.add_argument(
        "--in_json", type=str, default="", help="Path to input jsonal file"
    )
    parser.add_argument(
        "--out_json", type=str, default="", help="Path to output jsonl file"
    )
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./results/stage1",
        help="Path to checkpoint directory",
    )
    parser.add_argument("--device", type=str, default="cuda:7", help="gpu device")
    parser.add_argument("--bf16", action="store_true", help="use bf16")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.bf16 else None
    tokenizer, model = build_wrapper(
        base_llm=args.base_model,
        ckpt_dir=args.checkpoint,
        device=args.device,
        torch_dtype=dtype,
    )

    df = pd.read_json(args.in_json, lines=True)
    rows = []
    B = args.batch_size
    for i in tqdm(range(0, len(df), B)):
        chunk = df.iloc[i : i + B]
        emos = [str(x) for x in chunk["emotion"].tolist()]
        ins = [str(x) for x in chunk["instruction"].tolist()]
        gens = text_gen_batch(
            tokenizer,
            model,
            emos,
            ins,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for (idx, r), txt in zip(chunk.iterrows(), gens):
            rr = dict(r)
            rr["llm_text"] = txt
            rows.append(rr)
        # break  # only one batch for test
    with open(args.out_json, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"prefill done → {args.out_json}")


if __name__ == "__main__":
    main()
