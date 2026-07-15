import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import get_peft_model
import os
import swanlab
from qwen3_config import qwen3_config as config
from qwen3_config import process_func, PROMPT, MAX_LENGTH
from functools import partial
from emotion_conditioned_llm import EmotionConditionedCausalLM
from utils import text_gen, emotion2id


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

os.environ["SWANLAB_PROJECT"] = "emoctrl"


def format_params(num):
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def main():
    SEED = 2025
    set_seed(SEED)

    rank = int(os.environ.get("RANK", "0"))
    model_dir = "/mnt/d/models/Qwen3-0.6B"
    date = "cvpr"
    dataset = "1114-emoctrl"
    run_name = f"qwen3-0.6B-{date}"

    json_dataset_path = "./data"
    train_dataset_path = os.path.join(json_dataset_path, f"train_{dataset}.jsonl")
    test_dataset_path = os.path.join(json_dataset_path, f"val_{dataset}.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16
    )

    train_df = pd.read_json(train_dataset_path, lines=True)
    train_df["emotion_id"] = train_df["emotion"].map(emotion2id).astype(int)
    eval_df = pd.read_json(test_dataset_path, lines=True)
    eval_df["emotion_id"] = eval_df["emotion"].map(emotion2id).astype(int)
    print(f"train size: {len(train_df)}, eval size: {len(eval_df)}")

    def proc_with_emotion(ex, tokenizer):
        y = process_func(tokenizer, ex)
        y["emotion_id"] = ex["emotion_id"]
        return y

    proc_fn = partial(proc_with_emotion, tokenizer=tokenizer)
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)

    train_dataset = train_ds.map(
        proc_fn,
        remove_columns=[c for c in train_ds.column_names if c not in ["emotion_id"]],
        num_proc=16,
    )
    eval_dataset = eval_ds.map(
        proc_fn,
        remove_columns=[c for c in eval_ds.column_names if c not in ["emotion_id"]],
        num_proc=16,
    )

    print(
        f"train dataset size: {len(train_dataset)}, eval dataset size: {len(eval_dataset)}"
    )

    peft_model = get_peft_model(base_model, config)
    base_model.eval()
    model = EmotionConditionedCausalLM(base_lm=peft_model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {format_params(total_params)}")
    print(f"Trainable parameters: {format_params(trainable_params)}")

    if rank == 0:
        swanlab.init(
            project="emoctrl",
            experiment_name=run_name,
            config={
                "model": model_dir,
                "prompt": PROMPT,
                "data_max_length": MAX_LENGTH,
                "dataset": dataset,
            },
        )

    report_to = "swanlab" if rank == 0 else "none"

    args = TrainingArguments(
        output_dir=f"./results/{run_name}",
        seed=SEED,
        data_seed=SEED,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        num_train_epochs=5,
        save_steps=100,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        save_on_each_node=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to=report_to,
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    if rank == 0:
        model.eval()
        test_text_list = []

        for index, row in eval_df.iterrows():
            emotion = row["emotion"]
            instruction = row["instruction"]
            output = row["response"]

            response = text_gen(tokenizer, model, emotion, instruction)

            response_text = f"""
            [{emotion}]  Question: {instruction}

            Reference: {output}

            LLM: {response}
            """

            test_text_list.append(swanlab.Text(response_text))
            # print(response_text)

        swanlab.log({"Prediction": test_text_list})

        swanlab.finish()


if __name__ == "__main__":
    main()
