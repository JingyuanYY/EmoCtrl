import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType

PROMPT = "You are an expert in the field of visual emotions and you need to give answers based on the user's questions."
MAX_LENGTH = 2048

qwen3_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=False,  # 训练模式
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    modules_to_save=["embed_tokens"],
)


def load_tokenizer_and_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    return tokenizer, model


def process_func(tokenizer, example):
    """
    将数据集进行预处理
    """
    question = example["instruction"]
    # system + user, 模仿原格式
    system_part = f"<|im_start|>system\n{PROMPT}<|im_end|>\n"
    user_part = f"<|im_start|>user\n{question}<|im_end|>\n"
    assistant_prefix = "<|im_start|>assistant\n"
    # 分词
    instruction = tokenizer(
        system_part + user_part + assistant_prefix, add_special_tokens=False
    )
    response = tokenizer(f"{example['response']}", add_special_tokens=False)
    # 拼接
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    model.eval()
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # 禁止思考
    )
    model_inputs = tokenizer([text], return_tensors="pt")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
