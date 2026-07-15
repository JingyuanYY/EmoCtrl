from safetensors.torch import load_file
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from emotion_enhancement import EmotionMapper
from qwen3_config import qwen3_config as config, PROMPT
from emotion_conditioned_llm import EmotionConditionedCausalLM, EmotionEmbeddingBank
from peft import get_peft_model
from diffusers import DiffusionPipeline
from typing import Union, List

emotion_list = [
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
]

emotion2id = {e: i for i, e in enumerate(emotion_list)}


def load_tokenizer_and_model(model_dir, device="cuda:0", torch_dtype=None):
    """加载基座模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype)
    model.to(device)
    return tokenizer, model


def build_wrapper(base_llm, ckpt_dir, device="cuda:0", torch_dtype=None):
    """构建情感条件化模型"""
    tokenizer, base_model = load_tokenizer_and_model(
        base_llm, device=device, torch_dtype=torch_dtype
    )
    base_model = get_peft_model(base_model, config)
    model = EmotionConditionedCausalLM(base_lm=base_model)

    model_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(model_path):
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)
        print(f"[Qwen] Loaded model from {model_path}")
    else:
        print(f"[Qwen] No model found at {model_path}")
    model.to(device, dtype=torch_dtype)
    model.eval()
    return tokenizer, model


def load_pipe_emoenhance(
    base_sd, ckpt_dir, use_lora=False, device="cuda:0", torch_dtype=None
):
    """加载情感增强模块"""
    pipe = DiffusionPipeline.from_pretrained(base_sd, torch_dtype=torch_dtype)
    lora_path = os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors")
    if use_lora and os.path.exists(lora_path):
        pipe.load_lora_weights(lora_path)
        print(f"[SDXL] Loaded LoRA weights from {lora_path}")
    else:
        print(f"[SDXL] No LoRA weights found at {lora_path}")
    pipe = pipe.to(device)

    d_model = (
        pipe.text_encoder.config.hidden_size + pipe.text_encoder_2.config.hidden_size
    )
    emoenhance = EmotionMapper(input_dim=d_model, emo_bank=EmotionEmbeddingBank())
    model_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(model_path):
        state_dict = load_file(model_path)
        emoenhance.load_state_dict(state_dict)
        print(f"[EmotionEnhancer] Loaded model from {model_path}")
    else:
        print(f"[EmotionEnhancer] No model found at {model_path}")
    emoenhance.to(device, dtype=torch_dtype)
    emoenhance.eval()
    return pipe, emoenhance


def apply_chat_template(tokenizer, system_prompt: str, user_text: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return text


@torch.inference_mode()
def text_gen(
    tokenizer,
    model: EmotionConditionedCausalLM,
    emotion: Union[str, List[str]],
    instruction: str,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_p: float = 0.9,
    is_ablation=False,
    is_verbose=True,
):
    """文本生成，含消融实验"""
    device = model.base_lm.device
    model_dtype = model.base_lm.dtype

    embed = model.base_lm.get_input_embeddings()

    if is_verbose:
        print(f"==========\n[{emotion}] {instruction}")

    text = apply_chat_template(tokenizer, PROMPT, instruction)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    inp_embeds = embed(inputs["input_ids"])  # [1, T, D]
    atten_mask = inputs["attention_mask"]

    emo_idx = model.emo_bank.names_to_indices(emotion).to(device)
    emo_tokens = model.emo_bank.get_tokens(emo_idx, dtype=model_dtype)  # [1, K, S]
    if emo_tokens.shape[0] > 1:
        emo_tokens = emo_tokens.view(1, -1, emo_tokens.shape[-1])
    prompt_embeds = model.emo_proj(emo_tokens).to(model_dtype)  # [1, K, D]

    if is_ablation:
        out_ids = model.base_lm.generate(
            inputs_embeds=inp_embeds,
            attention_mask=atten_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
        if is_verbose:
            print(
                f"wo emo token: {tokenizer.decode(out_ids[0], skip_special_tokens=True)}"
            )

    inputs_embeds = torch.cat([prompt_embeds, inp_embeds], dim=1)
    prefix_mask = torch.ones(1, prompt_embeds.size(1), dtype=model_dtype, device=device)
    attention_mask = torch.cat([prefix_mask, atten_mask], dim=1)

    out_ids = model.base_lm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    if is_verbose:
        print(f"w emo token: {tokenizer.decode(out_ids[0], skip_special_tokens=True)}")
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


@torch.inference_mode()
def image_gen(
    pipe,
    emoenhance,
    generator,
    prompt,
    emotion,
    device,
    intensity=0.3,
    sd_img=None,
    emo_img=None,
    is_ablation=False,
    original_prompt=None,
):
    """图像生成，含消融实验"""
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    if original_prompt is not None:
        (
            ori_prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=original_prompt,
            negative_prompt="dark, underexposed, low light, heavy shadows, dim",
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

    if is_ablation:
        image = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]
        image.save(sd_img)
        print(f"Image saved as {sd_img}")

    emo_prompt_embeds = emoenhance(
        prompt_embeds=prompt_embeds, emotion=emotion, intensity=intensity
    )
    if original_prompt is not None:
        emo_prompt_embeds = torch.cat([ori_prompt_embeds, emo_prompt_embeds], dim=1)

    image = pipe(
        prompt_embeds=emo_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    image.save(emo_img)
    print(f"Image saved as {emo_img}")


@torch.inference_mode()
def text_gen_batch(
    tokenizer,
    model: EmotionConditionedCausalLM,
    emotions,  # Union[str, int, List[Union[str,int]], Dict[str,float], torch.Tensor]
    instructions,  # List[str]
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    is_verbose: bool = False,
):
    """
    批量文本生成。
    - emotions: 可为单个情感或与 instructions 等长的列表
    - instructions: 用户输入（已拼好元素的 prompt）
    返回: List[str]，与 instructions 对齐
    """
    assert isinstance(instructions, (list, tuple)) and len(instructions) > 0
    results = []
    for idx in range(len(instructions)):
        text = text_gen(
            tokenizer=tokenizer,
            model=model,
            emotion=emotions[idx],
            instruction=instructions[idx],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            is_verbose=is_verbose,
        )
        results.append(text)
    return results
