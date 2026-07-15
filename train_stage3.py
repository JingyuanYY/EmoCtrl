from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
import math
import itertools
from typing import Any, Dict, List, Optional, Union
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
import torch
import swanlab as wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

from emotion_conditioned_llm import EmotionEmbeddingBank
from emotion_enhancement import EmotionMapper
from safetensors.torch import load_file
from peft import set_peft_model_state_dict

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

DEFAULT_EMOTIONS = [
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
]


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        emotion = None
        if "|" in prompt:
            prompt, emotion_raw = prompt.rsplit("|", 1)
            emotion = emotion_raw.strip()
        assert emotion is not None, "Emotion must be specified"
        return {"prompt": prompt, "metadata": {"emotion": emotion}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # 每个副本上的批大小
        self.k = k  # 每个样本重复次数
        self.num_replicas = num_replicas  # 副本总数
        self.rank = rank  # 当前副本编号
        self.seed = seed  # 用于同步的随机种子

        # 计算每轮迭代需要的唯一样本数
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, (
            f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        )
        self.m = self.total_samples // self.k  # 唯一样本数
        self.epoch = 0

    def __iter__(self):
        while True:
            # 生成可复现的随机序列，确保所有副本采样一致
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # 随机选择 m 个不同样本
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()

            # 每个样本重复 k 次，得到 n*b 总样本
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            # 打乱顺序，保证分布均匀
            shuffled_indices = torch.randperm(
                len(repeated_indices), generator=g
            ).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            # 将样本切分到各个副本
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])

            # 返回当前副本的样本索引
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch  # 用于在训练轮之间同步随机状态


def _get_module_dtype(module):
    if hasattr(module, "module"):
        return module.module.dtype
    return module.dtype


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: Union[str, List[str]],
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    elif text_input_ids is None:
        raise ValueError("text_input_ids must be provided when tokenizer is None")

    prompt_output = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_output[0]
    prompt_embeds = prompt_output.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(
        dtype=_get_module_dtype(text_encoder), device=device
    )
    pooled_prompt_embeds = pooled_prompt_embeds.to(
        dtype=_get_module_dtype(text_encoder), device=device
    )

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(
        batch_size * num_images_per_prompt, -1
    )
    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: Union[str, List[str]],
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    del max_sequence_length
    prompt = [prompt] if isinstance(prompt, str) else prompt

    prompt_embeds_list = []
    pooled_prompt_embeds = None
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        prompt_embeds, pooled_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        prompt_embeds_list.append(prompt_embeds)
        pooled_prompt_embeds = pooled_embeds

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def get_add_time_ids(
    pipeline: StableDiffusionXLPipeline,
    original_size,
    crops_coords_top_left,
    target_size,
    dtype,
    device,
):
    projection_dim = getattr(
        getattr(pipeline.text_encoder_2, "config", None), "projection_dim", None
    )
    original_unet = pipeline.unet
    if hasattr(original_unet, "module"):
        pipeline.unet = original_unet.module
    try:
        try:
            add_time_ids = pipeline._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=dtype,
                text_encoder_projection_dim=projection_dim,
            )
        except TypeError:
            add_time_ids = pipeline._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=dtype,
            )
    finally:
        pipeline.unet = original_unet
    return add_time_ids.to(device=device, dtype=dtype)


def ddim_step_with_logprob(
    scheduler: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: torch.FloatTensor,
    sample: torch.FloatTensor,
    noise_level: float = 0.0,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
):
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    if not torch.is_tensor(timestep):
        timestep = torch.tensor([timestep], device=sample.device)
    if timestep.ndim == 0:
        timestep = timestep.unsqueeze(0)
    timestep = timestep.to(sample.device)

    schedule_timesteps = scheduler.timesteps.to(sample.device)
    step_indices = []
    for t in timestep:
        matches = (schedule_timesteps == t).nonzero(as_tuple=False)
        if len(matches) == 0:
            matches = (schedule_timesteps == t.to(schedule_timesteps.dtype)).nonzero(
                as_tuple=False
            )
        if len(matches) == 0:
            raise ValueError(f"timestep {t.item()} not found in scheduler.timesteps")
        step_indices.append(matches[0].item())

    prev_timestep = torch.full_like(timestep, -1)
    for idx, step_index in enumerate(step_indices):
        if step_index + 1 < len(schedule_timesteps):
            prev_timestep[idx] = schedule_timesteps[step_index + 1]

    alphas_cumprod = scheduler.alphas_cumprod.to(
        device=sample.device, dtype=sample.dtype
    )
    final_alpha_cumprod = scheduler.final_alpha_cumprod
    if not torch.is_tensor(final_alpha_cumprod):
        final_alpha_cumprod = torch.tensor(
            final_alpha_cumprod, device=sample.device, dtype=sample.dtype
        )
    else:
        final_alpha_cumprod = final_alpha_cumprod.to(
            device=sample.device, dtype=sample.dtype
        )

    timestep_long = timestep.long()
    prev_timestep_long = prev_timestep.long()
    alpha_prod_t = alphas_cumprod[timestep_long]
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0, alphas_cumprod[prev_timestep_long], final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    while alpha_prod_t.ndim < sample.ndim:
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        alpha_prod_t_prev = alpha_prod_t_prev.unsqueeze(-1)
        beta_prod_t = beta_prod_t.unsqueeze(-1)
        beta_prod_t_prev = beta_prod_t_prev.unsqueeze(-1)

    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t.sqrt() * model_output
        ) / alpha_prod_t.sqrt()
        pred_epsilon = model_output
    elif prediction_type == "v_prediction":
        pred_original_sample = (
            alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        )
        pred_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
    elif prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (
            sample - alpha_prod_t.sqrt() * pred_original_sample
        ) / beta_prod_t.sqrt()
    else:
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    variance = torch.clamp(variance, min=0.0)
    std_dev_t = noise_level * torch.sqrt(variance)
    pred_sample_direction = (
        torch.sqrt(torch.clamp(1 - alpha_prod_t_prev - std_dev_t**2, min=0.0))
        * pred_epsilon
    )
    prev_sample_mean = (
        alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
    )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    std_eps = 1e-12
    safe_std = torch.clamp(std_dev_t, min=std_eps)
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (safe_std**2))
        - torch.log(safe_std)
        - math.log(math.sqrt(2 * math.pi))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    deterministic_mask = std_dev_t.flatten(start_dim=1).max(dim=1).values <= std_eps
    log_prob = torch.where(deterministic_mask, torch.zeros_like(log_prob), log_prob)
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


@torch.no_grad()
def pipeline_with_logprob(
    self: StableDiffusionXLPipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 40,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    fused_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_fused_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: str = "pil",
    noise_level: float = 0.7,
):
    del prompt_2, negative_prompt_2

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    device = self._execution_device

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        source_embeds = (
            prompt_embeds if prompt_embeds is not None else fused_prompt_embeds
        )
        if source_embeds is None:
            raise ValueError(
                "Either `prompt` or `prompt_embeds`/`fused_prompt_embeds` must be provided."
            )
        batch_size = source_embeds.shape[0]

    if prompt_embeds is None or pooled_prompt_embeds is None:
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            [self.text_encoder, self.text_encoder_2],
            [self.tokenizer, self.tokenizer_2],
            prompt,
            max_sequence_length=77,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance and (
        negative_prompt_embeds is None or negative_pooled_prompt_embeds is None
    ):
        negative_prompt = negative_prompt if negative_prompt is not None else ""
        negative_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt(
            [self.text_encoder, self.text_encoder_2],
            [self.tokenizer, self.tokenizer_2],
            [negative_prompt] * batch_size
            if isinstance(negative_prompt, str)
            else negative_prompt,
            max_sequence_length=77,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )

    add_time_ids = get_add_time_ids(
        self,
        original_size=(height, width),
        crops_coords_top_left=(0, 0),
        target_size=(height, width),
        dtype=prompt_embeds.dtype,
        device=device,
    ).repeat(batch_size * num_images_per_prompt, 1)

    if do_classifier_free_guidance:
        if fused_prompt_embeds is None:
            fused_prompt_embeds = prompt_embeds
        if negative_fused_prompt_embeds is None:
            negative_fused_prompt_embeds = negative_prompt_embeds
        encoder_hidden_states = torch.cat(
            [negative_fused_prompt_embeds, fused_prompt_embeds], dim=0
        )
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )
        # 时间条件向量需要与 CFG 扩展后的批次保持严格对齐。
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    else:
        encoder_hidden_states = (
            fused_prompt_embeds if fused_prompt_embeds is not None else prompt_embeds
        )
        add_text_embeds = pooled_prompt_embeds

    unet_for_config = self.unet.module if hasattr(self.unet, "module") else self.unet
    num_channels_latents = unet_for_config.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        encoder_hidden_states.dtype,
        device,
        generator,
        latents,
    ).float()

    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )

    all_latents = [latents]
    all_log_probs = []
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                },
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents, log_prob, _, _ = ddim_step_with_logprob(
                self.scheduler,
                noise_pred.float(),
                t.unsqueeze(0).repeat(latents.shape[0]),
                latents.float(),
                noise_level=noise_level,
                generator=generator,
            )
            all_latents.append(latents)
            all_log_probs.append(log_prob)

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    latents = latents / self.vae.config.scaling_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)
    self.maybe_free_model_hooks()
    return image, all_latents, all_log_probs


def compute_text_embeddings(
    prompt,
    text_encoders,
    tokenizers,
    max_sequence_length,
    device,
    emoenhance=None,
    emotions=None,
):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders,
            tokenizers,
            prompt,
            max_sequence_length,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)

        if emoenhance is not None and emotions is not None:
            # emoenhance is EmotionMapper and needs emotion strings.
            prompt_embeds = emoenhance(prompt_embeds, emotions, intensity=0.3)

    return prompt_embeds, pooled_prompt_embeds


def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    计算奖励标准差为 0 的唯一样本提示词占比。

    Args:
        prompts: 提示词列表。
        gathered_rewards: 奖励字典，必须包含键 'ori_avg'。

    Returns:
        zero_std_ratio: 标准差为 0 的提示词占比。
        prompt_std_devs: 全部唯一样本提示词的平均标准差。
    """
    # 将提示词列表转换为 NumPy 数组
    prompt_array = np.array(prompts)

    # 获取唯一提示词及其分组信息
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )

    # 将奖励按提示词分组
    grouped_rewards = gathered_rewards["ori_avg"][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)

    # 计算每组标准差
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])

    # 计算标准差为 0 的比例
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)

    return zero_std_ratio, prompt_std_devs.mean()


def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # 使用稳定哈希（SHA256）生成与提示词绑定的整数种子
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(
            hash_digest[:4], "big"
        )  # 取前 4 字节作为种子的一部分
        seed = (base_seed + prompt_hash_int) % (2**31)  # 保证种子在合法范围
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def resolve_lora_target_modules(unet):
    """
    根据当前 UNet 的真实模块名自动选择 LoRA 目标层，避免不同版本命名差异导致注入失败。
    """
    module_names = [name for name, _ in unet.named_modules()]

    # SDXL（diffusers 常见实现）注意力线性层命名
    sdxl_style = ["to_q", "to_k", "to_v", "to_out.0"]
    if any(name.endswith("to_q") for name in module_names):
        missing = [
            m for m in sdxl_style if not any(name.endswith(m) for name in module_names)
        ]
        if missing:
            raise ValueError(f"检测到 SDXL 风格注意力层，但缺少目标层: {missing}。")
        return sdxl_style

    # 兜底：部分实现可能使用 q_proj/k_proj/v_proj/out_proj
    proj_style = ["q_proj", "k_proj", "v_proj", "out_proj"]
    if any(name.endswith("q_proj") for name in module_names):
        missing = [
            m for m in proj_style if not any(name.endswith(m) for name in module_names)
        ]
        if missing:
            raise ValueError(f"检测到 *_proj 风格注意力层，但缺少目标层: {missing}。")
        return proj_style

    preview = module_names[:30]
    raise ValueError(
        f"无法从当前 UNet 自动识别 LoRA 目标层命名风格。模块名前30项示例: {preview}"
    )


def compute_log_prob(
    unet,
    pipeline,
    sample,
    j,
    embeds,
    pooled_embeds,
    add_time_ids,
    config,
):
    model_input = sample["latents"][:, j]
    timestep = sample["timesteps"][:, j]

    if config.train.cfg:
        latent_model_input = torch.cat([model_input] * 2)
        latent_model_input = pipeline.scheduler.scale_model_input(
            latent_model_input, timestep[0]
        )
        noise_pred = unet(
            latent_model_input,
            torch.cat([timestep] * 2),
            encoder_hidden_states=embeds,
            added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
    else:
        latent_model_input = pipeline.scheduler.scale_model_input(
            model_input, timestep[0]
        )
        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=embeds,
            added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
            return_dict=False,
        )[0]

    prev_sample, log_prob, prev_sample_mean, std_dev_t = ddim_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        timestep,
        model_input.float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def eval(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    accelerator,
    global_step,
    reward_fn,
    executor,
    autocast,
    num_train_timesteps,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""],
        text_encoders,
        tokenizers,
        max_sequence_length=128,
        device=accelerator.device,
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(
        config.sample.test_batch_size, 1, 1
    )
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(
        config.sample.test_batch_size, 1
    )

    all_rewards = defaultdict(list)

    for test_batch in tqdm(
        test_dataloader,
        desc="Eval: ",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        prompts, prompt_metadata = test_batch

        emotions = []
        for meta in prompt_metadata:
            emotions.append(meta.get("emotion", "amusement") if meta else "amusement")

        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=128,
            device=accelerator.device,
            emoenhance=getattr(pipeline, "emoenhance", None),
            emotions=emotions,
        )

        # 最后一个批次可能不足满批
        if len(prompt_embeds) < len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[: len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[
                : len(prompt_embeds)
            ]
        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds[
                        : len(prompt_embeds)
                    ],
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[
                        : len(prompt_embeds)
                    ],
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                )
        rewards = executor.submit(
            reward_fn, images, prompts, prompt_metadata, only_strict=False
        )
        # 主动让出调度，确保奖励计算线程尽快启动
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = (
                accelerator.gather(torch.as_tensor(value, device=accelerator.device))
                .cpu()
                .numpy()
            )
            all_rewards[key].append(rewards_gather)

    last_batch_images_gather = (
        accelerator.gather(torch.as_tensor(images, device=accelerator.device))
        .cpu()
        .numpy()
    )
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=tokenizers[0].model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = (
        accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    )
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = (
            accelerator.gather(torch.as_tensor(value, device=accelerator.device))
            .cpu()
            .numpy()
        )

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [
                last_batch_prompts_gather[index] for index in sample_indices
            ]
            sampled_rewards = [
                {
                    k: last_batch_rewards_gather[k][index]
                    for k in last_batch_rewards_gather
                }
                for index in sample_indices
            ]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(
                                f"{k}: {v:.2f}" for k, v in reward.items() if v != -10
                            ),
                        )
                        for idx, (prompt, reward) in enumerate(
                            zip(sampled_prompts, sampled_rewards)
                        )
                    ],
                    **{
                        f"eval_reward_{key}": np.mean(value[value != -10])
                        for key, value in all_rewards.items()
                    },
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_ckpt(
    save_dir,
    transformer,
    global_step,
    accelerator,
    ema,
    transformer_trainable_parameters,
    config,
    llm_config=None,
    optimizer=None,
    epoch=0,
):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)

        # 记录 LLM 配置信息，便于推理时重建
        if llm_config is not None:
            import json as _json

            with open(os.path.join(save_root, "llm_config.json"), "w") as f:
                _json.dump(dict(llm_config), f, indent=2)

        # 保存训练状态（optimizer、EMA、epoch、global_step）
        training_state = {
            "epoch": epoch,
            "global_step": global_step,
        }
        if optimizer is not None:
            training_state["optimizer_state_dict"] = optimizer.state_dict()
        if config.train.ema and ema is not None:
            training_state["ema_state_dict"] = ema.state_dict()
        torch.save(training_state, os.path.join(save_root, "training_state.pt"))

        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

    accelerator.wait_for_everyone()


def find_latest_checkpoint(save_dir):
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-", 1)[1])
                full_path = os.path.join(ckpt_dir, name)
                if os.path.isfile(os.path.join(full_path, "training_state.pt")):
                    candidates.append((step, full_path))
            except ValueError:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_ckpt(
    resume_path,
    transformer,
    accelerator,
    ema=None,
    transformer_trainable_parameters=None,
    config=None,
    optimizer=None,
):
    logger.info(f"Resuming from checkpoint: {resume_path}")

    # 1) 加载 LoRA 权重
    lora_path = os.path.join(resume_path, "lora")
    if os.path.isdir(lora_path):
        adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
        if os.path.isfile(adapter_path):
            lora_state = load_file(adapter_path, device=str(accelerator.device))
            set_peft_model_state_dict(
                unwrap_model(transformer, accelerator), lora_state
            )
            logger.info(f"  Loaded LoRA weights from {adapter_path}")
        else:
            logger.warning(f"  LoRA weights not found at {adapter_path}")
    else:
        logger.warning(f"  LoRA directory not found at {lora_path}")

    # 2) 加载训练状态
    state_path = os.path.join(resume_path, "training_state.pt")
    epoch = 0
    global_step = 0
    if os.path.isfile(state_path):
        training_state = torch.load(state_path, map_location="cpu")
        epoch = training_state.get("epoch", 0)
        global_step = training_state.get("global_step", 0)

        # 恢复 optimizer
        if optimizer is not None and "optimizer_state_dict" in training_state:
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
            logger.info("  Loaded optimizer state")

        # 恢复 EMA
        if (
            config is not None
            and config.train.ema
            and ema is not None
            and "ema_state_dict" in training_state
        ):
            ema.load_state_dict(training_state["ema_state_dict"])
            logger.info("  Loaded EMA state")

        logger.info(f"  Resumed at epoch={epoch}, global_step={global_step}")
    else:
        logger.warning(
            f"  training_state.pt not found at {state_path}, starting from scratch."
        )

    return epoch, global_step


def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
        * num_train_timesteps,
    )
    if accelerator.is_main_process:
        wandb.init(project="flow_grpo", experiment_name=config.run_name)
    logger.info(f"\n{config}")

    # 设置随机种子（启用设备相关模式可确保不同设备拿到不同提示词）
    set_seed(config.seed, device_specific=True)

    # 加载 SDXL。显式切换到 DDIM，使每个反推步都有可解析的方差与对数概率
    pipeline = StableDiffusionXLPipeline.from_pretrained(config.pretrained.model)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # 加载情感增强模块 EmotionEnhance
    emotion_enhance_path = (
        config.train.emoenhance_path
        if hasattr(config.train, "emoenhance_path")
        else "./results/stage2"
    )

    d_model = (
        pipeline.text_encoder.config.hidden_size
        + pipeline.text_encoder_2.config.hidden_size
    )
    emoenhance = EmotionMapper(input_dim=d_model, emo_bank=EmotionEmbeddingBank())
    model_path = os.path.join(emotion_enhance_path, "model.safetensors")
    if os.path.exists(model_path):
        state_dict = load_file(model_path)
        emoenhance.load_state_dict(state_dict)
        logger.info(f"[EmotionEnhancer] Loaded model from {model_path}")
    else:
        logger.warning(f"[EmotionEnhancer] No model found at {model_path}")
    emoenhance.eval()
    emoenhance.requires_grad_(False)

    pipeline.emoenhance = emoenhance

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
    if hasattr(pipeline, "watermark"):
        pipeline.watermark = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device)
    pipeline.emoenhance.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        target_modules = resolve_lora_target_modules(pipeline.unet)
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.unet = PeftModel.from_pretrained(
                pipeline.unet, config.train.lora_path
            )
            pipeline.unet.set_adapter("default")
        else:
            pipeline.unet = get_peft_model(pipeline.unet, transformer_lora_config)

    transformer = pipeline.unet
    transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )

    ema = EMAModuleWrapper(
        transformer_trainable_parameters,
        decay=0.9,
        update_step_interval=8,
        device=accelerator.device,
    )

    # 在 Ampere GPU 上开启 TF32 以提升训练速度。
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # 构造提示词与奖励函数
    reward_fn = getattr(flow_grpo.rewards, "multi_score")(
        accelerator.device, config.reward_fn
    )
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(
        accelerator.device, config.reward_fn
    )

    # ============ 构建数据集与数据加载器 ============
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")

        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=8,
            collate_fn=TextPromptDataset.collate_fn,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=4 if config.use_llm_prompt else 8,
        )

    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")

        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError(
            f"Unsupported prompt_fn: {config.prompt_fn}. "
            "Supported: 'general_ocr', 'geneval', or set config.use_llm_prompt=True."
        )

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""],
        text_encoders,
        tokenizers,
        max_sequence_length=128,
        device=accelerator.device,
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(
        config.sample.train_batch_size, 1, 1
    )
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(
        config.sample.train_batch_size, 1
    )
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(
        config.train.batch_size, 1
    )

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    object.__setattr__(train_dataloader, "batch_size", config.sample.train_batch_size)
    if (
        hasattr(test_dataloader, "batch_sampler")
        and test_dataloader.batch_sampler is not None
    ):
        object.__setattr__(test_dataloader, "batch_size", config.sample.test_batch_size)

    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader, test_dataloader
    )
    pipeline.unet = transformer

    # 线程池用于异步执行回调（如远程推理类奖励），减少主训练阻塞。
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # 开始训练
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0

    # ============ 断点续练：从 checkpoint 恢复 ============
    if config.resume_from:
        if config.resume_from == "latest":
            resume_path = find_latest_checkpoint(config.save_dir)
        else:
            resume_path = config.resume_from
        if resume_path is not None and os.path.isdir(resume_path):
            epoch, global_step = load_ckpt(
                resume_path,
                transformer=transformer,
                accelerator=accelerator,
                ema=ema,
                transformer_trainable_parameters=transformer_trainable_parameters,
                config=config,
                optimizer=optimizer,
            )
            logger.info(
                f"Checkpoint loaded. Continuing from epoch={epoch}, global_step={global_step}"
            )
        else:
            logger.warning(
                f"resume_from='{config.resume_from}' but no valid checkpoint found. "
                "Training from scratch."
            )

    train_iter = iter(train_dataloader)

    while True:
        #################### 评估 ####################
        pipeline.unet.eval()
        if epoch % config.eval_freq == 0:
            eval_batches_limit = getattr(config, "eval_batches_limit", 4)
            test_batches = list(test_dataloader)
            random.shuffle(test_batches)
            sampled_test_dataloader = itertools.islice(test_batches, eval_batches_limit)

            eval(
                pipeline,
                sampled_test_dataloader,  ## test_dataloader
                text_encoders,
                tokenizers,
                config,
                accelerator,
                global_step,
                eval_reward_fn,
                executor,
                autocast,
                num_train_timesteps,
                ema,
                transformer_trainable_parameters,
            )
        if epoch % config.save_freq == 0 and epoch > 0:
            save_ckpt(
                config.save_dir,
                transformer,
                global_step,
                accelerator,
                ema,
                transformer_trainable_parameters,
                config,
                llm_config=config.llm if config.use_llm_prompt else None,
                optimizer=optimizer,
                epoch=epoch,
            )

        #################### 采样 ####################
        pipeline.unet.eval()

        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            # ---- 解包 batch ----
            prompts, prompt_metadata = next(train_iter)

            emotions = []
            for meta in prompt_metadata:
                emotions.append(
                    meta.get("emotion", "amusement") if meta else "amusement"
                )

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=accelerator.device,
                emoenhance=getattr(pipeline, "emoenhance", None),
                emotions=emotions,
            )

            add_time_ids = get_add_time_ids(
                pipeline,
                original_size=(config.resolution, config.resolution),
                crops_coords_top_left=(0, 0),
                target_size=(config.resolution, config.resolution),
                dtype=prompt_embeds.dtype,
                device=accelerator.device,
            ).repeat(len(prompts), 1)
            # 保存 add_time_ids，确保训练阶段复用与采样阶段完全一致的条件输入。
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=tokenizers[0].model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # 采样
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch * 10000 + i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    images, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[
                            : len(prompt_embeds)
                        ],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[
                            : len(prompt_embeds)
                        ],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator,
                    )

            latents = torch.stack(latents, dim=1)
            log_probs = torch.stack(log_probs, dim=1)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )

            # 异步计算奖励
            rewards = executor.submit(
                reward_fn, images, prompts, prompt_metadata, only_strict=True
            )
            # 主动让出调度，保证奖励线程尽快启动
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds_raw": pooled_prompt_embeds,
                    "add_time_ids": add_time_ids,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # 等待所有奖励计算完成
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # 合并采样结果：字典中每个键都拼成（每轮批次数 × 采样批大小, ...）
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards["avg"][i] for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(
                                zip(sampled_prompts, sampled_rewards)
                            )
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = (
            samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )
        # 跨进程汇总奖励
        gathered_rewards = {
            key: accelerator.gather(value) for key, value in samples["rewards"].items()
        }
        gathered_rewards = {
            key: value.cpu().numpy() for key, value in gathered_rewards.items()
        }
        # 记录奖励与可视化结果
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{key}": value.mean()
                        for key, value in gathered_rewards.items()
                        if "_strict_accuracy" not in key and "_accuracy" not in key
                    },
                },
                step=global_step,
            )

        if config.per_prompt_stat_tracking:
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards["avg"])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(
                prompts, gathered_rewards
            )

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards["avg"] - gathered_rewards["avg"].mean()) / (
                gathered_rewards["avg"].std() + 1e-4
            )

        advantages = torch.as_tensor(advantages)
        samples["advantages"] = advantages.reshape(
            accelerator.num_processes, -1, advantages.shape[-1]
        )[accelerator.process_index].to(accelerator.device)
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        mask = samples["advantages"].abs().sum(dim=1) != 0

        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                mask[false_indices[:num_to_change]] = True
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size = samples["timesteps"].shape[0]

        local_bs = torch.tensor([total_batch_size], device=accelerator.device)
        global_bs = accelerator.gather(local_bs)  # (num_processes,)
        min_batch_size = global_bs.min().item()

        num_batches = config.sample.num_batches_per_epoch
        usable = (min_batch_size // num_batches) * num_batches

        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_batch_size": usable // num_batches if usable > 0 else 0,
                    "min_batch_across_ranks": min_batch_size,
                    "local_valid_samples": total_batch_size,
                },
                step=global_step,
            )

        if usable == 0:
            logger.warning(
                f"Epoch {epoch}: after sync, usable batch size is 0 "
                f"(local={total_batch_size}, global_min={min_batch_size}), "
                f"skipping training."
            )
            epoch += 1
            continue

        if usable < total_batch_size:
            samples = {k: v[:usable] for k, v in samples.items()}
        total_batch_size = usable

        num_timesteps = samples["timesteps"].shape[1]
        assert num_timesteps == config.sample.num_steps

        #################### 训练 ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # 按批维打乱样本
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # 重新切分批次以便训练
            samples_batched = {
                k: v.reshape(
                    -1,
                    total_batch_size // config.sample.num_batches_per_epoch,
                    *v.shape[1:],
                )
                for k, v in samples.items()
            }

            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            pipeline.unet.train()

            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                train_timesteps = [
                    step_index for step_index in range(num_train_timesteps)
                ]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        if config.train.cfg:
                            bs = sample["prompt_embeds"].shape[0]
                            embeds = torch.cat(
                                [train_neg_prompt_embeds[:bs], sample["prompt_embeds"]]
                            )
                            pooled_embeds = torch.cat(
                                [
                                    train_neg_pooled_prompt_embeds[:bs],
                                    sample["pooled_prompt_embeds_raw"],
                                ]
                            )
                            add_time_ids = torch.cat(
                                [sample["add_time_ids"], sample["add_time_ids"]], dim=0
                            )
                        else:
                            embeds = sample["prompt_embeds"]
                            pooled_embeds = sample["pooled_prompt_embeds_raw"]
                            add_time_ids = sample["add_time_ids"]

                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = (
                                compute_log_prob(
                                    transformer,
                                    pipeline,
                                    sample,
                                    j,
                                    embeds,
                                    pooled_embeds,
                                    add_time_ids,
                                    config,
                                )
                            )
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    model_with_adapter = (
                                        transformer.module
                                        if hasattr(transformer, "module")
                                        else transformer
                                    )
                                    disable_adapter_ctx = (
                                        model_with_adapter.disable_adapter()
                                        if hasattr(
                                            model_with_adapter, "disable_adapter"
                                        )
                                        else contextlib.nullcontext()
                                    )
                                    with disable_adapter_ctx:
                                        _, _, prev_sample_mean_ref, _ = (
                                            compute_log_prob(
                                                transformer,
                                                pipeline,
                                                sample,
                                                j,
                                                embeds,
                                                pooled_embeds,
                                                add_time_ids,
                                                config,
                                            )
                                        )

                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(
                            torch.maximum(unclipped_loss, clipped_loss)
                        )
                        if config.train.beta > 0:
                            kl_loss = (
                                (prev_sample_mean - prev_sample_mean_ref) ** 2
                            ).mean(dim=(1, 2, 3), keepdim=True) / (2 * std_dev_t**2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean((ratio - 1.0 > config.train.clip_range).float())
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean((1.0 - ratio > config.train.clip_range).float())
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
        epoch += 1


if __name__ == "__main__":
    app.run(main)
