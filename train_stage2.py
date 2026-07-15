import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from tqdm import tqdm
import random
import numpy as np

from emotion_conditioned_llm import EmotionEmbeddingBank
from emotion_enhancement import EmotionMapper
import swanlab

logger = get_logger(__name__)


# Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file,
        tokenizer,
        tokenizer_2,
        size=1024,
        center_crop=True,
        t_drop_rate=0.05,
        i_drop_rate=0.05,
        ti_drop_rate=0.05,
        image_root_path="",
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.clip_image_processor = CLIPImageProcessor()

        self.prompt_template = (
            "Based on the <elem>, give me a prompt to generate an image."
        )

        with open("./data/emoset_plus.json", "r", encoding="utf-8") as f:
            self.all_label = json.load(f)
        print(f"Dataset size: {len(self.data)}")

    def get_llm_response(self, image_file):
        image_id = image_file.split("/")[-1].split(".")[0]
        label = self.all_label.get(image_id, {})
        raw_elements_str = label.get("Objective Elements", "").strip("[]")
        raw_elements = (
            [e.strip() for e in raw_elements_str.split(", ") if e.strip()]
            if raw_elements_str
            else []
        )
        elems = ", ".join(raw_elements)
        prompt = self.prompt_template.replace("<elem>", elems)
        return prompt

    def __getitem__(self, idx):
        item = self.data[idx]
        if type(item["id"]) is float:
            return self.__getitem__(random.randint(0, len(self.data) - 1))
        image_id = item["id"].split("__")[0]
        emotion = image_id.split("_")[0]
        image_file = os.path.join(emotion, image_id + ".jpg")
        text = item["llm_text"]

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))

        llm_prompt = self.get_llm_response(image_file)

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])

        image_tensor = self.transform(raw_image.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])

        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left])

        clip_image = self.clip_image_processor(
            images=raw_image, return_tensors="pt"
        ).pixel_values

        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
            "emotion": emotion,
            "llm_prompt": llm_prompt,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat(
        [example["text_input_ids_2"] for example in data], dim=0
    )
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack(
        [example["crop_coords_top_left"] for example in data]
    )
    target_size = torch.stack([example["target_size"] for example in data])
    emotion = [example["emotion"] for example in data]
    llm_prompt = [example["llm_prompt"] for example in data]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        "emotion": emotion,
        "llm_prompt": llm_prompt,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="EmoEnhance SDXL training.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/mnt/d/models/stable-diffusion-xl-base-1.0",
        help="Path to the pretrained SDXL model.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default="./data/prefill_llm_prompts.jsonl",
        help="Path to the training data JSON file.",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="/mnt/d/data/EmoSet_v5_train-test-val/image",
        help="Path to the training data root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/stage2",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--logging_dir", type=str, default="logs", help="Path to the logging directory."
    )
    parser.add_argument(
        "--resolution", type=int, default=1024, help="The resolution for input images."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=5, help="Number of training epochs."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size for training."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for the data loader.",
    )
    parser.add_argument(
        "--save_steps", type=int, default=2000, help="Number of steps between saves."
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"]
    )
    parser.add_argument("--report_to", type=str, default="swanlab")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    accelerator.init_trackers(
        project_name="emoctrl",
        config=vars(args),
        init_kwargs={"swanlab": {"experiment_name": args.output_dir.split("/")[-1]}},
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    d_model = text_encoder.config.hidden_size + text_encoder_2.config.hidden_size
    emoenhance = EmotionMapper(
        input_dim=d_model, emo_bank=EmotionEmbeddingBank(), num_heads=16
    )

    params_to_optimize = list(
        filter(lambda p: p.requires_grad, emoenhance.parameters())
    )

    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def format_params(num):
        if num >= 1e9:
            return f"{num / 1e9:.2f}B"
        elif num >= 1e6:
            return f"{num / 1e6:.2f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.2f}K"
        else:
            return str(num)

    # 打印训练的参数量
    total_params = sum(p.numel() for p in emoenhance.parameters())
    trainable_params = sum(
        p.numel() for p in emoenhance.parameters() if p.requires_grad
    )
    print(f"[emoenhance] Total parameters: {format_params(total_params)}")
    print(f"[emoenhance] Trainable parameters: {format_params(trainable_params)}")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device)  # VAE用fp32更稳
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device)

    train_dataset = MyDataset(
        args.data_json_file,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        image_root_path=args.data_root_path,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    emoenhance, optimizer, train_dataloader = accelerator.prepare(
        emoenhance, optimizer, train_dataloader
    )

    global_step = 0
    emoenhance.train()

    for epoch in range(args.num_train_epochs):
        epoch_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_main_process,
            dynamic_ncols=True,
        )
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(emoenhance):
                images = batch["images"].to(accelerator.device, dtype=torch.float32)
                B = images.size(0)
                with torch.no_grad():
                    latents = (
                        vae.encode(images).latent_dist.sample()
                        * vae.config.scaling_factor
                    )

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (B,),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                text_ids_1 = batch["text_input_ids"].to(accelerator.device)
                text_ids_2 = batch["text_input_ids_2"].to(accelerator.device)
                with torch.no_grad():
                    encoder_output = text_encoder(text_ids_1, output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(
                        text_ids_2, output_hidden_states=True
                    )
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.cat([text_embeds, text_embeds_2], dim=-1)

                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(
                    accelerator.device, dtype=weight_dtype
                )
                unet_added_cond_kwargs = {
                    "text_embeds": pooled_text_embeds,
                    "time_ids": add_time_ids,
                }

                emotion = batch["emotion"]
                emo_embeds = emoenhance(text_embeds, emotion=emotion)

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=emo_embeds,
                    added_cond_kwargs=unet_added_cond_kwargs,
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                avg_loss = (
                    accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                )
                accelerator.log({"train_loss": avg_loss}, step=global_step)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.is_main_process:
                    epoch_bar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        }
                    )
                    epoch_bar.update(1)
                    if args.report_to == "swanlab":
                        log_dict = {
                            "loss/total": avg_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                        swanlab.log(log_dict, step=global_step)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(save_path)

    accelerator.end_training()


if __name__ == "__main__":
    main()
