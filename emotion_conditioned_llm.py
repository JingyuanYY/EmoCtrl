import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List

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
emotion2id = {name: i for i, name in enumerate(emotion_list)}


class EmotionEmbeddingBank(nn.Module):
    """
    情感嵌入库 nn.Embedding(num_emotions, shared_dim * token_len)
    - 提供统一的情感向量 shared_dim
    - 不同模型侧通过线性投影适配到各自 hidden_size
    """

    def __init__(
        self, num_emotions: int = 8, shared_dim: int = 1024, token_len: int = 1
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.shared_dim = shared_dim
        self.token_len = token_len
        self.embedding = nn.Embedding(num_emotions, shared_dim * token_len)

    @torch.no_grad()
    def names_to_indices(self, emotion: Union[str, List[str]]) -> torch.LongTensor:
        if isinstance(emotion, str):
            return torch.tensor([emotion2id[emotion]], dtype=torch.long)
        return torch.tensor([emotion2id[e] for e in emotion], dtype=torch.long)

    def indices_from_input(
        self,
        emotion: Union[str, int, List[Union[str, int]], Dict[str, float], torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.LongTensor:
        # Tensor 直接返回
        if isinstance(emotion, torch.Tensor):
            assert emotion.dim() == 1 and emotion.numel() == batch_size
            return emotion.to(device=device, dtype=torch.long)
        # dict 取 argmax
        if isinstance(emotion, dict):
            assert len(emotion) > 0
            best = max(emotion.items(), key=lambda kv: kv[1])[0]
            if isinstance(best, str):
                idx_val = emotion2id[best]
            else:
                idx_val = int(best)
            assert 0 <= idx_val < self.num_emotions
            return torch.full((batch_size,), idx_val, device=device, dtype=torch.long)
        # 单个 str
        if isinstance(emotion, str):
            idx_val = emotion2id[emotion]
            return torch.full((batch_size,), idx_val, device=device, dtype=torch.long)
        # 单个 int
        if isinstance(emotion, int):
            assert 0 <= emotion < self.num_emotions
            return torch.full(
                (batch_size,), int(emotion), device=device, dtype=torch.long
            )
        # 列表
        if isinstance(emotion, list):
            assert len(emotion) == batch_size
            out: List[int] = []
            for e in emotion:
                if isinstance(e, str):
                    out.append(emotion2id[e])
                else:
                    ei = int(e)
                    assert 0 <= ei < self.num_emotions
                    out.append(ei)
            return torch.tensor(out, device=device, dtype=torch.long)
        raise TypeError("Unsupported emotion type")

    def get_tokens(
        self, emo_indices: torch.LongTensor, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        B = emo_indices.size(0)
        flat = self.embedding(emo_indices)  # [B, S*K]
        K = self.token_len
        S = self.shared_dim
        tokens = flat.view(B, K, S)
        if dtype is not None:
            tokens = tokens.to(dtype)
        return tokens


class EmotionConditionedCausalLM(nn.Module):
    def __init__(
        self,
        base_lm,
        emo_bank: Optional[EmotionEmbeddingBank] = None,
    ):
        super().__init__()
        self.base_lm = base_lm
        self.emo_bank = emo_bank or EmotionEmbeddingBank().to(
            device=base_lm.device, dtype=base_lm.dtype
        )
        self.emo_proj = nn.Linear(
            self.emo_bank.shared_dim, base_lm.config.hidden_size, bias=False
        ).to(device=base_lm.device, dtype=base_lm.dtype)
        self.prompt_len = self.emo_bank.token_len

    @property
    def config(self):
        # 让 Trainer 能访问到模型配置（dtype、hidden_size等）
        return self.base_lm.config

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.base_lm, "gradient_checkpointing_enable"):
            return self.base_lm.gradient_checkpointing_enable(*args, **kwargs)
        return None  # 不支持时无操作

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_lm, "gradient_checkpointing_disable"):
            return self.base_lm.gradient_checkpointing_disable()
        return None

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        k1 = "base_lm.base_model.model.lm_head.weight"
        k2 = "base_lm.base_model.model.model.embed_tokens.original_module.weight"
        if k1 in sd and k2 in sd:
            try:
                if sd[k1].data_ptr() == sd[k2].data_ptr():
                    sd[k1] = sd[k1].clone()
            except Exception:
                sd[k1] = sd[k1].clone()
        return sd

    def generate(self, *args, **kwargs):
        return self.base_lm.generate(*args, **kwargs)

    def get_input_embeddings(self):
        return self.base_lm.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.base_lm.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        emotion_id: Optional[torch.LongTensor] = None,
        **kwargs: Dict[str, Any],
    ):
        assert input_ids is not None, "input_ids is required"
        assert emotion_id is not None, "emotion_id is required (per-sample)"

        B, T = input_ids.shape
        device = input_ids.device
        embed = self.base_lm.get_input_embeddings()
        inp_embeds = embed(input_ids)  # [B, T, D]

        # 构造前缀提示：从共享库取 [B, K, S]，再投到 LM hidden
        emo_tokens = self.emo_bank.get_tokens(
            emotion_id.to(self.emo_bank.embedding.weight.device),
            dtype=self.base_lm.dtype,
        )  # [B, K, S]
        prompt_embeds = self.emo_proj(emo_tokens)  # [B, K, D]
        P = prompt_embeds.size(1)

        # 拼接
        inputs_embeds = torch.cat([prompt_embeds, inp_embeds], dim=1)  # [B, P+T, D]

        # attention mask
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
        prefix_mask = torch.ones(B, P, dtype=attention_mask.dtype, device=device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, P+T]

        # labels 前缀置 -100
        if labels is not None:
            ignore_prefix = torch.full((B, P), -100, dtype=labels.dtype, device=device)
            labels = torch.cat([ignore_prefix, labels], dim=1)

        outputs = self.base_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        loss = outputs.loss

        return type(outputs)(
            loss=loss, **{k: v for k, v in outputs.items() if k != "loss"}
        )
