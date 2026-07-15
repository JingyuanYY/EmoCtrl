import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union
from emotion_conditioned_llm import EmotionEmbeddingBank, emotion_list


class EmotionMapper(nn.Module):
    """
    使用交叉注意力将情感信息融合到 prompt embeddings 中。
    """

    def __init__(
        self,
        input_dim,
        num_emotions: int = 8,
        token_len: int = 1,
        num_heads: int = 8,
        dropout: float = 0.05,
        emo_bank: Optional[EmotionEmbeddingBank] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_emotions = num_emotions
        self.dropout = dropout
        self.token_len = token_len
        self.emo_bank = EmotionEmbeddingBank() if emo_bank is None else emo_bank
        self.emo_proj = nn.Linear(self.emo_bank.shared_dim, input_dim, bias=False)

        self.emotion2id = {name: i for i, name in enumerate(emotion_list)}
        self.emotion_names = emotion_list

        self.emo_cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def _get_emotion_weights(
        self,
        emotion: Union[str, int, List[Union[str, int]], Dict[str, float], torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        weights = torch.zeros(batch_size, self.num_emotions, device=device, dtype=dtype)

        # Case 1: Tensor 输入
        if isinstance(emotion, torch.Tensor):
            if emotion.dim() == 2:  # [B, E] 已经是权重矩阵
                return emotion.to(device=device, dtype=dtype)
            elif emotion.dim() == 1:  # [B] 索引张量 -> 转 One-Hot
                idx = emotion.to(device=device, dtype=torch.long)
                weights.scatter_(1, idx.unsqueeze(1), 1.0)
                return weights

        # Case 2: Dict 输入 (混合情感，整批复用)
        elif isinstance(emotion, dict):
            single_weight = torch.zeros(self.num_emotions, device=device, dtype=dtype)
            for e, w in emotion.items():
                if e in self.emotion2id:
                    single_weight[self.emotion2id[e]] = w
                elif isinstance(e, int) and 0 <= e < self.num_emotions:
                    single_weight[e] = w
            return single_weight.unsqueeze(0).expand(batch_size, -1)

        # Case 3: List 输入 (智能识别)
        elif isinstance(emotion, list):
            # 策略 A: 如果长度匹配 Batch Size，优先尝试 Per-sample 模式
            # e.g. B=2, emotion=["amusement", "sadness"] -> Sample1=amusement, Sample2=sadness
            if len(emotion) == batch_size:
                try:
                    for i, e in enumerate(emotion):
                        idx = self.emotion2id[e] if isinstance(e, str) else int(e)
                        weights[i, idx] = 1.0
                    return weights
                except (KeyError, ValueError, TypeError):
                    # 如果解析失败（比如列表里不是单情感），回退到策略 B
                    weights.zero_()

            # 策略 B: Broadcast Mixing (混合并广播)
            # 适用于 len != batch_size，或者上面的解析失败
            # e.g. B=10, emotion=["amusement", "sadness"] -> 所有样本都是 0.5 amusement + 0.5 sadness
            val = 1.0 / max(len(emotion), 1)
            single_weight = torch.zeros(self.num_emotions, device=device, dtype=dtype)

            for e in emotion:
                if isinstance(e, str) and e in self.emotion2id:
                    single_weight[self.emotion2id[e]] += val
                elif isinstance(e, int) and 0 <= e < self.num_emotions:
                    single_weight[e] += val

            return single_weight.unsqueeze(0).expand(batch_size, -1)

        # Case 4: 单个 Str/Int (整批复用)
        elif isinstance(emotion, (str, int)):
            idx = self.emotion2id[emotion] if isinstance(emotion, str) else int(emotion)
            weights[:, idx] = 1.0
            return weights

        raise TypeError(f"Unsupported emotion type: {type(emotion)}")

    def forward(
        self,
        prompt_embeds: torch.Tensor,  # [B, L, D]
        emotion: Union[str, List[str], Dict[str, float], torch.Tensor],
        intensity: float = 1.0,  # 全局强度 [0,1+]，>1 也可
    ):
        assert prompt_embeds.dim() == 3, "prompt_embeds must be [B, L, D]"
        B, L, D = prompt_embeds.shape
        device, dtype = prompt_embeds.device, prompt_embeds.dtype

        weights = self._get_emotion_weights(emotion, B, device, dtype)
        all_emo_indices = torch.arange(self.num_emotions, device=device)
        all_emo_tokens = self.emo_bank.get_tokens(all_emo_indices, dtype=dtype)
        all_emo_tokens = self.emo_proj(all_emo_tokens)
        emo_tokens = torch.einsum("be, ekd -> bkd", weights, all_emo_tokens)
        emo_tokens = emo_tokens * intensity

        conditioned_embeds, _ = self.emo_cross_attn(
            query=prompt_embeds,
            key=emo_tokens,
            value=emo_tokens,
        )
        conditioned_embeds = self.dropout(conditioned_embeds)
        new_embeds = prompt_embeds + conditioned_embeds * intensity
        return new_embeds


if __name__ == "__main__":
    B, L, D = 2, 77, 2048
    intensity = 0.8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    prompt = torch.randn(B, L, D).to(device=device, dtype=dtype)

    mapper = EmotionMapper(input_dim=D).to(device=device, dtype=dtype)
    out = mapper(prompt, emotion="amusement", intensity=intensity)
    print("out shape:", out.shape)
    assert out.shape == prompt.shape, "Output shape should match input shape"

    with torch.no_grad():
        is_different = not torch.allclose(out, prompt)
        print("Output is different from input:", is_different)
        assert is_different, "Output should be different from input after attention"

        # 检查零强度
        out_zero_intensity = mapper(prompt, emotion="amusement", intensity=0.0)
        is_same = torch.allclose(out_zero_intensity, prompt, atol=1e-5)
        print("Output is same as input with intensity=0:", is_same)
        assert is_same, "Output should be same as input when intensity is 0"
