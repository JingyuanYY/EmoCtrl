import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

emotion_labels = [
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
]
emotion_label2idx = {e: i for i, e in enumerate(emotion_labels)}
EMOTION_GROUPS = {
    "positive": ["amusement", "awe", "contentment", "excitement"],
    "negative": ["anger", "disgust", "fear", "sadness"],
}
positive_indices = [emotion_label2idx[e] for e in EMOTION_GROUPS["positive"]]
negative_indices = [emotion_label2idx[e] for e in EMOTION_GROUPS["negative"]]


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class EmoScorer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        clip_path = "/mnt/d/models/clip-vit-large-patch14"
        cls_path = "/mnt/d/emoemo/metrics/model_6.pth"

        self.model = CLIPModel.from_pretrained(clip_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_path)
        self.classifier = self.load_emotion_classifier(cls_path, device)

    def load_emotion_classifier(self, cls_path, device, input_dim=768, num_classes=8):
        state_dict = torch.load(cls_path, map_location="cpu")

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model = Classifier(input_dim, num_classes)
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)
        return model

    @torch.no_grad()
    def __call__(self, images, targets):
        if isinstance(images, torch.Tensor) and torch.max(images) <= 1.0:
            inputs = self.processor(
                images=images, return_tensors="pt", do_rescale=False
            ).to(self.device)
        else:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        image_features = self.model.get_image_features(**inputs)

        w1 = 0.8  # 细粒度奖励权重
        w2 = 0.2  # 极性奖励权重

        scores = []
        for image_feature, target in zip(image_features, targets):
            target_idx = emotion_label2idx[target]

            logits = self.classifier(image_feature)  # [8]
            probs = torch.softmax(logits, dim=-1)  # [8]

            # A. 细粒度奖励：目标类别概率
            r_cls = probs[target_idx]  # [0,1]

            # B. 软极性奖励：目标极性的总概率质量（连续信号）
            pos_prob = probs[positive_indices].sum()  # [0,1]
            neg_prob = probs[negative_indices].sum()  # [0,1]

            if target in EMOTION_GROUPS["positive"]:
                same_polarity_prob = pos_prob
                opposite_polarity_prob = neg_prob
            else:
                same_polarity_prob = neg_prob
                opposite_polarity_prob = pos_prob

            alpha = 0.5  # 极性奖励的权重系数
            beta = 0.5  # 极性惩罚的权重系数
            polarity_reward = same_polarity_prob
            opposite_penalty = opposite_polarity_prob

            total_score = w1 * r_cls + w2 * (
                alpha * polarity_reward - beta * opposite_penalty
            )

            scores.append(total_score)
        return scores


if __name__ == "__main__":
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    emo_score = EmoScorer(device)
    images = ["assets/test.jpg", "assets/test.jpg"]
    targets = ["sadness", "amusement"]
    scores = emo_score(images, targets)
    print(scores)
