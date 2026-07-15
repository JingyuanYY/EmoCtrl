import os
import argparse
from collections import OrderedDict
from typing import List
import json

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from utils import emotion_list, emotion2id

with open("./data/test_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
infer_elem_list = data["element"]
infer_caption_list = data["caption"]
infer_elem_list = sorted(list(set(infer_elem_list)))
infer_caption_list = sorted(list(set(infer_caption_list)))
uni_content_list = infer_elem_list + infer_caption_list
id2content = {i: c for i, c in enumerate(uni_content_list)}
content2id = {c: i for i, c in enumerate(uni_content_list)}

content_elements = [f"a photo of a {el}" for el in sorted(list(set(infer_elem_list)))]
content_captions = sorted(list(set(infer_caption_list)))


# --- emotion classification 定义 ---
class Classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# --- emotion classification 模型加载 ---
def load_emotion_classifier(cls_path, clip_path, device, input_dim=768, num_classes=8):
    state = torch.load(cls_path, map_location="cpu")
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:] if k.startswith("module.") else k
            new_state[name] = v
        state = new_state
    model = Classifier(input_dim, num_classes)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# --- content classification ---
def classify_content(
    images: List[Image.Image],
    class_texts: List[str],
    processor,
    model,
    device,
    top_k: int = 1,
):
    if not images:
        return []
    inputs = processor(
        text=class_texts, images=images, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        all_probs = F.softmax(logits_per_image, dim=1)
        top_k_probs, top_k_indices = torch.topk(all_probs, top_k, dim=1)

    predicted_classes_topk = []
    for indices in top_k_indices:
        predicted_classes_topk.append([class_texts[i] for i in indices])
    return predicted_classes_topk, all_probs.cpu()


# --- 收集和解析图片 ---
def collect_image_paths(image_dir: str) -> List[str]:
    image_paths = []
    ablation_paths = []
    ground_truths = []
    ablation_ground_truths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(("jpg", "png", "jpeg")):
                try:
                    parts = os.path.splitext(file)[0].split("-")
                    if parts[0] == "sd":
                        gt_content, gt_emotion = parts[1], parts[2]
                        ablation_paths.append(os.path.join(root, file))
                        ablation_ground_truths.append(
                            {
                                "emotion": gt_emotion,
                                "content": gt_content,
                                "is_element": "." not in gt_content,
                            }
                        )
                        continue
                    gt_emotion, gt_content_id = parts[2], parts[0]
                    gt_content = id2content[int(gt_content_id)]
                    is_element = "." not in gt_content

                    if gt_emotion in emotion2id:
                        image_paths.append(os.path.join(root, file))
                        ground_truths.append(
                            {
                                "emotion": gt_emotion,
                                "content": gt_content,
                                "is_element": is_element,
                            }
                        )
                    else:
                        print(
                            f"Skipping {file}: emotion '{gt_emotion}' not in label list."
                        )
                except IndexError:
                    print(f"Skipping {file}: cannot parse filename.")
    return image_paths, ground_truths, ablation_paths, ablation_ground_truths


def compute_joint_metrics(
    args,
    clip_processor,
    clip_model,
    emotion_classifier,
    device,
    image_paths,
    ground_truths,
    results_file=None,
):
    # --- 评估 ---
    total_processed = 0
    content_correct_count = 0
    emotion_correct_count = 0
    joint_correct_count = 0

    # 分别统计 element 和 caption 的计数器
    element_total_count = 0
    element_correct_count = 0
    caption_total_count = 0
    caption_correct_count = 0

    # 预先计算候选文本的特征（避免每次重复编码）
    with torch.no_grad():
        if content_elements:
            elem_text_inputs = clip_processor(
                text=content_elements, return_tensors="pt", padding=True
            ).to(device)
            elem_text_feats = clip_model.get_text_features(**elem_text_inputs)
            elem_text_feats = F.normalize(elem_text_feats, p=2, dim=-1)  # [N_elem, D]
        else:
            elem_text_feats = None
        if content_captions:
            cap_text_inputs = clip_processor(
                text=content_captions, return_tensors="pt", padding=True
            ).to(device)
            cap_text_feats = clip_model.get_text_features(**cap_text_inputs)
            cap_text_feats = F.normalize(cap_text_feats, p=2, dim=-1)  # [N_cap, D]
        else:
            cap_text_feats = None

    for i in tqdm(
        range(0, len(image_paths), args.batch_size), desc="Evaluating Joint Metrics"
    ):
        batch_paths = image_paths[i : i + args.batch_size]
        batch_gts = ground_truths[i : i + args.batch_size]

        images = []
        valid_gts = []
        for path, gt in zip(batch_paths, batch_gts):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_gts.append(gt)
            except Exception as e:
                print(f"Error loading image {path}: {e}")

        if not images:
            continue

        # --- 情感预测 ---
        inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(
            device
        )
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features_norm = F.normalize(image_features, p=2, dim=-1)
            emotion_logits = emotion_classifier(image_features)
            emotion_probs = F.softmax(emotion_logits, dim=1)
            emotion_preds = torch.argmax(emotion_probs, dim=1).cpu().numpy()

        if elem_text_feats is not None:
            scores_elem = torch.matmul(image_features_norm, elem_text_feats.T)
            probs_elem = F.softmax(scores_elem, dim=1)  # 内容概率分布
            topk_vals_elem, topk_idx_elem = torch.topk(
                probs_elem, k=min(args.top_k_content, probs_elem.size(1)), dim=1
            )

        gt_texts = []
        for gt in valid_gts:
            t = gt["content"]
            if gt["is_element"]:
                t = f"a photo of a {t}"
            gt_texts.append(t)
        with torch.no_grad():
            gt_text_inputs = clip_processor(
                text=gt_texts, return_tensors="pt", padding=True
            ).to(device)
            gt_text_feats = clip_model.get_text_features(**gt_text_inputs)
            gt_text_feats = F.normalize(gt_text_feats, p=2, dim=-1)
            clip_scores_batch = (
                (image_features_norm * gt_text_feats).sum(dim=1).cpu().tolist()
            )

        for j, gt in enumerate(valid_gts):
            # 检查情感是否正确
            pred_emotion_label = emotion_list[emotion_preds[j]]
            emotion_correct = pred_emotion_label == gt["emotion"]
            if emotion_correct:
                emotion_correct_count += 1

            clip_score = clip_scores_batch[j]

            content_correct = False
            if gt["is_element"]:
                topk_indices = topk_idx_elem[j].cpu().tolist()
                topk_elements = [
                    content_elements[idx].replace("a photo of a ", "")
                    for idx in topk_indices
                ]
                if gt["content"] in topk_elements:
                    content_correct = True
                    element_correct_count += 1
                element_total_count += 1
            else:
                if clip_score > args.clip_score_threshold:
                    content_correct = True
                    caption_correct_count += 1
                caption_total_count += 1

            if content_correct:
                content_correct_count += 1

            # 联合正确计数
            if emotion_correct and content_correct:
                joint_correct_count += 1
        total_processed += len(images)

    # --- 计算并打印最终指标 ---
    if total_processed == 0:
        print("No images were processed. Exiting.")
        return

    joint_precision = (joint_correct_count / total_processed) * 100
    content_accuracy = (content_correct_count / total_processed) * 100
    emotion_accuracy = (emotion_correct_count / total_processed) * 100
    element_accuracy = (
        (element_correct_count / element_total_count) * 100
        if element_total_count > 0
        else 0
    )
    caption_accuracy = (
        (caption_correct_count / caption_total_count) * 100
        if caption_total_count > 0
        else 0
    )

    print("\n--- Joint Evaluation Results ---")
    print(f"Target dir: {args.image_dir}")
    print(f"Total Images Processed: {total_processed}")
    print("-" * 30)
    print(f"Emotion Accuracy: {emotion_accuracy:.2f}%")
    print(f"Overall Content Accuracy: {content_accuracy:.2f}%")
    if element_total_count > 0:
        print(
            f"  - Element Accuracy (Top-{args.top_k_content}): {element_accuracy:.2f}% ({element_correct_count}/{element_total_count})"
        )
    if caption_total_count > 0:
        print(
            f"  - Caption Accuracy (CLIP>{args.clip_score_threshold}): {caption_accuracy:.2f}% ({caption_correct_count}/{caption_total_count})"
        )

    print("-" * 30)
    print(f"Joint Precision (Content=True AND Emotion=True): {joint_precision:.2f}%")
    print("-" * 30)
    # --- 将结果写入文件 ---
    if results_file is not None:
        with open(results_file, "w", encoding="utf-8") as f:
            f.write("--- Joint Evaluation Results ---\n")
            f.write(f"Target dir: {args.image_dir}\n")
            f.write(f"Total Images Processed: {total_processed}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Emotion Accuracy: {emotion_accuracy:.2f}%\n")
            f.write(f"Overall Content Accuracy: {content_accuracy:.2f}%\n")
            if element_total_count > 0:
                f.write(
                    f"  - Element Accuracy (Top-{args.top_k_content}): {element_accuracy:.2f}% ({element_correct_count}/{element_total_count})\n"
                )
            if caption_total_count > 0:
                f.write(
                    f"  - Caption Accuracy (CLIP>{args.clip_score_threshold}): {caption_accuracy:.2f}% ({caption_correct_count}/{caption_total_count})\n"
                )
            f.write("-" * 30 + "\n")
            f.write(
                f"Joint Precision (Content=True AND Emotion=True): {joint_precision:.2f}%\n"
            )
            f.write("-" * 30 + "\n")

        print(f"Results written to {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Joint Precision and Joint Score"
    )
    parser.add_argument("--image_dir", default="", help="Directory containing images.")
    parser.add_argument("--cls_path", default="", help="Path to the emotion classifier")
    parser.add_argument("--clip_path", default="", help="Path to the CLIP model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cuda", help="Device to run the models on")
    parser.add_argument("--top_k_content", type=int, default=5, help="Top-K for CLIP-A")
    parser.add_argument(
        "--clip_score_threshold", type=float, default=0.15, help="Threshold for CLIP-A"
    )
    args = parser.parse_args()
    print(f"image_dir: {args.image_dir}")

    device = torch.device(args.device)

    # --- 加载模型 ---
    print("Loading models...")
    clip_processor = CLIPProcessor.from_pretrained(args.clip_path)
    clip_model = CLIPModel.from_pretrained(args.clip_path).to(device)
    clip_model.eval()
    emotion_classifier = load_emotion_classifier(args.cls_path, args.clip_path, device)
    print("Models loaded.")

    image_paths, ground_truths, ablation_paths, ablation_ground_truths = (
        collect_image_paths(args.image_dir)
    )

    compute_joint_metrics(
        args,
        clip_processor,
        clip_model,
        emotion_classifier,
        device,
        image_paths,
        ground_truths,
        results_file=os.path.join(args.image_dir, "joint_evaluation_results.txt"),
    )
    compute_joint_metrics(
        args,
        clip_processor,
        clip_model,
        emotion_classifier,
        device,
        ablation_paths,
        ablation_ground_truths,
        results_file=os.path.join(
            args.image_dir, "joint_evaluation_results-ablation.txt"
        ),
    )


if __name__ == "__main__":
    main()
