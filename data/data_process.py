import os
import json
import random
from pathlib import Path
from tqdm import tqdm
from element_filter import get_neutral_nouns_by_tfidf

# ----------------- 配置 -----------------
DATE = "1114"
RANDOM_SEED = 2025
VAL_SIZE = 100

# 路径配置
PROMPT_TEMPLATE = "Based on the <content>, give me a prompt to generate an image."

EMOSET_PLUS_PATH = "./data/emoset_plus.json"
EMOEDITSET_PLUS_PATH = "./data/emoeditset_captions-0_2.jsonl"

EMOEDITSET_DIR = "/mnt/d/data/EmoEditSet"

OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(json_file_path):
    data = {}
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            for line in f:
                result = json.loads(line)
                data[result["image_id"]] = result["response"]
    return data


# ----------------- 主流程 -----------------
def main():
    random.seed(RANDOM_SEED)
    with open(EMOSET_PLUS_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[INFO] Loaded {len(dataset)} items from emoset+.")

    neutral_nouns = set(get_neutral_nouns_by_tfidf())
    print(f"[INFO] Neutral nouns loaded: {len(neutral_nouns)}")

    results = []
    single_cnt = 0

    for image_id, label in tqdm(dataset.items(), desc="Building samples"):
        emotion = label.get("Emotion", "").strip()
        if not emotion:
            continue

        # 原始元素
        raw_elements_str = label.get("Objective Elements", "").strip("[]")
        raw_elements = (
            [e.strip() for e in raw_elements_str.split(", ") if e.strip()]
            if raw_elements_str
            else []
        )

        # 过滤中性词
        zhongxing_elements = []
        feizhongxing_elements = []
        for e in raw_elements:
            if e in neutral_nouns:
                zhongxing_elements.append(e)
            else:
                feizhongxing_elements.append(e)
        objective_caption = label.get("Objective Caption", "")

        if len(zhongxing_elements) > 1:
            for zx_ele in zhongxing_elements:
                prompt_text = PROMPT_TEMPLATE.replace("<content>", zx_ele)
                sample_id = f"{image_id}__{zx_ele}"
                results.append(
                    {
                        "id": sample_id,
                        "emotion": emotion,
                        "instruction": prompt_text,
                        "response": objective_caption,
                    }
                )
                single_cnt += 1

    print(f"[INFO] Built {len(results)} raw expand samples.")
    print(f"number of single element samples: {single_cnt}")

    # ----------------- EmoEditSet 样本合并 -----------------
    data = load_data(EMOEDITSET_PLUS_PATH)
    print(f"Loaded {len(data)} existing captions from {EMOEDITSET_PLUS_PATH}")
    # print(f"data sample: {data.keys()}")

    origin_images = []
    edited_images = []
    caption_cnt = 0
    miss_caption_cnt = 0

    for root, _, files in os.walk(EMOEDITSET_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                if "origin" in root:
                    origin_images.append(os.path.join(root, file))
                else:
                    edited_images.append(os.path.join(root, file))
    print(
        f"Found {len(origin_images)} original images and {len(edited_images)} edited images."
    )

    edited_map = {}
    for img_path in edited_images:
        try:
            # 文件名格式为 {origin_id}_{suffix}.ext
            base_name = os.path.basename(img_path)
            origin_id = base_name.split("_")[0]
            if origin_id not in edited_map:
                edited_map[origin_id] = []
            edited_map[origin_id].append(img_path)
        except IndexError:
            print(f"Warning: Could not parse origin_id from filename: {img_path}")

    emo_edit_results = []
    for image_path in tqdm(origin_images, desc="Processing EmoEditSet Images"):
        image_id = os.path.basename(image_path).split(".")[0]
        origin_caption = data.get(image_id, None)
        related_edited = edited_map.get(image_id, [])
        for edited_img in related_edited:
            edited_id = os.path.basename(edited_img).split(".")[0]
            edited_caption = data.get(edited_id, None)
            if not origin_caption or not edited_caption:
                # print(f"Warning: No caption found for image ID {image_id}, skipping.")
                miss_caption_cnt += 1
                continue
            emotion = edited_img.split("/")[-2]
            emo_edit_results.append(
                {
                    "id": edited_id,
                    "emotion": emotion,
                    "instruction": origin_caption,
                    "response": edited_caption,
                }
            )
            caption_cnt += 1
    print(f"[INFO] Built {len(emo_edit_results)} EmoEditSet samples.")
    print(f"Captions matched: {caption_cnt}, missed: {miss_caption_cnt}")
    results.extend(emo_edit_results)
    print(f"[INFO] Total samples after merging: {len(results)}")

    # 采样验证集
    if len(results) <= VAL_SIZE:
        raise ValueError("Not enough samples to carve out validation set.")
    all_indices = list(range(len(results)))
    random.shuffle(all_indices)
    val_indices = set(all_indices[:VAL_SIZE])
    val_set = [results[i] for i in val_indices]
    train_set = [results[i] for i in range(len(results)) if i not in val_indices]
    print(f"[INFO] Train: {len(train_set)}  Val: {len(val_set)}")

    # 保存 JSONL
    def write_jsonl(path: Path, data_list):
        with open(path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_jsonl_path = OUTPUT_DIR / f"train_{DATE}-emoctrl.jsonl"
    val_jsonl_path = OUTPUT_DIR / f"val_{DATE}-emoctrl.jsonl"
    write_jsonl(train_jsonl_path, train_set)
    write_jsonl(val_jsonl_path, val_set)

    print(f"[DONE] Saved:\n  {train_jsonl_path}\n  {val_jsonl_path}")


if __name__ == "__main__":
    main()
