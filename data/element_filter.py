import json
import nltk
from nltk.corpus import stopwords
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

stop_words = set(stopwords.words("english"))

stop_words |= ENGLISH_STOP_WORDS


def is_valid_noun(word):
    return word.lower() not in stop_words and word.isalpha() and len(word) > 1


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


def get_data():
    with open("./data/emoset_plus.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


"""
Example data structure:
{
    "anger_07969": {
        "Objective Caption": "A group of actors on stage, one making a confrontational gesture towards another.",
        "Objective Elements": "[actors, stage]",
        "emotion": "anger"
    },
}
"""

data = get_data()
# 按情感分组
emotion_docs = defaultdict(list)
for item in data.values():
    emotion = item.get("Emotion", "")
    objective_elements = item.get("Objective Elements", "")
    elements = objective_elements.strip("[]").replace(", ", " ")
    emotion_docs[emotion].append(elements)

# 统计每个名词在各情感类别中的分布和出现次数
noun_emotion_map = defaultdict(set)
noun_counter = Counter()
for emo in emotion_list:
    docs = emotion_docs[emo]
    if not docs:
        continue
    for doc in docs:
        words = doc.split()
        nouns = [
            w
            for w, pos in nltk.pos_tag(words)
            if pos in ["NN", "NNS", "NNP", "NNPS"] and is_valid_noun(w)
        ]
        noun_counter.update(nouns)
        for n in nouns:
            noun_emotion_map[n].add(emo)


def get_neutral_nouns_by_tfidf():
    """获取情感中性名词"""
    all_noun_docs = []
    for emo in emotion_list:
        docs = emotion_docs[emo]
        for doc in docs:
            words = doc.split()
            nouns = [
                w
                for w, pos in nltk.pos_tag(words)
                if pos in ["NN", "NNS", "NNP", "NNPS"] and is_valid_noun(w)
            ]
            all_noun_docs.append(" ".join(nouns))

    # 用TF-IDF计算所有类别的名词
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = vectorizer.fit_transform(all_noun_docs)
    feature_names = vectorizer.get_feature_names_out()
    avg_tfidf = tfidf_matrix.mean(axis=0).A1
    tfidf_scores = dict(zip(feature_names, avg_tfidf))

    # 统计每个词出现的情感类别数
    word_emotion_count = {w: len(noun_emotion_map[w]) for w in feature_names}

    # 只保留在8个情感类别都出现且TF-IDF分数较高的词
    neutral_candidates = [
        w
        for w in feature_names
        if word_emotion_count.get(w, 0) >= 8 and tfidf_scores[w] > 0.003
    ]

    # 按TF-IDF分数排序
    neutral_candidates = sorted(
        neutral_candidates, key=lambda w: tfidf_scores[w], reverse=True
    )

    print("TF-IDF高分情感中性词：", neutral_candidates)
    return neutral_candidates
