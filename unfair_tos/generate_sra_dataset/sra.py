# -*- coding: utf-8 -*-
"""
RSA for unfair_tos (multi-label):
TF-IDF -> SBERT re-rank -> top-1, only augment samples containing any low-frequency label.
Saves HF datasets with an extra column 'augmented_text' for train/validation/test.
"""

from pathlib import Path
from typing import List
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer

# ===================== Config =====================
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
SBERT_BATCH = 256
SBERT_MAX_LEN = 256
TFIDF_TOPK = 20
DENSE_TOPK = 5
FINAL_TOPK = 1
MAX_RETRIEVED_TOKENS = 64

LOW_FREQ_THRESHOLD = 200
LOW_FREQ_PERCENTILE = None

OUTPUT_DIR = Path("./aug_unfair_tos")
CACHE_DIR  = Path("./aug_cache_unfairtos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

from transformers import RobertaTokenizerFast
SEP_TOKEN = RobertaTokenizerFast.from_pretrained("roberta-base").sep_token or "</s>"

device = "cuda" if torch.cuda.is_available() else "cpu"

def encode_cached(texts: List[str], cache_path: Path, device: str) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)
    model = SentenceTransformer(SBERT_MODEL, device=device)
    model.max_seq_length = SBERT_MAX_LEN
    model.eval()
    with torch.inference_mode():
        embs = model.encode(
            texts,
            batch_size=SBERT_BATCH,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32, copy=False)
    np.save(cache_path, embs)
    return embs


def build_tfidf_on_train(train_texts: List[str]):
    from scipy import sparse
    vec_p = CACHE_DIR / "tfidf_vectorizer.joblib"
    mat_p = CACHE_DIR / "train_tfidf.npz"

    try:
        import joblib
        if vec_p.exists() and mat_p.exists():
            vectorizer = joblib.load(vec_p)
            train_mat = sparse.load_npz(mat_p)
        else:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=200_000)
            train_mat = vectorizer.fit_transform(train_texts)
            joblib.dump(vectorizer, vec_p)
            sparse.save_npz(mat_p, train_mat)
    except Exception:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=200_000)
        train_mat = vectorizer.fit_transform(train_texts)
        sparse.save_npz(mat_p, train_mat)

    return vectorizer, train_mat


def tfidf_topk_indices(vectorizer, train_mat, query_texts: List[str], k=50, batch=2048):
    out = []
    for i in range(0, len(query_texts), batch):
        qs = query_texts[i:i+batch]
        q_mat = vectorizer.transform(qs)             # (b, V)
        scores = q_mat @ train_mat.T                 # (b, n_train)
        scores = scores.tocsr()
        for r in range(scores.shape[0]):
            row = scores.getrow(r)
            if row.nnz == 0:
                out.append(np.array([], dtype=np.int64))
            elif row.nnz <= k:
                idx = row.indices[np.argsort(row.data)[::-1]]
                out.append(idx)
            else:
                part = np.argpartition(row.data, -k)[-k:]
                idx = row.indices[part[np.argsort(row.data[part])[::-1]]]
                out.append(idx)
    return out


def dense_select_from_ids(q_vec: np.ndarray, all_embs: np.ndarray, cand_ids: np.ndarray, topk: int):
    if cand_ids.size == 0:
        return []
    cands = all_embs[cand_ids]        # (m, d)
    q = q_vec.reshape(1, -1)          # (1, d)
    sims = (cands @ q.T).squeeze(-1)  # (m,)
    if sims.size <= topk:
        order = np.argsort(sims)[::-1]
    else:
        part = np.argpartition(sims, -topk)[-topk:]
        order = part[np.argsort(sims[part])[::-1]]
    return cand_ids[order].tolist()


def truncate_to_word_limit(text: str, max_words: int = MAX_RETRIEVED_TOKENS) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def compute_low_freq_labels_sparse(label_lists: List[List[int]], num_classes: int = 8):
    cnt = Counter([l for labels in label_lists for l in labels])
    counts = np.array([cnt.get(i, 0) for i in range(num_classes)], dtype=int)

    if LOW_FREQ_PERCENTILE is not None:
        thr = np.quantile(counts, LOW_FREQ_PERCENTILE)
        low_freq = {i for i, c in enumerate(counts) if c <= thr}
    else:
        low_freq = {i for i, c in enumerate(counts) if c <= LOW_FREQ_THRESHOLD}

    return low_freq, counts


# ===================== Main =====================
def main():
    print("Loading unfair_tos ...")
    ds = load_dataset("lex_glue", "unfair_tos")

    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["labels"]  # list[list[int]]

    low_freq_labels, counts = compute_low_freq_labels_sparse(train_labels, num_classes=8)
    print("Label counts:", counts.tolist())
    print("Low-freq labels (selected for augmentation):", sorted(list(low_freq_labels)))

    print("Building TF-IDF on train ...")
    vectorizer, train_tfidf = build_tfidf_on_train(train_texts)

    print("Encoding SBERT train embeddings (cached) ...")
    train_emb = encode_cached(train_texts, CACHE_DIR / "train_emb.npy", device=device)  # (n_train, d), L2 normed

    print("Loading SBERT encoder for queries ...")
    q_encoder = SentenceTransformer(SBERT_MODEL, device=device)
    q_encoder.max_seq_length = SBERT_MAX_LEN
    q_encoder.eval()

    def sample_needs_aug(label_list: List[int]) -> bool:
        return any(l in low_freq_labels for l in label_list)

    for split_name in ["train", "validation", "test"]:
        print(f"\n Generating augmented_text for [{split_name}] ...")
        split: Dataset = ds[split_name]
        texts: List[str] = split["text"]
        labels: List[List[int]] = split["labels"]

        tfidf_ids_list = tfidf_topk_indices(vectorizer, train_tfidf, texts, k=TFIDF_TOPK, batch=2048)

        if split_name == "train":
            from collections import defaultdict
            inv = defaultdict(list)
            for i, t in enumerate(train_texts):
                inv[t].append(i)

        augmented_texts = []
        for i, (q, cand_ids, y) in enumerate(tqdm(zip(texts, tfidf_ids_list, labels), total=len(texts), desc=f"TFIDF→SBERT ({split_name})")):
            if not sample_needs_aug(y) or cand_ids.size == 0:
                augmented_texts.append(q)
                continue

            if split_name == "train":
                if q in inv:
                    cand_ids = np.array([cid for cid in cand_ids if cid not in inv[q]], dtype=np.int64)
                    if cand_ids.size == 0:
                        augmented_texts.append(q)
                        continue

            with torch.inference_mode():
                q_vec = q_encoder.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
            top_dense_ids = dense_select_from_ids(q_vec, train_emb, cand_ids, topk=DENSE_TOPK)

            final_ids = top_dense_ids[:FINAL_TOPK]
            if len(final_ids) == 0:
                augmented_texts.append(q)
            else:
                retrieved = truncate_to_word_limit(train_texts[final_ids[0]], max_words=MAX_RETRIEVED_TOKENS)
                aug = f"Original clause: {q} {SEP_TOKEN} Related clause for reference: {retrieved}"
                augmented_texts.append(aug)

        out_ds = split.add_column("augmented_text", augmented_texts)
        if split_name == "train":
            out_ds = out_ds.shuffle(seed=42)

        out_dir = OUTPUT_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_ds.save_to_disk(str(out_dir))
        print(f"Saved: {out_dir}")

    print("\n All splits processed & saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
