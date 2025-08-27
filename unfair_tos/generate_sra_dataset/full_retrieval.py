# -*- coding: utf-8 -*-
"""
Full retrieval for unfair_tos (multi-label):
TF-IDF -> SBERT re-rank -> top-1, augment ALL samples (train/validation/test).
Outputs HF datasets with an extra column 'augmented_text' for each split.
"""
from pathlib import Path
from typing import List
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizerFast

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
SBERT_BATCH = 256
SBERT_MAX_LEN = 256

TFIDF_TOPK = 20
DENSE_TOPK = 5
FINAL_TOPK = 1
MAX_RETRIEVED_TOKENS = 64

OUTPUT_DIR = Path("./aug_unfair_tos_full")
CACHE_DIR  = Path("./aug_cache_unfairtos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEP_TOKEN = RobertaTokenizerFast.from_pretrained("roberta-base").sep_token or "</s>"
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== 工具函数 =====================
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
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=200_000
            )
            train_mat = vectorizer.fit_transform(train_texts)
            joblib.dump(vectorizer, vec_p)
            sparse.save_npz(mat_p, train_mat)
    except Exception:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=200_000
        )
        train_mat = vectorizer.fit_transform(train_texts)
        from scipy import sparse
        sparse.save_npz(mat_p, train_mat)

    return vectorizer, train_mat


def tfidf_topk_indices(vectorizer, train_mat, query_texts: List[str], k=50, batch=2048):
    out = []
    for i in range(0, len(query_texts), batch):
        qs = query_texts[i:i+batch]
        q_mat = vectorizer.transform(qs)     # (b, V)
        scores = q_mat @ train_mat.T         # (b, n_train), CSR
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
    cands = all_embs[cand_ids]                           # (m, d)
    sims = (cands @ q_vec.reshape(-1, 1)).squeeze(-1)    # (m,)
    if sims.size <= topk:
        order = np.argsort(sims)[::-1]
    else:
        part = np.argpartition(sims, -topk)[-topk:]
        order = part[np.argsort(sims[part])[::-1]]
    return cand_ids[order].tolist()


def truncate_to_word_limit(text: str, max_words: int = MAX_RETRIEVED_TOKENS) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def normalize_for_match(s: str) -> str:
    return " ".join(s.strip().lower().split())


def main():
    np.random.seed(SEED)

    print("Loading unfair_tos ...")
    ds = load_dataset("lex_glue", "unfair_tos")

    train_texts: List[str] = list(ds["train"]["text"])
    train_norm  = [normalize_for_match(t) for t in train_texts]

    print("Building TF-IDF on train ...")
    vectorizer, train_tfidf = build_tfidf_on_train(train_texts)

    print("Encoding SBERT train embeddings (cached) ...")
    train_emb = encode_cached(train_texts, CACHE_DIR / "train_emb.npy", device=device)

    print("Loading SBERT encoder for queries ...")
    q_encoder = SentenceTransformer(SBERT_MODEL, device=device)
    q_encoder.max_seq_length = SBERT_MAX_LEN
    q_encoder.eval()

    for split_name in ["train", "validation", "test"]:
        print(f"\n Generating augmented_text for [{split_name}] (Full SRA; augment ALL samples) ...")
        split: Dataset = ds[split_name]
        texts: List[str] = list(split["text"])

        tfidf_ids_list = tfidf_topk_indices(
            vectorizer, train_tfidf, texts, k=TFIDF_TOPK, batch=2048
        )

        with torch.inference_mode():
            q_embs = q_encoder.encode(
                texts, batch_size=SBERT_BATCH,
                convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=True
            ).astype(np.float32, copy=False)

        augmented_texts = []
        pbar = tqdm(total=len(texts), desc=f"TFIDF→SBERT ({split_name})", ncols=100)

        for i, (q, cand_ids) in enumerate(zip(texts, tfidf_ids_list)):
            if cand_ids.size == 0:
                augmented_texts.append(q)
                pbar.update(1)
                continue

            if split_name == "train":
                q_norm = normalize_for_match(q)
                filtered = [int(cid) for cid in cand_ids if train_norm[int(cid)] != q_norm]
                cand_ids = np.array(filtered, dtype=np.int64)
                if cand_ids.size == 0:
                    augmented_texts.append(q)
                    pbar.update(1)
                    continue

            q_vec = q_embs[i]
            top_dense_ids = dense_select_from_ids(q_vec, train_emb, cand_ids, topk=DENSE_TOPK)

            final_ids = top_dense_ids[:FINAL_TOPK]
            if len(final_ids) == 0:
                augmented_texts.append(q)
            else:
                retrieved = truncate_to_word_limit(train_texts[final_ids[0]], max_words=MAX_RETRIEVED_TOKENS)
                aug = f"Original clause: {q} {SEP_TOKEN} Related clause for reference: {retrieved}"
                augmented_texts.append(aug)

            pbar.update(1)

        pbar.close()

        out_ds = split.add_column("augmented_text", augmented_texts)
        if split_name == "train":
            out_ds = out_ds.shuffle(seed=SEED)

        out_dir = OUTPUT_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_ds.save_to_disk(str(out_dir))
        print(f"Saved: {out_dir}")

    print("\nAll splits processed & saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
