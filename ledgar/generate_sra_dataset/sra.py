from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

import torch
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, RobertaTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib
from collections import Counter

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
SBERT_MAX_LEN = 256
SBERT_BATCH = 512
TFIDF_TOPK = 20
DENSE_TOPK = 5
FINAL_TOPK = 1
MAX_RETRIEVED_TOKENS = 64
LOW_FREQ_CUTOFF = 0.65
OUTPUT_DIR = Path("./aug_ledgar")
CACHE_DIR = Path("./aug_cache")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")

SEP_TOKEN = tokenizer.sep_token

def encode_cached(texts: List[str], cache_path: Path, device="cuda"):
    if cache_path.exists():
        return np.load(cache_path)
    model = SentenceTransformer(SBERT_MODEL, device=device)
    model.max_seq_length = SBERT_MAX_LEN
    model.eval()
    model = model.half()
    with torch.inference_mode():
        embs = model.encode(
            texts,
            batch_size=SBERT_BATCH,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    np.save(cache_path, embs.astype(np.float16))
    return np.load(cache_path)


def build_tfidf_on_train(train_texts: List[str]):
    vec_p = CACHE_DIR / "tfidf_vectorizer.joblib"
    mat_p = CACHE_DIR / "train_tfidf.npz"
    if vec_p.exists() and mat_p.exists():
        vectorizer = joblib.load(vec_p)
        train_mat = sparse.load_npz(mat_p)
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.8)
        train_mat = vectorizer.fit_transform(train_texts)
        joblib.dump(vectorizer, vec_p)
        sparse.save_npz(mat_p, train_mat)
    return vectorizer, train_mat


def tfidf_topk_indices(vectorizer, train_mat, query_texts: List[str], k=50, batch=2048):
    out = []
    for i in range(0, len(query_texts), batch):
        qs = query_texts[i:i+batch]
        q_mat = vectorizer.transform(qs)
        scores = q_mat @ train_mat.T
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


def dense_select_from_ids(query_vec: np.ndarray, all_embs: np.ndarray, cand_ids: np.ndarray, topk: int):
    if len(cand_ids) == 0:
        return []
    cands = all_embs[cand_ids].astype(np.float32, copy=False)
    q = query_vec.astype(np.float32, copy=False)
    sims = cands @ q
    if len(sims) <= topk:
        order = np.argsort(sims)[::-1]
    else:
        part = np.argpartition(sims, -topk)[-topk:]
        order = part[np.argsort(sims[part])[::-1]]
    return cand_ids[order].tolist()

def truncate_to_token_limit(text: str, max_tokens: int = MAX_RETRIEVED_TOKENS) -> str:
    tokens = tokenizer.tokenize(text)
    return tokenizer.convert_tokens_to_string(tokens[:max_tokens])


def compute_low_freq_labels(all_labels, cutoff=LOW_FREQ_CUTOFF):
    label_counter = Counter(all_labels)
    sorted_labels = sorted(label_counter.items(), key=lambda x: x[1])
    num_low_freq = int(len(sorted_labels) * cutoff)
    low_freq_labels = set(label for label, _ in sorted_labels[:num_low_freq])
    return low_freq_labels

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading LEDGAR ...")
    ds = load_dataset("lex_glue", "ledgar")

    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    print("Encoding train embeddings (cached) ...")
    train_emb = encode_cached(train_texts, CACHE_DIR / "train_emb.npy", device=device)

    print("Building TF-IDF on train ...")
    vectorizer, train_tfidf = build_tfidf_on_train(train_texts)

    low_freq_labels = compute_low_freq_labels(train_labels, cutoff=LOW_FREQ_CUTOFF)

    for split_name in ["train", "validation", "test"]:
        print(f"\n Generating augmented_text for [{split_name}] ...")
        split: Dataset = ds[split_name]
        texts = split["text"]
        labels = split["label"]
        q_emb = encode_cached(texts, CACHE_DIR / f"{split_name}_emb.npy", device=device)

        tfidf_ids_list = tfidf_topk_indices(vectorizer, train_tfidf, texts, k=TFIDF_TOPK, batch=2048)
        if split_name == "train":
            for i in range(len(texts)):
                tfidf_ids_list[i] = np.array([j for j in tfidf_ids_list[i] if j != i], dtype=np.int64)

        dense_top_ids_batch = []
        for i, cand_ids in tqdm(list(enumerate(tfidf_ids_list)), desc=f"TFIDF→dense ({split_name})"):
            top_ids = dense_select_from_ids(q_emb[i], train_emb, cand_ids, topk=DENSE_TOPK)
            dense_top_ids_batch.append(top_ids)

        final_texts_batch = [[train_texts[j] for j in ids[:FINAL_TOPK]] for ids in dense_top_ids_batch]

        augmented_texts = []
        for q, cands, label in zip(texts, final_texts_batch, labels):
            if label in low_freq_labels and len(cands) > 0:
                retrieved = truncate_to_token_limit(cands[0], max_tokens=MAX_RETRIEVED_TOKENS)
                aug = f"Original clause: {q} {SEP_TOKEN} Related clause for reference: {retrieved}"
                augmented_texts.append(aug)
            else:
                augmented_texts.append(q)

        out_ds = split.add_column("augmented_text", augmented_texts)
        if split_name == "train":
            out_ds = out_ds.shuffle(seed=42)

        out_dir = OUTPUT_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_ds.save_to_disk(str(out_dir))
        print(f"Saved: {out_dir}")

    print("\nAll splits processed & saved.")

if __name__ == "__main__":
    main()
