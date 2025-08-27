from pathlib import Path
import numpy as np
from collections import Counter
from datasets import load_dataset
from scipy import sparse
import joblib

CACHE_DIR = Path("")
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
TFIDF_TOPK = 20
DENSE_TOPK = 5
SPLIT = "test"       # or validation

ds = load_dataset("lex_glue", "ledgar")
train_texts = ds["train"]["text"]
train_labels = ds["train"]["label"]
split_texts = ds[SPLIT]["text"]
split_labels = ds[SPLIT]["label"]

train_emb = np.load(CACHE_DIR / "train_emb.npy")                # shape [N_train, d], L2-normalized
q_emb = np.load(CACHE_DIR / f"{SPLIT}_emb.npy")                 # shape [N_split, d], L2-normalized

vectorizer = joblib.load(CACHE_DIR / "tfidf_vectorizer.joblib")
train_tfidf = sparse.load_npz(CACHE_DIR / "train_tfidf.npz").tocsr()

def compute_low_freq_labels(all_labels, cutoff: float):
    cnt = Counter(all_labels)
    sorted_items = sorted(cnt.items(), key=lambda x: x[1])
    k = int(len(sorted_items) * cutoff)
    return set(lbl for lbl, _ in sorted_items[:k])

def tfidf_topk_indices(query_texts, k=TFIDF_TOPK, batch=2048):
    out = []
    for i in range(0, len(query_texts), batch):
        qs = query_texts[i:i+batch]
        q_mat = vectorizer.transform(qs).tocsr()
        scores = q_mat @ train_tfidf.T  # csr @ csr^T → csr
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

def dense_top1_for_each(query_embeds, cand_ids_list, topk=DENSE_TOPK):
    top1_ids = []
    top1_sims = []
    for i, cand_ids in enumerate(cand_ids_list):
        if len(cand_ids) == 0:
            top1_ids.append(-1)
            top1_sims.append(float("nan"))
            continue
        cands = train_emb[cand_ids].astype(np.float32, copy=False)
        q = query_embeds[i].astype(np.float32, copy=False)
        sims = cands @ q
        if len(sims) > topk:
            part = np.argpartition(sims, -topk)[-topk:]
            order = part[np.argsort(sims[part])[::-1]]
        else:
            order = np.argsort(sims)[::-1]
        best = int(cand_ids[order[0]])
        top1_ids.append(best)
        top1_sims.append(float(sims[order[0]]))
    return np.array(top1_ids, dtype=np.int64), np.array(top1_sims, dtype=np.float32)

def eval_stats_for_cutoff(cutoff: float):
    low_freq = compute_low_freq_labels(train_labels, cutoff)
    tfidf_ids = tfidf_topk_indices(split_texts, k=TFIDF_TOPK)
    top1_ids, top1_sims = dense_top1_for_each(q_emb, tfidf_ids, topk=DENSE_TOPK)

    is_low = np.array([lbl in low_freq for lbl in split_labels], dtype=bool)
    has_cand = (top1_ids != -1)
    augmented_mask = is_low & has_cand

    coverage = augmented_mask.mean()
    sims = top1_sims[augmented_mask]
    if sims.size:
        stats = dict(
            coverage=float(coverage),
            sim_mean=float(np.nanmean(sims)),
            sim_median=float(np.nanmedian(sims)),
            sim_std=float(np.nanstd(sims)),
            count=int(sims.size),
        )
    else:
        stats = dict(coverage=float(coverage), sim_mean=float("nan"),
                     sim_median=float("nan"), sim_std=float("nan"), count=0)
    return stats

if __name__ == "__main__":
    for cutoff in [0.2, 0.3, 0.5, 0.55, 0.6, 0.65, 0.7, 0.9]:
        s = eval_stats_for_cutoff(cutoff)
        print(f"cutoff={int(cutoff*100)}% [{SPLIT}] -> "
              f"coverage={s['coverage']:.3f}, "
              f"cos_mean={s['sim_mean']:.3f}, "
              f"cos_med={s['sim_median']:.3f}, "
              f"cos_std={s['sim_std']:.3f}, "
              f"n_aug={s['count']}")
