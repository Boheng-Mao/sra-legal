# -*- coding: utf-8 -*-
"""
Train RoBERTa-base on unfair_tos (multi-label, 8 classes) with SRA.
"""

import random
import json
from typing import List
import os
import numpy as np
import torch
from scipy.special import expit
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import f1_score

MODEL_NAME = "roberta-base"
NUM_LABELS = 8
MAX_LEN = 128
SEED = 2
BATCH_SIZE = 8
LR = 3e-5
EPOCHS = 20
USE_NO_LABEL_CLASS = True

OUTPUT_DIR = ""
SAVE_TOTAL_LIMIT = 2

AUG_DATA_DIR = "./aug_unfair_tos"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


def to_multihot_sparse_indices(label_list: List[int], num_labels: int) -> List[float]:
    vec = [0.0] * num_labels
    for i in label_list:
        if 0 <= i < num_labels:
            vec[i] = 1.0
    return vec

def load_aug_dataset(path: str):
    try:
        ds = load_from_disk(path)
        if isinstance(ds, DatasetDict):
            for split in ds.keys():
                if "augmented_text" not in ds[split].column_names:
                    ds[split] = ds[split].add_column("augmented_text", ds[split]["text"])
            return ds, "augmented_text"
    except Exception:
        pass

    needed = ["train", "validation", "test"]
    splits = {}
    for sp in needed:
        subdir = os.path.join(path, sp)
        if not os.path.isdir(subdir):
            raise FileNotFoundError()
        splits[sp] = load_from_disk(subdir)

        if "augmented_text" not in splits[sp].column_names:
            splits[sp] = splits[sp].add_column("augmented_text", splits[sp]["text"])

    ds = DatasetDict(splits)
    return ds, "augmented_text"


def prepare_tokenized(ds: DatasetDict, text_col: str, tokenizer: AutoTokenizer) -> DatasetDict:
    def map_labels(batch):
        batch["labels"] = [to_multihot_sparse_indices(lbls, NUM_LABELS) for lbls in batch["labels"]]
        return batch
    ds = ds.map(map_labels, batched=True, desc="labels -> multi-hot(float32)")

    def tokenize_fn(ex):
        return tokenizer(
            ex[text_col],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
    ds = ds.map(tokenize_fn, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in ["labels"]], desc=f"Tokenizing on '{text_col}'")

    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def compute_metrics(eval_pred):
    logits, label_ids = eval_pred  # logits: (N,8), label_ids: (N,8)
    logits = np.asarray(logits)
    y_true8 = np.asarray(label_ids).astype(int)
    y_pred8 = (expit(logits) > 0.5).astype(int)

    if USE_NO_LABEL_CLASS:
        N, C = y_true8.shape  # C=8
        y_true = np.zeros((N, C + 1), dtype=int)
        y_pred = np.zeros((N, C + 1), dtype=int)
        y_true[:, :C] = y_true8
        y_pred[:, :C] = y_pred8
        y_true[:, C] = (y_true8.sum(axis=1) == 0).astype(int)
        y_pred[:, C] = (y_pred8.sum(axis=1) == 0).astype(int)
    else:
        y_true, y_pred = y_true8, y_pred8

    micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    exact = np.all(y_pred8 == y_true8, axis=1).mean()

    return {"macro-f1": macro, "micro-f1": micro, "eval_accuracy": exact}


from torch.utils.data import default_collate
class MultilabelCollator:
    def __call__(self, features):
        batch = default_collate(features)
        if "labels" in batch and batch["labels"].dtype != torch.float32:
            batch["labels"] = batch["labels"].float()
        return batch


def main():
    ds, text_col = load_aug_dataset(AUG_DATA_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenized = prepare_tokenized(ds, text_col, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,

        load_best_model_at_end=True,
        metric_for_best_model="micro-f1",
        greater_is_better=True,

        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,

        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        weight_decay=0.0,

        fp16=True,
        fp16_full_eval=True,

        logging_strategy="epoch",
        report_to="none",
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=MultilabelCollator()
    )

    trainer.train()

    print("== DEV ==", trainer.evaluate(tokenized["validation"]))
    test_metrics = trainer.evaluate(tokenized["test"])
    print("== TEST ==", test_metrics)

    preds = trainer.predict(tokenized["test"])
    logits = preds.predictions
    probs  = expit(logits)
    y_pred = (probs > 0.5).astype(int)

    np.save("unfairtos_test_probs.npy", probs)
    np.save("unfairtos_test_pred8.npy", y_pred)
    with open("unfairtos_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    print("[saved] unfairtos_test_probs.npy, unfairtos_test_pred8.npy, unfairtos_test_metrics.json")


if __name__ == "__main__":
    main()
