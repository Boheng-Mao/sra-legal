import random
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from datasets import load_dataset, Features, Sequence, Value
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)

MODEL_NAME = "roberta-base"
NUM_LABELS = 8
MAX_LEN = 128
SEED = 3
BATCH_SIZE = 8
LR = 3e-5
EPOCHS = 20
USE_NO_LABEL_CLASS = True

OUTPUT_DIR = ""
LOGGING_DIR = ""
SAVE_TOTAL_LIMIT = 2

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

dataset = load_dataset("coastalcph/lex_glue", "unfair_tos")

def to_multihot(example):
    vec = [0.0] * NUM_LABELS
    for i in example["labels"]:
        vec[i] = 1.0
    example["labels"] = vec
    return example

dataset = dataset.map(to_multihot)

def drop_label_field(ex):
    if "label" in ex:
        del ex["label"]
    return ex
dataset = dataset.map(drop_label_field)

new_features = dataset["train"].features.copy()
new_features["labels"] = Sequence(Value("float32"), length=NUM_LABELS)
dataset = dataset.cast(new_features)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(ex):
    return tokenizer(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)

from torch.utils.data import default_collate
class MultilabelCollator:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    def __call__(self, features):
        batch = default_collate(features)
        if "labels" in batch and batch["labels"].dtype != torch.float32:
            batch["labels"] = batch["labels"].float()
        return batch

def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
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

    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    exact = np.all(y_pred8 == y_true8, axis=1).mean()

    return {"macro-f1": macro, "micro-f1": micro, "eval_accuracy": exact}

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOGGING_DIR,

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
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    data_collator=MultilabelCollator(tokenizer)
)

trainer.train()

print("== DEV ==", trainer.evaluate(tokenized["validation"]))
print("== TEST ==", trainer.evaluate(tokenized["test"]))

def plot_metrics(log_history, save_path_prefix="metric_plot"):
    wanted = ["eval_loss", "eval_accuracy", "micro-f1", "macro-f1"]
    series = {k: {"x": [], "y": []} for k in wanted}

    for log in log_history:
        if ("eval_loss" in log) and ("epoch" in log):
            x = log["epoch"]
            for k in wanted:
                if k in log:
                    series[k]["x"].append(x)
                    series[k]["y"].append(log[k])

    for k, xy in series.items():
        xs, ys = xy["x"], xy["y"]
        if len(xs) > 0 and len(xs) == len(ys):
            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.title(f"{k} over epochs")
            plt.xlabel("Epoch")
            plt.ylabel(k)
            plt.grid(True)
            fname = f"{save_path_prefix}_{k.replace('/', '_')}.png"
            plt.savefig(fname); plt.close()
            print(f"[saved] {fname}")

plot_metrics(trainer.state.log_history)
