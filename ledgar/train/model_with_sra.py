from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    EarlyStoppingCallback, RobertaForSequenceClassification, RobertaTokenizerFast,
    Trainer, TrainerCallback, TrainingArguments
)
import numpy as np
import matplotlib.pyplot as plt

train_dataset = load_from_disk("./aug_ledgar/train")
eval_dataset = load_from_disk("./aug_ledgar/validation")
test_dataset = load_from_disk("./aug_ledgar/test")

tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
max_length = 512

def tokenize_augmented_text(batch):
    return tokenizer(batch["augmented_text"], truncation=True, padding="max_length", max_length=max_length)

train_dataset = train_dataset.map(tokenize_augmented_text, batched=True, batch_size=8)
eval_dataset = eval_dataset.map(tokenize_augmented_text, batched=True, batch_size=8)
test_dataset = test_dataset.map(tokenize_augmented_text, batched=True, batch_size=8)

train_dataset = train_dataset.rename_column("label", "labels")
eval_dataset = eval_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.train_steps = []
        self.eval_epochs = []
        self.eval_accuracy = []
        self.eval_micro_f1 = []
        self.eval_macro_f1 = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        if 'loss' in logs:
            self.train_loss.append(logs['loss'])
            self.train_steps.append(state.global_step)

        if any(k.startswith('eval_') for k in logs.keys()):
            epoch = logs.get('epoch', len(self.eval_epochs) + 1)
            self.eval_epochs.append(epoch)

            if 'eval_accuracy' in logs:
                self.eval_accuracy.append(logs['eval_accuracy'])
            elif 'eval_eval_accuracy' in logs:
                self.eval_accuracy.append(logs['eval_eval_accuracy'])

            if 'eval_micro-f1' in logs:
                self.eval_micro_f1.append(logs['eval_micro-f1'])
            if 'eval_macro-f1' in logs:
                self.eval_macro_f1.append(logs['eval_macro-f1'])

    def plot(self):
        if self.train_loss:
            plt.figure(); plt.plot(self.train_steps, self.train_loss)
            plt.title("Training Loss"); plt.xlabel("Global Step"); plt.ylabel("Loss")
            plt.savefig("loss.png"); plt.close()

        if self.eval_accuracy:
            plt.figure(); plt.plot(self.eval_epochs[:len(self.eval_accuracy)], self.eval_accuracy, marker='o')
            plt.title("Eval Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
            plt.savefig("accuracy.png"); plt.close()

        if self.eval_micro_f1:
            plt.figure(); plt.plot(self.eval_epochs[:len(self.eval_micro_f1)], self.eval_micro_f1, marker='o')
            plt.title("Eval micro-F1"); plt.xlabel("Epoch"); plt.ylabel("micro-F1")
            plt.savefig("micro_f1.png"); plt.close()

        if self.eval_macro_f1:
            plt.figure(); plt.plot(self.eval_epochs[:len(self.eval_macro_f1)], self.eval_macro_f1, marker='o')
            plt.title("Eval macro-F1"); plt.xlabel("Epoch"); plt.ylabel("macro-F1")
            plt.savefig("macro_f1.png"); plt.close()

log_callback = LoggingCallback()

model = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=100)

training_args = TrainingArguments(
    output_dir="./roberta_rag_results",

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    load_best_model_at_end=True,
    metric_for_best_model="micro-f1",
    greater_is_better=True,

    num_train_epochs=20,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=3e-5,

    lr_scheduler_type="linear",
    warmup_ratio=0.0,
    weight_decay=0.0,

    fp16=True,
    fp16_full_eval=True,

    logging_strategy="epoch",
    report_to=[],

    seed=1,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=1,
    max_grad_norm=1.0,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "micro-f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro-f1": f1_score(labels, preds, average="macro", zero_division=0),
        "eval_accuracy": accuracy_score(labels, preds),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[log_callback, EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

predictions = trainer.predict(test_dataset)
print("Final test set results:", predictions.metrics)

log_callback.plot()
