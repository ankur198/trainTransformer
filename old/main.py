# %%
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
dataset["train"][100]

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# %%
import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", torch_dtype=torch.bfloat16, num_labels=5)

# %%
import numpy as np
import evaluate

metric = evaluate.load("accuracy")
# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# %%
from transformers import TrainingArguments

# Hyperparameters
lr = 1e-3 # size of optimization step
batch_size = 4 # No.of.examples processed per optimization step
num_epochs = 10 # No.of.times the model runs through training data.

training_args = TrainingArguments(
    output_dir= "test_trainer",
    # learning_rate=lr,
    # per_device_train_batch_size=batch_size,
    # per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# %%
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
peft_config = LoraConfig(task_type="SEQ_CLS", # Sequence Classification.
                        r=4,  # Intrinsic rank of trainable weight matrix.
                        lora_alpha=32,  # similar to Learning rate.
                        lora_dropout=0.01, # probability of dropout nodes.
                        # target_modules = ['query'] # LoRA is applied to the query layer.
                        ) 
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# %%
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
# %%