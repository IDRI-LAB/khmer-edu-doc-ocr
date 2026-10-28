# Copyright (c) 2024-2026 Unsloth AI (https://github.com/unslothai/notebooks)
# 
# This file contains code derived from the Unsloth AI notebooks collection.
# Modified and converted from .ipynb (https://github.com/unslothai/notebooks/blob/main/nb/Deepseek_OCR_2_(3B).ipynb) to .py by IDRI on 27/Oct/2026.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from unsloth import FastVisionModel  # FastLanguageModel for LLMs
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModel, Trainer, TrainingArguments, EarlyStoppingCallback
from unsloth import is_bf16_supported

os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

# ================================== Config ===================================
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR.parent / "configs" / "lora.yaml"


def load_config(path = CONFIG_PATH):
    with open(path, "r", encoding = "utf-8") as f:
        return yaml.safe_load(f)


config = load_config()

# ============================ Download base model ============================
# The base model is downloaded into the same directory as this script.
MODEL_DIR = SCRIPT_DIR / config["model"]["local_dir"]
LORA_OUTPUT_PATH = SCRIPT_DIR / config["model"]["output_path"]  # Save LoRA adapters here

snapshot_download(config["model"]["name"], local_dir = str(MODEL_DIR))

# Make `deepseek_ocr2.modeling_deepseekocr2` importable (downloaded with the model)
sys.path.insert(0, str(SCRIPT_DIR))

from data import load_datasets, DeepSeekOCR2DataCollator

# ================================ Load model =================================
model, tokenizer = FastVisionModel.from_pretrained(
    str(MODEL_DIR),
    load_in_4bit = config["model"]["load_in_4bit"],  # Use 4bit to reduce memory use. False for 16bit LoRA.
    auto_model = AutoModel,
    trust_remote_code = config["model"]["trust_remote_code"],
    unsloth_force_compile = config["model"]["unsloth_force_compile"],
    use_gradient_checkpointing = config["model"]["use_gradient_checkpointing"],  # True or "unsloth" for long context
)

# ================================== LoRA =====================================
model = FastVisionModel.get_peft_model(model, **config["lora"])

# ================================== Data =====================================
data_config = config["data"]
dataset = load_datasets(
    dataset_name = data_config["dataset_name"],
    subset = data_config["subset"],
    instruction = data_config["instruction"],
    splits = data_config["splits"],
)

print({split: len(ds) for split, ds in dataset.items()})
print(dataset["train"][0]["messages"][1])  # first assistant message (ground-truth text)

# ================================== Trainer ==================================
FastVisionModel.for_training(model)  # Enable for training!

data_collator = DeepSeekOCR2DataCollator(
    tokenizer = tokenizer,
    model = model,
    image_size = data_config["image_size"],
    base_size = data_config["base_size"],
    crop_mode = data_config["crop_mode"],
    train_on_responses_only = data_config["train_on_responses_only"],
)

training_args = dict(config["training"])
if training_args.get("fp16") == "auto":
    training_args["fp16"] = not is_bf16_supported()  # Use fp16 if bf16 is not supported
if training_args.get("bf16") == "auto":
    training_args["bf16"] = is_bf16_supported()      # Use bf16 if supported

early_stopping = config["early_stopping"]

trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = data_collator,  # Must use!
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"],
    args = TrainingArguments(**training_args),
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience = early_stopping["patience"],
            early_stopping_threshold = early_stopping["threshold"],
        ),
    ],
)

# ============================ Show current memory stats ======================
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# ============================== Logging helpers ==============================
LOG_FILE = "training_summary.log"
JSON_FILE = "training_results.json"


def write_log(text, path = LOG_FILE):
    """Print text and append it (with a timestamp) to a .log file."""
    print(text)
    with open(path, "a", encoding = "utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}]\n{text}\n\n")


# ================================== Train ====================================
start_time = time.time()

# To continue from an existing checkpoint (in "outputs/"), use this instead:
# trainer_stats = trainer.train(resume_from_checkpoint = True)
trainer_stats = trainer.train()

duration_seconds = int(time.time() - start_time)
hours, remainder = divmod(duration_seconds, 3600)
minutes, _ = divmod(remainder, 60)

# ========================== Show final memory and time stats =================
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

summary = "\n".join([
    f"Total training time: {hours}h {minutes}m",
    f"{trainer_stats.metrics['train_runtime']} seconds used for training.",
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.",
    f"Peak reserved memory = {used_memory} GB.",
    f"Peak reserved memory for training = {used_memory_for_lora} GB.",
    f"Peak reserved memory % of max memory = {used_percentage} %.",
    f"Peak reserved memory for training % of max memory = {lora_percentage} %.",
])
write_log(summary)

# ============================ Final training/validation loss =================
# load_best_model_at_end=True, so the model in memory is the best checkpoint (lowest eval_loss).
# Evaluate it on the validation set to get the validation loss of the final model.
final_eval = trainer.evaluate()

log_history = trainer.state.log_history
train_logs = [h for h in log_history if "loss" in h]  # training-loss entries only

best_ckpt = trainer.state.best_model_checkpoint
best_step = int(best_ckpt.split("-")[-1]) if best_ckpt else trainer.state.global_step
train_loss_at_best_step = next(
    (h["loss"] for h in reversed(train_logs) if h["step"] <= best_step), None
)

results = {
    "final_model": {
        "best_checkpoint": best_ckpt,
        "best_step": best_step,
        "validation_loss": final_eval["eval_loss"],            # final (best) model on validation set
        "train_loss_at_best_step": train_loss_at_best_step,    # last logged training loss up to best step
    },
    "training_run": {
        "average_train_loss": trainer_stats.metrics["train_loss"],  # mean over the whole run
        "last_logged_train_loss": train_logs[-1]["loss"] if train_logs else None,
        "total_steps": trainer.state.global_step,
        "epochs_completed": trainer.state.epoch,
        "train_runtime_seconds": trainer_stats.metrics["train_runtime"],
        "peak_reserved_memory_gb": used_memory,
    },
    "log_history": log_history,  # every logged train/eval loss, for plotting curves
}

with open(JSON_FILE, "w", encoding = "utf-8") as f:
    json.dump(results, f, indent = 2, ensure_ascii = False)

write_log(
    f"Final model (step {best_step}): "
    f"train loss = {train_loss_at_best_step}, validation loss = {final_eval['eval_loss']}"
)

# ================================== Save =====================================
# This ONLY saves the LoRA adapters, not the full model.
model.save_pretrained(LORA_OUTPUT_PATH)  # Local saving
tokenizer.save_pretrained(LORA_OUTPUT_PATH)