# Copyright (c) 2024-2026 Unsloth AI (https://github.com/unslothai/notebooks)
#
# This file contains code derived from the Unsloth AI notebooks collection.
# Modified and converted from .ipynb
# (https://github.com/unslothai/notebooks/blob/main/nb/Deepseek_OCR_2_(3B).ipynb)
# to .py by IDRI on 27/Oct/2026.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from unsloth import FastVisionModel
import os
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, Trainer, TrainingArguments, EarlyStoppingCallback
from unsloth import is_bf16_supported

os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import CONFIG_PATH, load_config, resolve_path
from utils.training_utils import resolve_precision, save_training_artifacts

# ================================== Config ===================================
config = load_config(CONFIG_PATH, section="deepseek")
model_config = config["model"]
data_config = config["data"]
training_config = config["training"]
early_stopping = config["early_stopping"]

MODEL_DIR = resolve_path(model_config["local_dir"])
LORA_OUTPUT_PATH = resolve_path(model_config["output_path"])
LOG_FILE = "deepseek_training_summary.log"
JSON_FILE = "deepseek_training_results.json"

# ============================ Download base model ============================
snapshot_download(model_config["name"], local_dir=str(MODEL_DIR))

# Make `deepseek_ocr2.modeling_deepseekocr2` importable (downloaded with the model)
sys.path.insert(0, str(MODEL_DIR.parent))

from data import load_datasets, DeepSeekOCR2DataCollator

# ================================ Load model =================================
model, tokenizer = FastVisionModel.from_pretrained(
    str(MODEL_DIR),
    max_seq_length=model_config["max_seq_length"],
    load_in_4bit=model_config["load_in_4bit"],
    auto_model=AutoModel,
    trust_remote_code=model_config["trust_remote_code"],
    unsloth_force_compile=model_config["unsloth_force_compile"],
    use_gradient_checkpointing=model_config["use_gradient_checkpointing"],
)

# ================================== LoRA =====================================
model = FastVisionModel.get_peft_model(model, **config["lora"])

# ================================== Data =====================================
train_split = data_config["train_split"]
validation_split = data_config["validation_split"]
dataset = load_datasets(
    dataset_name=data_config["dataset_name"],
    subset=data_config["subset"],
    instruction=data_config["instruction"],
    splits=[train_split, validation_split],
    conversation_format="deepseek",
)

print({split: len(ds) for split, ds in dataset.items()})
print(dataset[train_split][0]["messages"][1])

# ================================== Trainer ==================================
FastVisionModel.for_training(model)

data_collator = DeepSeekOCR2DataCollator(
    tokenizer=tokenizer,
    model=model,
    image_size=data_config["image_size"],
    base_size=data_config["base_size"],
    crop_mode=data_config["crop_mode"],
    train_on_responses_only=data_config["train_on_responses_only"],
)

training_args = resolve_precision(training_config, is_bf16_supported())
training_args["output_dir"] = str(resolve_path(training_args["output_dir"]))

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=dataset[train_split],
    eval_dataset=dataset[validation_split],
    args=TrainingArguments(**training_args),
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=early_stopping["patience"],
            early_stopping_threshold=early_stopping["threshold"],
        ),
    ],
)

# ============================ Show current memory stats ======================
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# ================================== Train ====================================
start_time = time.time()
trainer_stats = trainer.train()

save_training_artifacts(
    trainer=trainer,
    trainer_stats=trainer_stats,
    start_time=start_time,
    start_gpu_memory=start_gpu_memory,
    max_memory=max_memory,
    log_file=LOG_FILE,
    json_file=JSON_FILE,
    loss_curve_dir=LORA_OUTPUT_PATH,
    title="DeepSeek-OCR-2 Khmer LoRA",
)

# ================================== Save =====================================
LORA_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
model.save_pretrained(LORA_OUTPUT_PATH)
tokenizer.save_pretrained(LORA_OUTPUT_PATH)
