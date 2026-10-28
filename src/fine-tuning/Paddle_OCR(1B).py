import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

from unsloth import FastVisionModel
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import PaddleOCRVLDataCollator, load_datasets
from utils.config import CONFIG_PATH, load_config, resolve_path
from utils.training_utils import resolve_precision, save_training_artifacts

# ================================== Config ===================================
config = load_config(CONFIG_PATH, section="paddle")
model_config = config["model"]
data_config = config["data"]
training_config = config["training"]
early_stopping = config["early_stopping"]
collator_config = config["collator"]

model_path = model_config["name"]
LORA_OUTPUT_PATH = resolve_path(model_config["output_path"])
LOG_FILE = "paddle_training_summary.log"
JSON_FILE = "paddle_training_results.json"

# ================================ Load model =================================
model, tokenizer = FastVisionModel.from_pretrained(
    model_path,
    max_seq_length=model_config["max_seq_length"],
    load_in_4bit=model_config["load_in_4bit"],
    load_in_8bit=model_config["load_in_8bit"],
    full_finetuning=model_config["full_finetuning"],
    auto_model=AutoModelForCausalLM,
    trust_remote_code=model_config["trust_remote_code"],
    unsloth_force_compile=model_config["unsloth_force_compile"],
)

processor = AutoProcessor.from_pretrained(
    model_path, trust_remote_code=model_config["trust_remote_code"]
)

# ================================== LoRA =====================================
model = FastVisionModel.get_peft_model(model, **config["lora"])
model.print_trainable_parameters()

# ================================== Data =====================================
train_split = data_config["train_split"]
validation_split = data_config["validation_split"]
dataset = load_datasets(
    dataset_name=data_config["dataset_name"],
    subset=data_config["subset"],
    instruction=data_config["instruction"],
    splits=[train_split, validation_split],
    conversation_format="paddle",
)

print(dataset)
print(dataset[train_split][0])


# ==========================================================================
# OOM-safe SFT trainer
# ==========================================================================
class OOMSafeSFTTrainer(SFTTrainer):
    """Skip CUDA OOM batches instead of stopping the whole run."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._oom_skip_count = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            if num_items_in_batch is not None:
                return super().training_step(model, inputs, num_items_in_batch)
            return super().training_step(model, inputs)
        except torch.cuda.OutOfMemoryError:
            self._oom_skip_count += 1
            input_ids = inputs.get("input_ids", torch.zeros(1, 0))
            seq_len = input_ids.shape[-1]
            step = self.state.global_step

            with open("oom_skipped_batches.log", "a", encoding="utf-8") as f:
                f.write(
                    f"[OOM SKIP #{self._oom_skip_count}] TRAIN Step={step} | SeqLen={seq_len}\n"
                )

            print(
                f"\n[OOMSafeSFTTrainer] OOM at TRAIN step {step} "
                f"(SeqLen={seq_len}). Skipping...\n"
            )
            torch.cuda.empty_cache()
            model.zero_grad()
            return torch.tensor(
                0.0,
                device=next(model.parameters()).device,
                requires_grad=False,
            )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        try:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )
        except torch.cuda.OutOfMemoryError:
            input_ids = inputs.get("input_ids", torch.zeros(1, 0))
            seq_len = input_ids.shape[-1]
            step = self.state.global_step

            with open("oom_skipped_batches.log", "a", encoding="utf-8") as f:
                f.write(f"[OOM SKIP] EVAL Step={step} | SeqLen={seq_len}\n")

            print(
                f"\n[OOMSafeSFTTrainer] OOM at EVALUATION (SeqLen={seq_len}). "
                "Skipping...\n"
            )
            torch.cuda.empty_cache()
            dummy_loss = torch.tensor(
                0.0, device=next(model.parameters()).device
            )
            return (dummy_loss, None, None)


# ==========================================================================
# Build collator + trainer
# ==========================================================================
FastVisionModel.for_training(model)

custom_collator = PaddleOCRVLDataCollator(
    model=model,
    processor=processor,
    ignore_index=collator_config["ignore_index"],
    max_seq_length=collator_config["max_seq_length"],
    train_on_responses_only=collator_config["train_on_responses_only"],
    instruction_part=collator_config["instruction_part"],
    response_part=collator_config["response_part"],
    pad_to_multiple_of=collator_config["pad_to_multiple_of"],
)

training_args = resolve_precision(training_config, torch.cuda.is_bf16_supported())
training_args["output_dir"] = str(resolve_path(training_args["output_dir"]))

trainer = OOMSafeSFTTrainer(
    model=model,
    tokenizer=processor.tokenizer,
    data_collator=custom_collator,
    train_dataset=dataset[train_split],
    eval_dataset=dataset[validation_split],
    args=SFTConfig(**training_args),
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
    title="PaddleOCR-VL Khmer LoRA",
)

# ================================== Save =====================================
LORA_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
model.save_pretrained(LORA_OUTPUT_PATH)
processor.tokenizer.save_pretrained(LORA_OUTPUT_PATH)
