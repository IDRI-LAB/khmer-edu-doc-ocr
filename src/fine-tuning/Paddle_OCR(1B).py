import os

os.environ["UNSLOTH_COMPILE_LOCATION"] = "/home/ltu_team/workspace/leangchhay/.cache/unsloth_compiled_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

from unsloth import FastVisionModel  # FastLanguageModel for LLMs
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from transformers import AutoModelForCausalLM, AutoProcessor, TrainerCallback, EarlyStoppingCallback

from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent))

from utils.telegram_alert import send_telegram

# ==========================================================================
# Load model + processor
# ==========================================================================
# This mirrors unslothai/notebooks: nb/Paddle_OCR_(1B)_Vision.ipynb exactly -
# PaddleOCR-VL is loaded via FastVisionModel + AutoModelForCausalLM, and
# needs a separate AutoProcessor (unlike DeepSeek-OCR2 which only needed
# the tokenizer). trust_remote_code is required for both.
model_path = "unsloth/PaddleOCR-VL"

model, tokenizer = FastVisionModel.from_pretrained(
    model_path,
    max_seq_length = 4096,        # Choose any for long context!
    load_in_4bit = False,         # 4bit uses much less memory
    load_in_8bit = False,         # A bit more accurate, uses 2x memory
    full_finetuning = False,      # IMPORTANT: must be False for LoRA - a frozen
                                   # base model + small trainable LoRA adapters
                                   # (attached below via get_peft_model). Setting
                                   # this True would make the whole base model
                                   # trainable regardless of get_peft_model.
    auto_model = AutoModelForCausalLM,
    trust_remote_code = True,
    unsloth_force_compile = True,
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code = True)


# ========================== Telegram Alert Callback ==========================
class StepAlertCallback(TrainerCallback):
    def __init__(self, interval = 20):
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step

        message = f"Fine-Tuning PaddleOCR-VL(1B) for Khmer OCR (cv4dc2026|v1) - Progress Update: step {step}"

        if step > 0 and step % self.interval == 0:
            send_telegram(message)

        return control


# ==========================================================================
# LoRA adapters
# ==========================================================================
# target_modules taken directly from the official Paddle_OCR_(1B)_Vision.ipynb -
# note the extra out_proj/fc1/fc2/linear_1/linear_2 modules compared to
# DeepSeek-OCR2's projection-only list; PaddleOCR-VL's architecture exposes
# these additional linear layers.
model = FastVisionModel.get_peft_model(
    model,
    r = 64,             # The larger, the higher the accuracy, but might overfit
    lora_alpha = 64,    # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "out_proj", "fc1", "fc2",
        "linear_1", "linear_2",
    ],
)

# Sanity check: confirms LoRA is actually active. Trainable% should be a
# small fraction (a few percent) of all params - if it's close to 100%,
# something upstream (e.g. full_finetuning=True) is overriding LoRA.
model.print_trainable_parameters()

# ==========================================================================
# Dataset
# ==========================================================================
# Expected local JSONL schema per row - already in the final conversation
# shape, except "image" values are file path STRINGS instead of decoded
# PIL.Image objects:
#
# {
#   "images": ["path/to/image.png"],
#   "messages": [
#     {"role": "user", "content": [
#         {"type": "text", "text": "OCR:"},
#         {"type": "image", "image": "path/to/image.png"}
#     ]},
#     {"role": "assistant", "content": [
#         {"type": "text", "text": "some ocr text"}
#     ]}
#   ]
# }
#
# convert_to_conversation() below just opens each path string into a PIL.Image
# (what UnslothVisionDataCollator actually needs) - nothing else about the
# structure changes.
from datasets import load_dataset
from PIL import Image

DATASET_BASE_DIR = "data/input"
dataset = load_dataset(
    "json",
    data_files = {
        "train": f"{DATASET_BASE_DIR}/annotations/train.jsonl",
        "test": f"{DATASET_BASE_DIR}/annotations/test.jsonl",
        "validation": f"{DATASET_BASE_DIR}/annotations/val.jsonl",
    },
)

print(dataset)
print(dataset["train"][0])

# --- Validate all image paths resolve before training starts ---
# Set VERIFY_IMAGES_INTERACTIVE = True to list every resolved (and missing)
# image path and pause for a manual yes/no confirmation before training.
# Set it to False to skip the review entirely and go straight to training.
VERIFY_IMAGES_INTERACTIVE = True

import json as _json

_SPLIT_TO_FILENAME = {"train": "train", "test": "test", "validation": "val"}


def _collect_image_paths(base_dir):
    """Reads the raw JSONL and checks every path in each row's top-level
    "images" list exists on disk."""
    resolved, missing = [], []
    checked = 0
    for split_name, fname in _SPLIT_TO_FILENAME.items():
        raw_path = f"{base_dir}/annotations/{fname}.jsonl"
        with open(raw_path, "r", encoding = "utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = _json.loads(line)
                img_paths = row.get("images", []) or []
                for img_path in img_paths:
                    checked += 1
                    if img_path and os.path.isfile(img_path):
                        resolved.append((split_name, idx, img_path))
                    else:
                        missing.append((split_name, idx, img_path))
    return resolved, missing, checked


resolved, missing, checked = _collect_image_paths(DATASET_BASE_DIR)

print(f"Checked {checked} image references.")
print(f"Resolved: {len(resolved)}  |  Missing: {len(missing)}")

if VERIFY_IMAGES_INTERACTIVE:
    print("\n" + "=" * 60)
    print("IMAGE PATH REVIEW (VERIFY_IMAGES_INTERACTIVE = True)")
    print("=" * 60)

    print(f"\nResolved image paths ({len(resolved)}):")
    for split_name, idx, path in resolved:
        print(f"  [OK]      [{split_name} row {idx}] {path}")

    with open("resolved_images_review.txt", "w") as f:
        for split_name, idx, path in resolved:
            f.write(f"[{split_name} row {idx}] {path}\n")
    print(f"\nFull resolved list also written to resolved_images_review.txt")

    if missing:
        print(f"\nMissing image paths ({len(missing)}):")
        for split_name, idx, path in missing:
            print(f"  [MISSING] [{split_name} row {idx}] {path}")

        with open("missing_images.txt", "w") as f:
            for split_name, idx, path in missing:
                f.write(f"[{split_name} row {idx}] {path}\n")
        print("Full missing list written to missing_images.txt")
    else:
        print("\nAll image paths resolved successfully - none missing.")

    answer = input(
        "\nReview the paths above (or the .txt files just written). "
        "Proceed to training? [y/N]: "
    ).strip().lower()

    if answer not in ("y", "yes"):
        print("Aborting before training, per user response.")
        sys.exit(0)

    print("Confirmed - proceeding to training.\n")
else:
    print(
        "VERIFY_IMAGES_INTERACTIVE = False -> skipping image path review, "
        "proceeding straight to training."
    )
    if missing:
        # Still fail loudly rather than silently training on broken data.
        with open("missing_images.txt", "w") as f:
            for split_name, idx, path in missing:
                f.write(f"[{split_name} row {idx}] {path}\n")
        raise FileNotFoundError(
            f"{len(missing)} image path(s) could not be resolved. "
            f"See missing_images.txt for details. Set VERIFY_IMAGES_INTERACTIVE = True "
            f"to review interactively, or fix the paths and re-run."
        )

# --- Image path strings -> PIL.Image objects happens inside the collator now ---
# (see PaddleOCRVLDataCollator below) - matching how DeepSeekOCR2DataCollator
# opened images inside its own __call__, only for the current batch. No
# per-row/per-dataset transform is needed here; the raw dataset (still just
# path strings) is passed straight to the trainer.
converted_train = dataset["train"]
converted_validation = dataset["validation"]

print(converted_train[0])  # still just path strings at this point - safe & instant


# ==========================================================================
# OOM-safe trainer (same pattern as DeepSeek-OCR2 script, subclassing
# TRL's SFTTrainer instead of transformers.Trainer since vision finetuning
# for PaddleOCR-VL goes through SFTTrainer)
# ==========================================================================
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator


class OOMSafeSFTTrainer(SFTTrainer):
    """
    An SFTTrainer subclass that catches CUDA Out-of-Memory errors on both
    training and evaluation steps, skips the offending batch, clears
    the GPU cache, and continues safely.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._oom_skip_count = 0

    def training_step(self, model, inputs, num_items_in_batch = None):
        try:
            if num_items_in_batch is not None:
                return super().training_step(model, inputs, num_items_in_batch)
            else:
                return super().training_step(model, inputs)
        except torch.cuda.OutOfMemoryError as e:
            self._oom_skip_count += 1
            input_ids = inputs.get("input_ids", torch.zeros(1, 0))
            seq_len = input_ids.shape[-1]
            step = self.state.global_step

            try:
                with open("oom_skipped_batches.log", "a", encoding = "utf-8") as f:
                    f.write(f"[OOM SKIP #{self._oom_skip_count}] TRAIN Step={step} | SeqLen={seq_len}\n")
            except Exception:
                pass

            print(f"\n[OOMSafeSFTTrainer] ⚠️  OOM at TRAIN step {step} (SeqLen={seq_len}). "
                  f"Skipping...\n")

            torch.cuda.empty_cache()
            model.zero_grad()

            return torch.tensor(0.0, device = next(model.parameters()).device,
                                 requires_grad = False)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
        try:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        except torch.cuda.OutOfMemoryError as e:
            input_ids = inputs.get("input_ids", torch.zeros(1, 0))
            seq_len = input_ids.shape[-1]
            step = self.state.global_step

            try:
                with open("oom_skipped_batches.log", "a", encoding = "utf-8") as f:
                    f.write(f"[OOM SKIP] EVAL Step={step} | SeqLen={seq_len}\n")
            except Exception:
                pass

            print(f"\n[OOMSafeSFTTrainer] ⚠️  OOM at EVALUATION (SeqLen={seq_len}). "
                  f"Skipping...\n")

            torch.cuda.empty_cache()

            dummy_loss = torch.tensor(0.0, device = next(model.parameters()).device)
            return (dummy_loss, None, None)


# ==========================================================================
# Build collator + trainer
# ==========================================================================
FastVisionModel.for_training(model)  # Enable for training!


class PaddleOCRVLDataCollator:
    """
    Thin wrapper around UnslothVisionDataCollator that resolves each row's
    "image" path strings into PIL.Image objects at collation time - i.e.
    only for the handful of rows in the batch currently being assembled -
    the same way DeepSeekOCR2DataCollator opened images inside its own
    __call__ instead of the whole dataset being pre-converted upfront.
    """

    def __init__(self, *args, **kwargs):
        self._inner = UnslothVisionDataCollator(*args, **kwargs)

    def __call__(self, features):
        resolved_features = []

        for row in features:
            path_to_image = {}

            def _open(path):
                img = path_to_image.get(path)
                if img is None:
                    img = Image.open(path).convert("RGB")
                    path_to_image[path] = img
                return img

            resolved_images = [_open(p) for p in row["images"]]

            new_messages = []
            for msg in row["messages"]:
                new_content = []
                for block in msg["content"]:
                    if block.get("type") == "image":
                        new_content.append({"type": "image", "image": _open(block["image"])})
                    else:
                        new_content.append(block)
                new_messages.append({"role": msg["role"], "content": new_content})

            resolved_features.append({"images": resolved_images, "messages": new_messages})

        return self._inner(resolved_features)


custom_collator = PaddleOCRVLDataCollator(
    model = model,
    processor = processor,
    ignore_index = -100,
    max_seq_length = 4096,
    train_on_responses_only = True,
    instruction_part = "User:",
    response_part = "\nAssistant:",
    pad_to_multiple_of = 8,
)

trainer = OOMSafeSFTTrainer(
    model = model,
    tokenizer = processor.tokenizer,
    data_collator = custom_collator,
    train_dataset = converted_train,
    eval_dataset = converted_validation,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 5,
        per_device_eval_batch_size = 2,
        eval_accumulation_steps = 1,
        max_grad_norm = 1.0,
        warmup_steps = 5,

        eval_strategy = "steps",
        eval_steps = 1600,

        # max_steps = 20,
        num_train_epochs = 2,  # Set this instead of max_steps for full training runs
        learning_rate = 5e-5,
        logging_steps = 1,
        save_steps = 1600,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "./data/output/checkpoints",
        report_to = "none",
        dataloader_num_workers = 4,

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 4096,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),

        metric_for_best_model = "eval_loss",
        load_best_model_at_end = True,
    ),
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience = 2,
            early_stopping_threshold = 0.001,
        ),
        StepAlertCallback(interval = 100),
    ],
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

import time
from datetime import datetime

start_time = time.time()
message = f"Starting Fine-tuning PaddleOCR-VL(1B) for cv4dc | v1: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
send_telegram(message)

try:
    # trainer_stats = trainer.train(resume_from_checkpoint = True)
    trainer_stats = trainer.train()

    end_time = time.time()
    duration_seconds = end_time - start_time

    hours, remainder = divmod(int(duration_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    summary = f"""
        {"=" * 30}
        Start Time : {time.ctime(start_time)}
        End Time   : {time.ctime(end_time)}
        Duration   : {hours}h {minutes}m {seconds}s
        {"=" * 30}
    """
    send_telegram(summary)
    send_telegram("The fine-tuning is done")

    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    try:
        import json, os as _os

        trainer.save_state()  # canonical trainer_state.json in output_dir

        _os.makedirs("./data/output/paddleocr_vl_lora", exist_ok = True)
        with open("./data/output/paddleocr_vl_lora/training_history.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent = 2)
        model.save_pretrained("./data/output/paddleocr_vl_lora")
        processor.tokenizer.save_pretrained("./data/output/paddleocr_vl_lora")

        # --- Plot & save training (and eval, if present) loss curve(s) ---
        try:
            import matplotlib
            matplotlib.use("Agg")  # no display available on a headless server
            import matplotlib.pyplot as plt

            log_history = trainer.state.log_history

            train_steps = [e["step"] for e in log_history if "loss" in e]
            train_losses = [e["loss"] for e in log_history if "loss" in e]

            eval_steps = [e["step"] for e in log_history if "eval_loss" in e]
            eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]

            # --- Combined graph: train loss + eval loss together ---
            plt.figure(figsize = (10, 6))
            if train_losses:
                plt.plot(train_steps, train_losses, label = "Train loss", linewidth = 1.5)
            if eval_losses:
                plt.plot(eval_steps, eval_losses, label = "Eval loss", linewidth = 2, marker = "o")

            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("PaddleOCR-VL Fine-tuning Loss Curve")
            plt.legend()
            plt.grid(True, alpha = 0.3)
            plt.tight_layout()

            loss_curve_path = "./data/output/paddleocr_vl_lora/training_loss_curve.png"
            plt.savefig(loss_curve_path, dpi = 150)
            plt.close()

            print(f"Saved training loss curve to {loss_curve_path}")

            # --- Separate train-only loss graph ---
            if train_losses:
                plt.figure(figsize = (10, 6))
                plt.plot(train_steps, train_losses, label = "Train loss", linewidth = 1.5, color = "tab:blue")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.title("PaddleOCR-VL Fine-tuning - Train Loss")
                plt.legend()
                plt.grid(True, alpha = 0.3)
                plt.tight_layout()

                train_only_path = "./data/output/paddleocr_vl_lora/train_loss_curve.png"
                plt.savefig(train_only_path, dpi = 150)
                plt.close()
                print(f"Saved train-only loss curve to {train_only_path}")

            # --- Separate eval-only loss graph ---
            if eval_losses:
                plt.figure(figsize = (10, 6))
                plt.plot(eval_steps, eval_losses, label = "Eval loss", linewidth = 2, marker = "o", color = "tab:orange")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.title("PaddleOCR-VL Fine-tuning - Eval Loss")
                plt.legend()
                plt.grid(True, alpha = 0.3)
                plt.tight_layout()

                eval_only_path = "./data/output/paddleocr_vl_lora/eval_loss_curve.png"
                plt.savefig(eval_only_path, dpi = 150)
                plt.close()
                print(f"Saved eval-only loss curve to {eval_only_path}")

        except Exception as plot_err:
            # Don't fail the whole run just because plotting failed -
            # the raw numbers are already safe in training_history.json.
            print(f"Warning: could not save loss curve plot: {type(plot_err).__name__}: {plot_err}")

    except Exception as e:
        send_telegram(f"ERROR saving model: {type(e).__name__}: {e}")
        raise

except Exception as e:
    end_time = time.time()
    duration_seconds = end_time - start_time
    hours, remainder = divmod(int(duration_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    error_message = f"""
        ERROR during fine-tuning!
        {"=" * 30}
        Error Type : {type(e).__name__}
        Error Msg  : {e}
        {"=" * 30}
        Start Time : {time.ctime(start_time)}
        Failed At  : {time.ctime(end_time)}
        Elapsed    : {hours}h {minutes}m {seconds}s
        {"=" * 30}
    """
    send_telegram(error_message)
    raise