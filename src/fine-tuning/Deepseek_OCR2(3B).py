import os

os.environ["UNSLOTH_COMPILE_LOCATION"] = "/home/ltu_team/workspace/leangchhay/.cache/unsloth_compiled_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from transformers import AutoModel, TrainerCallback, EarlyStoppingCallback
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

model, tokenizer = FastVisionModel.from_pretrained(
    "../../../models/deepseek_ocr2/deepseek_ocr2_baseline",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    auto_model = AutoModel,
    trust_remote_code = True,
    unsloth_force_compile = True,
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent))

from utils.telegram_alert import send_telegram

# ========================== Telegram Alert Callback ==========================
class StepAlertCallback(TrainerCallback):
    def __init__(self, interval=20):
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step

        message = f"Fine-Tuning DeepSeek-OCR2(3B) for Khmer OCR (cv4dc2026|v1) - Progress Update: step {step}"
        
        if step > 0 and step % self.interval == 0:
            send_telegram(message)

        return control
    
model = FastVisionModel.get_peft_model(
    model,
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

from datasets import load_dataset
DATASET_BASE_DIR = "data/input"
dataset = load_dataset(
    "json",
    data_files={
        "train": f"{DATASET_BASE_DIR}/annotations/train.jsonl",
        "test": f"{DATASET_BASE_DIR}/annotations/test.jsonl", 
        "validation": f"{DATASET_BASE_DIR}/annotations/val.jsonl",
    }
)

print(dataset)
print(dataset["train"][0])

# check if all images path reached
import os

missing = []
checked = 0

for split_name in dataset.keys():
    split_data = dataset[split_name]
    for idx in range(len(split_data)):
        example = split_data[idx]
        for msg in example["messages"]:
            images = msg.get("images")
            if not images:
                continue
            for img_path in images:
                checked += 1
                if not os.path.isfile(img_path):
                    missing.append((split_name, idx, img_path))

print(f"Checked {checked} image references.")
print(f"Missing: {len(missing)}")

if missing:
    print("\nFirst 20 missing examples:")
    for split_name, idx, path in missing[:20]:
        print(f"  [{split_name} row {idx}] -> {path}")

    with open("missing_images.txt", "w") as f:
        for split_name, idx, path in missing:
            f.write(f"[{split_name} row {idx}] {path}\n")
    print(f"\nFull list written to missing_images.txt")
else:
    print("All image paths resolved successfully.")
    
# @title Create datacollator
import torch
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None
from torch.nn.utils.rnn import pad_sequence
import io

from deepseek_ocr2.modeling_deepseekocr2 import (
    format_messages,
    text_encode,
    BasicImageTransform,
    dynamic_preprocess,
)

@dataclass
class DeepSeekOCR2DataCollator:
    """
    Args:
        tokenizer: Tokenizer
        model: Model
        image_size: Size for image patches (default: 768)
        base_size: Size for global view (default: 1024)
        crop_mode: Whether to use dynamic cropping for large images
        train_on_responses_only: If True, only train on assistant responses (mask user prompts)
    """
    tokenizer: Any
    model: Any
    image_size: int = 768
    base_size: int = 1024
    crop_mode: bool = True
    image_token_id: int = 128815
    train_on_responses_only: bool = True

    def __init__(
        self,
        tokenizer,
        model,
        image_size: int = 768,
        base_size: int = 1024,
        crop_mode: bool = True,
        train_on_responses_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.image_token_id = 128815
        self.dtype = model.dtype  # Get dtype from model
        self.train_on_responses_only = train_on_responses_only

        self.image_transform = BasicImageTransform(
            mean = (0.5, 0.5, 0.5),
            std = (0.5, 0.5, 0.5),
            normalize = True
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        # Get BOS token ID from tokenizer
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            self.bos_id = tokenizer.bos_token_id
        else:
            self.bos_id = 0
            print(f"Warning: tokenizer has no bos_token_id, using default: {self.bos_id}")

    def deserialize_image(self, image_data) -> Image.Image:
        """Convert image data to PIL Image in RGB mode"""
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, str):
            with Image.open(image_data) as img:
                return img.convert("RGB")
        elif isinstance(image_data, dict) and 'bytes' in image_data:
            image_bytes = image_data['bytes']
            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

    def calculate_image_token_count(self, image: Image.Image, crop_ratio: Tuple[int, int]) -> int:
        """Calculate the number of tokens this image will generate"""
        num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

        width_crop_num, height_crop_num = crop_ratio

        if self.crop_mode:
            img_tokens = num_queries_base * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                img_tokens += (num_queries * width_crop_num) * (num_queries * height_crop_num)
        else:
            img_tokens = num_queries * num_queries + 1

        return img_tokens

    def process_image(self, image: Image.Image) -> Tuple[List, List, List, List, Tuple[int, int]]:
        """
        Process a single image based on crop_mode and size thresholds

        Returns:
            Tuple of (images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio)
        """
        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        if self.crop_mode:
            # Determine crop ratio based on image size
            if image.size[0] <= 768 and image.size[1] <= 768:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image, min_num = 2, max_num = 4,
                    image_size = self.image_size, use_thumbnail = False
                )

            # Process global view with padding
            global_view = ImageOps.pad(
                image, (self.base_size, self.base_size),
                color = tuple(int(x * 255) for x in self.image_transform.mean)
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            # Process local views (crops) if applicable
            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(
                        self.image_transform(crop_img).to(self.dtype)
                    )

            # Calculate image tokens
            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

            tokenized_image = ([self.image_token_id] * num_queries_base) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num)) * (
                    num_queries * height_crop_num)

        else:  # crop_mode = False
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])

            # For smaller base sizes, resize; for larger, pad
            if self.base_size <= 768:
                resized_image = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_list.append(self.image_transform(resized_image).to(self.dtype))
            else:
                global_view = ImageOps.pad(
                    image, (self.base_size, self.base_size),
                    color = tuple(int(x * 255) for x in self.image_transform.mean)
                )
                images_list.append(self.image_transform(global_view).to(self.dtype))

            num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries) * num_queries
            tokenized_image += [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio

    def process_single_sample(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Process a single conversation into model inputs.
        """

        # --- 1. Setup ---
        images = []
        for message in messages:
            if "images" in message and message["images"]:
                for img_data in message["images"]:
                    if img_data is not None:
                        pil_image = self.deserialize_image(img_data)
                        images.append(pil_image)

        if not images:
            raise ValueError("No images found in sample. Please ensure all samples contain images.")

        tokenized_str = []
        images_seq_mask = []
        images_list, images_crop_list, images_spatial_crop = [], [], []

        prompt_token_count = -1 # Index to start training
        assistant_started = False
        image_idx = 0

        # Add BOS token at the very beginning
        tokenized_str.append(self.bos_id)
        images_seq_mask.append(False)

        for message in messages:
            role = message["role"]
            content = message["content"]

            # Check if this is the assistant's turn
            if role == "<|Assistant|>":
                if not assistant_started:
                    # This is the split point. All tokens added *so far*
                    # are part of the prompt.
                    prompt_token_count = len(tokenized_str)
                    assistant_started = True

                # Append the EOS token string to the *end* of assistant content
                content = f"{content.strip()} {self.tokenizer.eos_token}"

            # Split this message's content by the image token
            text_splits = content.split('<image>')

            for i, text_sep in enumerate(text_splits):
                # Tokenize the text part
                tokenized_sep = text_encode(self.tokenizer, text_sep, bos = False, eos = False)
                tokenized_str.extend(tokenized_sep)
                images_seq_mask.extend([False] * len(tokenized_sep))

                # If this text is followed by an <image> tag
                if i < len(text_splits) - 1:
                    if image_idx >= len(images):
                        raise ValueError(
                            f"Data mismatch: Found '<image>' token but no corresponding image."
                        )

                    # Process the image
                    image = images[image_idx]
                    img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(image)

                    images_list.extend(img_list)
                    images_crop_list.extend(crop_list)
                    images_spatial_crop.extend(spatial_crop)

                    # Add image placeholder tokens
                    tokenized_str.extend(tok_img)
                    images_seq_mask.extend([True] * len(tok_img))

                    image_idx += 1 # Move to the next image

        # --- 3. Validation and Final Prep ---
        if image_idx != len(images):
            raise ValueError(
                f"Data mismatch: Found {len(images)} images but only {image_idx} '<image>' tokens were used."
            )

        # If we never found an assistant message, we're in a weird state
        # (e.g., user-only prompt). We mask everything.
        if not assistant_started:
            print("Warning: No assistant message found in sample. Masking all tokens.")
            prompt_token_count = len(tokenized_str)

        # Prepare image tensors
        images_ori = torch.stack(images_list, dim = 0)
        images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype = torch.long)

        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim = 0)
        else:
            images_crop = torch.zeros((1, 3, self.base_size, self.base_size), dtype = self.dtype)

        return {
            "input_ids": torch.tensor(tokenized_str, dtype = torch.long),
            "images_seq_mask": torch.tensor(images_seq_mask, dtype = torch.bool),
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop_tensor,
            "prompt_token_count": prompt_token_count, # This is now accurate
        }
        
    def deserialize_image(self, image_data) -> Image.Image:
        """Convert image data to PIL Image in RGB mode"""

        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")

        elif isinstance(image_data, str):
            # Handle image path
            return Image.open(image_data).convert("RGB")

        elif isinstance(image_data, dict) and 'bytes' in image_data:
            image_bytes = image_data['bytes']
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB")

        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_data = []
        batch_errors = []

        for feature in features:
            try:
                processed = self.process_single_sample(feature['messages'])
                batch_data.append(processed)
            except Exception as e:
                err_entry = {
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
                batch_errors.append(err_entry)
                try:
                    with open("collate_errors.log", "a", encoding="utf-8") as f:
                        f.write(json.dumps(err_entry) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                except Exception:
                    pass
                continue

        if not batch_data:
            first = batch_errors[0] if batch_errors else {"error_type": "unknown", "error": "unknown"}
            raise ValueError(
                f"No valid samples in batch. First error: {first['error_type']}: {first['error']}"
            )

        # Extract lists
        input_ids_list = [item['input_ids'] for item in batch_data]
        images_seq_mask_list = [item['images_seq_mask'] for item in batch_data]
        prompt_token_counts = [item['prompt_token_count'] for item in batch_data]

        # Pad sequences
        input_ids = pad_sequence(input_ids_list, batch_first = True, padding_value = self.tokenizer.pad_token_id)
        images_seq_mask = pad_sequence(images_seq_mask_list, batch_first = True, padding_value = False)

        # TRUNCATE to avoid CUBLAS_STATUS_EXECUTION_FAILED / Illegal Memory Access in RoPE
        MAX_SEQ_LEN = 4096
        if input_ids.shape[1] > MAX_SEQ_LEN:
            input_ids = input_ids[:, :MAX_SEQ_LEN]
            images_seq_mask = images_seq_mask[:, :MAX_SEQ_LEN]

        # Create labels
        labels = input_ids.clone()

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Mask image tokens (model shouldn't predict these)
        labels[images_seq_mask] = -100

        # Mask user prompt tokens when train_on_responses_only = True (only train on assistant responses)
        if self.train_on_responses_only:
            for idx, prompt_count in enumerate(prompt_token_counts):
                if prompt_count > 0:
                    labels[idx, :prompt_count] = -100

        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Prepare images batch (list of tuples)
        images_batch = []
        for item in batch_data:
            images_batch.append((item['images_crop'], item['images_ori']))

        # Stack spatial crop info
        images_spatial_crop = torch.cat([item['images_spatial_crop'] for item in batch_data], dim = 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }
        
from transformers import Trainer, TrainingArguments
from oom_safe_trainer import OOMSafeTrainer
from unsloth import is_bf16_supported
FastVisionModel.for_training(model) # Enable for training!
data_collator = DeepSeekOCR2DataCollator(
    tokenizer = tokenizer,
    model = model,
    image_size = 768,
    base_size = 1024,
    crop_mode = True,
    train_on_responses_only = True,
)

trainer = OOMSafeTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = data_collator, 
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"],
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 5,
        per_device_eval_batch_size = 2,
        eval_accumulation_steps = 1,
        max_grad_norm = 1.0,
        warmup_steps = 5,
        
        eval_strategy="steps", 
        eval_steps = 1600,
        # eval_steps = 1000, 
        # max_eval_samples = 4027,
            
        # max_steps = 20,
        num_train_epochs = 2, # Set this instead of max_steps for full training runs
        learning_rate = 1e-4,
        logging_steps = 1,
        save_steps = 1600,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        fp16 = not is_bf16_supported(),  # Use fp16 if bf16 is not supported
        bf16 = is_bf16_supported(),  # Use bf16 if supported
        output_dir = "./data/output/checkpoints",
        report_to = "none",     
        dataloader_num_workers = 4,
        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True
    ),
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.001,
        ),
        StepAlertCallback(interval=100)
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
message = f"Starting Fine-tuning DeepSeek-OCR2 for cv4dc | v2: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
send_telegram(message)

try:
    trainer_stats = trainer.train(resume_from_checkpoint=True)
    # trainer_stats = trainer.train()

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
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    try:
        import json, os

        trainer.save_state()  # canonical trainer_state.json in output_dir

        os.makedirs("./data/output/deepseek_ocr_lora", exist_ok=True)
        with open("./data/output/deepseek_ocr_lora/training_history.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        model.save_pretrained("./data/output/deepseek_ocr_lora")
        tokenizer.save_pretrained("./data/output/deepseek_ocr_lora")
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