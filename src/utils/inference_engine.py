import gc
import sys
import tempfile
from pathlib import Path

sys.modules["flash_attn"] = None
sys.modules["torchao"] = None

import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)


class DeepSeekOCREngine:
    """Load a DeepSeek-OCR-2 baseline or LoRA adapter and run OCR on PIL images."""

    def __init__(self, model_path, adapter_path=None, torch_dtype=torch.bfloat16):
        if adapter_path is not None and not Path(adapter_path).exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            model_path,
            _attn_implementation="eager",
            trust_remote_code=True,
            use_safetensors=True,
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()

        self.model = model.eval().cuda().to(torch_dtype)

    def predict(
        self,
        image,
        prompt="<image>\nFree OCR.",
        image_size=768,
        base_size=1024,
        crop_mode=True,
        **kwargs,
    ):
        with tempfile.TemporaryDirectory(prefix="deepseek_ocr_") as output_dir:
            image_path = Path(output_dir) / "input.png"
            image.convert("RGB").save(image_path)

            self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                output_path=output_dir,
                image_size=image_size,
                base_size=base_size,
                crop_mode=crop_mode,
                save_results=True,
                test_compress=False,
            )

            result_path = Path(output_dir) / "result.mmd"
            if not result_path.is_file():
                raise RuntimeError(
                    f"DeepSeek-OCR did not create an output file: {result_path}"
                )
            return result_path.read_text(encoding="utf-8").strip()

    def close(self):
        del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


class PaddleOCREngine:
    """Load a PaddleOCR-VL baseline or LoRA adapter and run OCR on PIL images."""

    def __init__(self, model_path, adapter_path=None, torch_dtype=torch.bfloat16):
        if adapter_path is not None and not Path(adapter_path).exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

        import peft.tuners.lora.model as lora_model

        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch_dtype,
            attn_implementation="sdpa",
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()

        self.model = model.eval().cuda()

    def predict(self, image, prompt="OCR:", max_new_tokens=4096, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            image,
            text_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)

        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        generated_ids = output_ids[:, prompt_length:]
        return self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    def close(self):
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()
