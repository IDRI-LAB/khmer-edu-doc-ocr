import gc
import sys
import tempfile
from pathlib import Path

sys.modules["flash_attn"] = None
sys.modules["torchao"] = None

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


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
