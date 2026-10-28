import io

from PIL import Image
from datasets import load_dataset


def load_test_split(dataset_name, subset="default", split="test"):
    return load_dataset(dataset_name, subset, split=split)


def get_row_id(example, index):
    book_id = example.get("book_id")
    page = example.get("page")
    if book_id is not None and page is not None:
        return f"{book_id}_p{page}"
    return str(index)


def get_ground_truth(example):
    return example.get("text", "")


def get_image(example):
    image = example.get("image")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and "bytes" in image:
        return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    raise ValueError(f"Unsupported image format: {type(image)}")
