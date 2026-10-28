from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "lora.yaml"


def load_config(path=CONFIG_PATH, section=None):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if section is not None:
        if section not in config:
            raise KeyError(f"Config section '{section}' not found in {path}")
        config = config[section]
    return config


def resolve_path(path):
    """Resolve a config path relative to the src/ directory."""
    path = Path(path).expanduser()
    return (path if path.is_absolute() else SCRIPT_DIR / path).resolve()
