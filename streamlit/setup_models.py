
"""Download OralSense AI model weights from Hugging Face.

The DINOv2 source directory must already exist locally.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

BASE = Path(__file__).resolve().parent
DEST = BASE / "models"

REPO_ID = "Divyasri28/OralSense-AI-Models"

REQUIRED = [
    "ConvNeXtTiny_best.pth",
    "DINOv2SVM_best.joblib",
    "OralSense_Adaptive_Policy.joblib",
    "dinov2_vits14_pretrain.pth",
]


def setup_models():
    DEST.mkdir(parents=True, exist_ok=True)

    source = DEST / "dinov2_hub_source" / "hubconf.py"

    if not source.is_file():
        raise FileNotFoundError(
            "DINOv2 source is missing. "
            "Restore models/dinov2_hub_source from "
            "your trusted Kaggle backup."
        )

    for filename in REQUIRED:
        target = DEST / filename

        if target.is_file():
            print(f"Already exists: {filename}")
            continue

        print(f"Downloading {filename}...")

        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=DEST,
        )

        print(f"Downloaded: {filename}")

    print("All OralSense AI model files are ready!")


if __name__ == "__main__":
    setup_models()
