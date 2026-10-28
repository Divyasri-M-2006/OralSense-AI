"""Prepare DINOv2 source and download OralSense AI model weights."""

from pathlib import Path
import subprocess
import shutil

from huggingface_hub import hf_hub_download


BASE = Path(__file__).resolve().parent
DEST = BASE / "models"

REPO_ID = "Divyasri28/OralSense-AI-Models"

# Official DINOv2 repository
DINO_REPO = "https://github.com/facebookresearch/dinov2.git"

# Exact DINOv2 version tested successfully
DINO_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"

REQUIRED = [
    "ConvNeXtTiny_best.pth",
    "DINOv2SVM_best.joblib",
    "OralSense_Adaptive_Policy.joblib",
    "dinov2_vits14_pretrain.pth",
]


def setup_dinov2():
    source_dir = DEST / "dinov2_hub_source"
    hubconf = source_dir / "hubconf.py"

    if hubconf.is_file():
        print("DINOv2 source already exists.")
        return

    print("Downloading official DINOv2 source...")

    # Clone into a temporary directory first
    temp_dir = DEST / "dinov2_download_temp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    try:
        # Step 1: Clone the official DINOv2 repository
        subprocess.run(
            [
                "git",
                "clone",
                DINO_REPO,
                str(temp_dir),
            ],
            check=True,
        )

        # Step 2: Checkout the exact tested DINOv2 commit
        subprocess.run(
            [
                "git",
                "-C",
                str(temp_dir),
                "checkout",
                "--detach",
                DINO_COMMIT,
            ],
            check=True,
        )

        # Step 3: Move downloaded source into models directory
        temp_dir.rename(source_dir)

        # Step 4: Verify DINOv2 source
        if not hubconf.is_file():
            raise FileNotFoundError(
                "DINOv2 download completed, but hubconf.py is missing."
            )

        print("DINOv2 source downloaded successfully!")

    finally:
        # Remove temporary directory if it still exists
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def setup_models():
    DEST.mkdir(parents=True, exist_ok=True)

    # Prepare DINOv2 source
    setup_dinov2()

    # Download trained models from Hugging Face
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