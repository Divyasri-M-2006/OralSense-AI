"""Extract OralSense runtime artifacts from YOUR trusted Kaggle backup ZIP.

Usage: python setup_models.py "C:\\Users\\...\\ORALSENSE_FINAL_BACKUP.zip"
Only the required trained models and DINOv2 source are extracted.
"""
from pathlib import Path
from zipfile import ZipFile
import argparse
import shutil

BASE = Path(__file__).resolve().parent
DEST = BASE / "models"
REQUIRED = {
    "results/ConvNeXtTiny_best.pth": "ConvNeXtTiny_best.pth",
    "results/DINOv2SVM_best.joblib": "DINOv2SVM_best.joblib",
    "results/OralSense_Adaptive_Policy.joblib": "OralSense_Adaptive_Policy.joblib",
    "pretrained_weights/dinov2_vits14_pretrain.pth": "dinov2_vits14_pretrain.pth",
}
PREFIX = "dinov2_hub_source/"


def setup_models(backup_zip: Path) -> None:
    if not backup_zip.is_file():
        raise FileNotFoundError(f"Backup not found: {backup_zip}")
    with ZipFile(backup_zip) as archive:
        names = set(archive.namelist())
        missing = [name for name in REQUIRED if name not in names]
        if PREFIX + "hubconf.py" not in names:
            missing.append(PREFIX + "hubconf.py")
        if missing:
            raise ValueError("Your backup is missing:\n  " + "\n  ".join(missing))
        DEST.mkdir(parents=True, exist_ok=True)
        for source, filename in REQUIRED.items():
            target = DEST / filename
            with archive.open(source) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"✓ {filename} ({target.stat().st_size / (1024**2):.1f} MB)")

        source_count = 0
        for member in archive.infolist():
            if not member.filename.startswith(PREFIX) or member.is_dir():
                continue
            subpath = Path(member.filename[len(PREFIX):])
            if not subpath.parts or subpath.is_absolute() or ".." in subpath.parts:
                continue
            if "__pycache__" in subpath.parts or any(part.startswith(".") for part in subpath.parts):
                continue
            target = (DEST / "dinov2_hub_source" / subpath).resolve()
            if not target.is_relative_to((DEST / "dinov2_hub_source").resolve()):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            source_count += 1
        print(f"✓ DINOv2 source files: {source_count}")
    print("\nREADY: start the website with 'python -m streamlit run app.py'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup_zip", type=Path, help="Path to your ORALSENSE_FINAL_BACKUP.zip")
    args = parser.parse_args()
    setup_models(args.backup_zip)
