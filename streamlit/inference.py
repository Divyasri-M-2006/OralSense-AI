"""OralSense AI: exact frozen Kaggle inference policy, for local Streamlit use.

No model training, no fabricated predictions, and no image persistence.
Important: checkpoint/joblib artifacts must come from the user's trusted Kaggle export.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import RLock
from time import perf_counter

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import convnext_tiny

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
CLASS_NAMES = [
    "Calculus", "Caries", "Gingivitis", "Hypodontia", "Mouth Ulcer", "Tooth Discoloration"
]
IMG_SIZE = 224
REQUIRED_FILES = {
    "ConvNeXt-Tiny checkpoint": MODEL_DIR / "ConvNeXtTiny_best.pth",
    "DINOv2–SVM classifier": MODEL_DIR / "DINOv2SVM_best.joblib",
    "Frozen adaptive policy": MODEL_DIR / "OralSense_Adaptive_Policy.joblib",
    "DINOv2 backbone weights": MODEL_DIR / "dinov2_vits14_pretrain.pth",
    "DINOv2 source": MODEL_DIR / "dinov2_hub_source" / "hubconf.py",
}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def missing_model_files() -> dict[str, str]:
    return {name: str(path.relative_to(ROOT)) for name, path in REQUIRED_FILES.items() if not path.is_file()}


def open_image(image_data: bytes) -> Image.Image:
    """Read supported images, respect EXIF orientation, reject unsafe huge images."""
    if not image_data or len(image_data) > 15 * 1024 * 1024:
        raise ValueError("Please upload a JPG, PNG or WEBP image smaller than 15 MB.")
    try:
        with Image.open(BytesIO(image_data)) as image:
            if image.width * image.height > 24_000_000:
                raise ValueError("Please choose an image smaller than 24 megapixels.")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc


def image_to_tensor(image: Image.Image) -> tuple[torch.Tensor, Image.Image]:
    """Same EXIF/RGB, neutral-grey ImageOps.pad, ToTensor and normalization as Kaggle."""
    padded = ImageOps.pad(
        image, (IMG_SIZE, IMG_SIZE),
        method=Image.Resampling.BICUBIC, color=(127, 127, 127),
    )
    return TRANSFORM(padded).unsqueeze(0), padded


def calculate_gate_features(primary: np.ndarray, specialist: np.ndarray) -> np.ndarray:
    """The *exact* eight features and order used in Kaggle Cell 23."""
    p, s = np.asarray(primary), np.asarray(specialist)
    if p.ndim != 2 or p.shape != s.shape or p.shape[1] != 6:
        raise ValueError("Gate expects matching Nx6 class probabilities.")
    ps, ss = np.sort(p, axis=1), np.sort(s, axis=1)
    pi, si = p.argmax(axis=1), s.argmax(axis=1)
    ix = np.arange(len(p))
    p_entropy = -np.sum(p * np.log(np.clip(p, 1e-9, 1)), axis=1)
    s_entropy = -np.sum(s * np.log(np.clip(s, 1e-9, 1)), axis=1)
    return np.column_stack([
        ps[:, -1], ss[:, -1],
        ps[:, -1] - ps[:, -2], ss[:, -1] - ss[:, -2],
        p_entropy, s_entropy,
        p[ix, si], s[ix, pi],
    ])


@dataclass
class GradCAMImages:
    padded: Image.Image
    heatmap: Image.Image
    overlay: Image.Image
    class_name: str


class OralSenseEngine:
    def __init__(self) -> None:
        missing = missing_model_files()
        if missing:
            raise FileNotFoundError("Missing model files: " + ", ".join(missing.values()))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = RLock()

        # SECURITY: torch.load and joblib must ONLY receive trusted user-created files.
        self.policy = joblib.load(REQUIRED_FILES["Frozen adaptive policy"])
        svm_package = joblib.load(REQUIRED_FILES["DINOv2–SVM classifier"])
        self.specialist = svm_package["classifier"]
        self.gate = self.policy["gate"]
        self.conf_threshold = float(self.policy["primary_confidence_threshold"])
        self.gate_threshold = float(self.policy["gate_threshold"])

        if list(self.policy["class_names"]) != CLASS_NAMES:
            raise ValueError("Frozen model class order differs from application labels.")
        if not np.array_equal(self.specialist.classes_, np.arange(6)):
            raise ValueError("Saved DINOv2–SVM class indices are not 0–5.")
        if not 0 <= self.conf_threshold <= 1 or not 0 <= self.gate_threshold <= 1:
            raise ValueError("Frozen decision policy has invalid thresholds.")

        self.primary = convnext_tiny(weights=None)
        in_features = self.primary.classifier[2].in_features
        self.primary.classifier[2] = torch.nn.Linear(in_features, len(CLASS_NAMES))
        checkpoint = torch.load(
            REQUIRED_FILES["ConvNeXt-Tiny checkpoint"],
            map_location="cpu", weights_only=True,
        )
        self.primary.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.primary = self.primary.to(self.device).eval()

        # Pretrained=False avoids any network call. Load the user's exact weights.
        source_dir = str((MODEL_DIR / "dinov2_hub_source").resolve())
        self.dino = torch.hub.load(
            source_dir, "dinov2_vits14", source="local", pretrained=False,
        )
        raw_weights = torch.load(
            REQUIRED_FILES["DINOv2 backbone weights"],
            map_location="cpu", weights_only=True,
        )
        state = raw_weights.get("state_dict", raw_weights) if isinstance(raw_weights, dict) else raw_weights
        self.dino.load_state_dict(state, strict=True)
        self.dino = self.dino.to(self.device).eval()
        self.dino.requires_grad_(False)

    def predict(self, image_data: bytes) -> dict:
        image = open_image(image_data)
        tensor, _ = image_to_tensor(image)
        tensor = tensor.to(self.device)
        start = perf_counter()

        with self.lock, torch.inference_mode():
            with torch.autocast("cuda", enabled=(self.device.type == "cuda")):
                logits = self.primary(tensor)
            p = torch.softmax(logits.float(), dim=1).cpu().numpy()
            primary_idx = int(np.argmax(p[0]))
            conf = float(p[0, primary_idx])
            uncertain = bool(conf <= self.conf_threshold)

            specialist_idx = None
            specialist_probs = None
            gate_score = None
            switched = False
            outcome = "PRIMARY"
            final_idx = primary_idx
            if uncertain:
                features = self.dino(tensor).float().cpu().numpy()
                specialist_probs = self.specialist.predict_proba(features)
                specialist_idx = int(np.argmax(specialist_probs[0]))
                if specialist_idx != primary_idx:
                    features_gate = calculate_gate_features(p, specialist_probs)
                    gate_score = float(self.gate.predict_proba(features_gate)[0, 1])
                    if gate_score >= self.gate_threshold:
                        outcome = "SPECIALIST"
                        final_idx = specialist_idx
                        switched = True
                    else:
                        outcome = "REFER"
                        final_idx = None
                else:
                    outcome = "REFER"
                    final_idx = None

        return {
            "decision": outcome,
            "final_index": final_idx,
            "final_label": CLASS_NAMES[final_idx] if final_idx is not None else None,
            "primary_index": primary_idx,
            "primary_label": CLASS_NAMES[primary_idx],
            "primary_confidence": conf,
            "primary_probabilities": p[0].tolist(),
            "uncertain": uncertain,
            "specialist_consulted": uncertain,
            "specialist_index": specialist_idx,
            "specialist_label": CLASS_NAMES[specialist_idx] if specialist_idx is not None else None,
            "specialist_confidence": float(specialist_probs[0].max()) if specialist_probs is not None else None,
            "specialist_probabilities": specialist_probs[0].tolist() if specialist_probs is not None else None,
            "gate_score": gate_score,
            "switched": switched,
            "confidence_threshold": self.conf_threshold,
            "gate_threshold": self.gate_threshold,
            "runtime_seconds": round(perf_counter() - start, 3),
            "device": str(self.device),
        }

    def gradcam(self, image_data: bytes, primary_index: int) -> GradCAMImages:
        """Gradient-weighted class activation map from ConvNeXt's last spatial stage.

        Explains the PRIMARY class only, not the specialist/gate/referral decision.
        """
        import matplotlib

        image = open_image(image_data)
        tensor, padded = image_to_tensor(image)
        tensor = tensor.to(self.device)
        state: dict[str, torch.Tensor] = {}

        def activation_hook(_module, _args, output):
            # Forward hooks run during Grad-CAM's grad-enabled pass, not inference_mode.
            state["activation"] = output.detach()
            output.register_hook(lambda grad: state.__setitem__("gradient", grad.detach()))

        layer = self.primary.features[-1]
        hook = layer.register_forward_hook(activation_hook)
        try:
            with self.lock, torch.enable_grad():
                self.primary.zero_grad(set_to_none=True)
                # Use float32 to keep Grad-CAM gradients stable across CPU/GPU.
                logits = self.primary(tensor)
                logits[0, int(primary_index)].backward()
            if "activation" not in state or "gradient" not in state:
                raise RuntimeError("Could not capture gradients from the ConvNeXt feature layer.")
            activations = state["activation"]
            gradients = state["gradient"]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            mask = torch.relu((activations * weights).sum(dim=1, keepdim=True))
            mask = torch.nn.functional.interpolate(
                mask, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False,
            )[0, 0]
            mask = mask.detach().float().cpu().numpy()
        finally:
            hook.remove()
            self.primary.zero_grad(set_to_none=True)

        mask -= mask.min()
        mask_max = float(mask.max())
        if mask_max > 1e-12:
            mask /= mask_max
        else:
            raise RuntimeError("Grad-CAM returned an empty activation map for this image.")

        # Color maps are explanatory overlays, NOT segmented/detected lesions.
        cmap = matplotlib.colormaps["turbo"]
        colored = (cmap(mask)[..., :3] * 255).astype(np.uint8)
        heatmap = Image.fromarray(colored, mode="RGB")
        overlay = Image.blend(padded, heatmap, alpha=0.37)
        return GradCAMImages(
            padded=padded, heatmap=heatmap, overlay=overlay,
            class_name=CLASS_NAMES[int(primary_index)],
        )
