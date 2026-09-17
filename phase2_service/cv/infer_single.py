#Single-image inference and quantification engine for CellScope Phase 2.
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import cv2
from PIL import Image

try:
    import torch
    from phase2_service.cv.unet import UNet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    UNet = None

from phase2_service.cv.postprocess import NucleiPostProcessor

_MODEL_CACHE: Dict[str, Any] = {}


def resolve_model_path(model_path: Optional[str] = None) -> Optional[str]:
    candidates = [
        model_path,
        os.environ.get("MODEL_PATH"),
        str(REPO_ROOT / "phase1_cv" / "cellscope_phase1" / "unet_nuclei_best.pt"),
        str(REPO_ROOT / "phase1_cv" / "cellscope_phase1_deliverables" / "unet_nuclei_best.pt"),
        str(REPO_ROOT / "phase2_service" / "models" / "unet_nuclei_best.pt"),
        str(REPO_ROOT / "models" / "unet_nuclei_best.pt"),
        "/app/models/unet_nuclei_best.pt",
        "/models/unet_nuclei_best.pt"
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return str(Path(c).resolve())
    return None


def load_model(model_path: Optional[str] = None, device: str = "cpu"):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to load U-Net weights.")

    resolved_path = resolve_model_path(model_path)
    if not resolved_path:
        raise FileNotFoundError("Model checkpoint unet_nuclei_best.pt not found.")

    cache_key = f"{resolved_path}_{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    device_obj = torch.device(device)
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    checkpoint = torch.load(resolved_path, map_location=device_obj)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model_keys = set(model.state_dict().keys())
    adapted_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k[7:] if k.startswith("module.") else k
        if clean_k == "outc.weight" and "outc.conv.weight" in model_keys:
            adapted_state_dict["outc.conv.weight"] = v
        elif clean_k == "outc.bias" and "outc.conv.bias" in model_keys:
            adapted_state_dict["outc.conv.bias"] = v
        elif clean_k == "outc.conv.weight" and "outc.weight" in model_keys:
            adapted_state_dict["outc.weight"] = v
        elif clean_k == "outc.conv.bias" and "outc.bias" in model_keys:
            adapted_state_dict["outc.bias"] = v
        else:
            adapted_state_dict[clean_k] = v

    model.load_state_dict(adapted_state_dict, strict=False)
    model.to(device_obj)
    model.eval()

    _MODEL_CACHE[cache_key] = model
    return model


def run_inference(
    image: Union[str, Path, np.ndarray, Image.Image],
    model: Optional[Any] = None,
    model_path: Optional[str] = None,
    device: str = "cpu",
    threshold: float = 0.5,
    method: str = "watershed",
    pixel_scale_um: float = 0.25
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any], np.ndarray]:
    if isinstance(image, (str, Path)):
        p = str(image)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found at {p}")
        bgr = cv2.imread(p)
        if bgr is None:
            raise ValueError(f"Could not decode image at {p}")
        rgb_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        rgb_orig = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        rgb_orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 and image.shape[2] == 3 else image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    orig_h, orig_w = rgb_orig.shape[:2]
    resized = cv2.resize(rgb_orig, (256, 256), interpolation=cv2.INTER_LINEAR)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    norm = (resized.astype(np.float32) / 255.0 - mean) / std
    tensor_input = np.expand_dims(np.transpose(norm, (2, 0, 1)), axis=0).astype(np.float32)

    if model is None and TORCH_AVAILABLE:
        try:
            model = load_model(model_path, device=device)
        except Exception:
            model = None

    if model is not None and TORCH_AVAILABLE:
        device_obj = torch.device(device)
        with torch.no_grad():
            x = torch.from_numpy(tensor_input).to(device_obj)
            prob_map_256 = model(x).squeeze().cpu().numpy()
    else:
        gray = cv2.cvtColor(rgb_orig, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        prob_map_256 = cv2.resize(thresh.astype(np.float32) / 255.0, (256, 256))

    prob_map = cv2.resize(prob_map_256, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    postprocessor = NucleiPostProcessor(
        prob_threshold=threshold,
        min_distance_factor=0.35,
        min_nucleus_area=15,
        pixel_scale_um=pixel_scale_um,
        method=method
    )

    instance_mask = postprocessor.separate_instances(prob_map, rgb_image=rgb_orig)
    features_df, summary = postprocessor.extract_features(instance_mask)
    overlay_image = postprocessor.render_overlay(rgb_orig, instance_mask)

    nuclei_features = features_df.to_dict(orient="records") if not features_df.empty else []
    return instance_mask, nuclei_features, summary, overlay_image

