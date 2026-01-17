
import logging
import os
import random
import shutil
import sys
import traceback
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import datetime as _dt
import yaml

import cv2
import numpy as np
from PIL import Image
import torch

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- AUTO-FIX FOR MISSING SAM3 ASSETS ---
def fix_sam3_assets():
    try:
        import sam3
        sam3_path = Path(sam3.__file__).parent
        site_packages = sam3_path.parent
        assets_dir = site_packages / "assets"
        target_file = assets_dir / "bpe_simple_vocab_16e6.txt.gz"

        if not target_file.exists():
            assets_dir.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
            try:
                urllib.request.urlretrieve(url, str(target_file))
            except Exception:
                pass 
    except Exception:
        pass

fix_sam3_assets()

try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()  # type: ignore[attr-defined]
except Exception:
    pillow_heif = None

# SAM3 Import
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except Exception as e:
    logger.warning(f"SAM3 import failed: {e}")
    build_sam3_image_model = None
    Sam3Processor = None

YOLO_CLASS_ID = 0
HEIC_EXTS = {'.heic', '.heif'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', *HEIC_EXTS}

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    score: float
    mask: Optional[np.ndarray]

def auto_device(preferred: Optional[str] = None) -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def write_yolo(path: Path, detections: Sequence[Detection], img_shape: Tuple[int, int]):
    h, w = img_shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{YOLO_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def _load_image_rgb(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in HEIC_EXTS:
        if not pillow_heif:
            raise RuntimeError("HEIC/HEIF support missing.")
        try:
            img = Image.open(path)
            return np.array(img.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read HEIC image {path}: {exc}")
    img_bgr = cv2.imread(str(path))
    if img_bgr is not None:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        img = Image.open(path)
        return np.array(img.convert("RGB"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read image {path}: {exc}")

def split_items(items: List[Path], train_ratio: float = 0.7, val_ratio: float = 0.2) -> Dict[str, List[Path]]:
    random.shuffle(items)
    n = len(items)
    
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        # If only 1 image, use it for everything or just train
        return {"train": items, "val": items, "test": []}
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    # Ensure at least 1 val if we have multiple images
    if n_val == 0 and n > 1:
        n_val = 1
        n_train = n - n_val
        
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }

def save_dataset(records: Dict[Path, Dict[str, object]], output_root: Path, prompt_text: str) -> Path:
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_dir = output_root / f"dataset_{timestamp}"
    images_dir = ds_dir / "images"
    labels_dir = ds_dir / "labels"
    # previews_dir = ds_dir / "previews" # Optional: Skip previews for headless to save time/space
    
    for base in (images_dir, labels_dir):
        for split in ("train", "val", "test"):
            (base / split).mkdir(parents=True, exist_ok=True)
            
    all_paths = list(records.keys())
    splits = split_items(all_paths)
    
    for split_name, paths in splits.items():
        for src in paths:
            rec = records[src]
            dets: List[Detection] = rec["detections"]
            
            if src.suffix.lower() in HEIC_EXTS:
                img = _load_image_rgb(src)
                dest_name = f"{src.stem}.jpg"
                dest_img_path = images_dir / split_name / dest_name
                cv2.imwrite(str(dest_img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            else:
                dest_img_path = images_dir / split_name / src.name
                shutil.copy2(src, dest_img_path)
            
            label_stem = dest_img_path.stem
            img_shape = _load_image_rgb(src).shape
            write_yolo(labels_dir / split_name / f"{label_stem}.txt", dets, (img_shape[0], img_shape[1]))

    data_yaml = {
        "path": str(ds_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {YOLO_CLASS_ID: prompt_text},
    }
    with (ds_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f)
        
    return ds_dir

class HeadlessAnnotator:
    def __init__(self, device_str: str = None):
        self.device = auto_device(device_str)
        logger.info(f"Using Device: {self.device}")
        self.sam3 = None
        
        if build_sam3_image_model and Sam3Processor:
            logger.info("Initializing SAM 3...")
            try:
                m = build_sam3_image_model()
                m.to(self.device)
                self.sam3 = Sam3Processor(m)
                logger.info("SAM 3 Initialized successfully.")
            except Exception as e:
                logger.error(f"SAM 3 Initialization Error: {e}")
                logger.error(traceback.format_exc())
                self.sam3 = None
        else:
            logger.error("SAM3 libraries not found! This script requires SAM3.")

    def process_folder(self, folder_path: str, prompt_text: str, output_dir: str, limit: Optional[int] = None) -> Optional[Path]:
        if not self.sam3:
            logger.error("SAM3 not loaded. Cannot process.")
            return None

        folder = Path(folder_path)
        paths = sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
        
        if not paths:
            logger.warning(f"No images found in {folder_path}")
            return None

        if limit is not None and limit > 0:
            logger.info(f"Limiting to first {limit} images.")
            paths = paths[:limit]

        records = {}
        logger.info(f"Found {len(paths)} images. Processing with prompt: '{prompt_text}'")

        for idx, path in enumerate(paths, 1):
            logger.info(f"Processing {idx}/{len(paths)}: {path.name}")
            dets = self._process_image(path, prompt_text)
            if dets is not None:
                records[path] = {"detections": dets}
        
        if not records:
            logger.warning("No detections made.")
            return None

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        ds_path = save_dataset(records, output_root, prompt_text)
        logger.info(f"Dataset saved to {ds_path}")
        return ds_path

    def _process_image(self, path: Path, prompt_text: str) -> Optional[List[Detection]]:
        if path.suffix.lower() in HEIC_EXTS and pillow_heif is None:
            return None
        try:
            img = _load_image_rgb(path)
        except Exception as exc:
            logger.error(f"Failed to load: {exc}")
            return None
            
        dets_sam: List[Detection] = []
        
        try:
            pil_image = Image.fromarray(img)
            state = self.sam3.set_image(pil_image)
            out = self.sam3.set_text_prompt(state=state, prompt=prompt_text)
            
            masks = out.get("masks", [])
            scores = out.get("scores", [])
            
            if len(masks) > 0:
                for i in range(len(masks)):
                    raw_mask = masks[i]
                    raw_score = scores[i] if len(scores) > i else 1.0

                    if hasattr(raw_mask, "cpu"):
                          raw_mask = raw_mask.detach().cpu().numpy()
                    if hasattr(raw_score, "item"):
                          raw_score = raw_score.item()

                    mask = np.array(raw_mask).astype(np.uint8)
                    if mask.ndim > 2:
                        mask = mask.squeeze()

                    score = float(raw_score)
                    if score < 0.15: continue

                    ys, xs = np.nonzero(mask)
                    if len(xs) == 0: continue
                    bx1, by1, bx2, by2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                    
                    dets_sam.append(Detection(bbox=(bx1, by1, bx2, by2), score=score, mask=None)) # Don't need mask for YOLO
                
        except Exception as e:
            logger.error(f"SAM3 Execution Failed: {e}")
            logger.error(traceback.format_exc())
            return None

        return dets_sam
