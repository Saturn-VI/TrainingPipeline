from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION
import torch
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import shutil
import logging
import sys
import os
import traceback

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------- 
# CONFIG DEFAULTS
# ------------------------------------------------- 
DEFAULT_BASE_MODEL = "yolo11n.pt" 
IMG_SIZE = 640
BATCH = 16
PATIENCE = 100
WORKERS = mp.cpu_count()
DEVICE = 0

def version_tuple(version: str) -> tuple[int, int, int]:
    parts = []
    for chunk in version.split("."):
        num = ""
        for ch in chunk:
            if ch.isdigit():
                num += ch
            else:
                break
        if num:
            parts.append(int(num))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

def require_ultralytics(min_version: str, reason: str) -> None: 
    if version_tuple(ULTRALYTICS_VERSION) < version_tuple(min_version):
        logger.warning(
            f"{reason}. Installed ultralytics {ULTRALYTICS_VERSION}. "
            "Run: pip install -U ultralytics"
        )

class Trainer:
    def __init__(self, base_model: str = DEFAULT_BASE_MODEL, epochs: int = 20):
        self.base_model = base_model
        self.epochs = epochs
        
        # Check GPU
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        else:
            logger.warning("CUDA not available. Training will be slow.")

    def train(self, data_yaml_path: str, export_root: str):
        data_yaml = Path(data_yaml_path)
        if not data_yaml.exists():
            logger.error(f"Data YAML not found: {data_yaml}")
            return

        export_root_path = Path(export_root)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"pipeline_run_{timestamp}"
        
        # Setup run-specific output directories
        export_root_path.mkdir(exist_ok=True)
        archive_dir = export_root_path / run_name
        archive_dir.mkdir(parents=True, exist_ok=True)
        latest_dir = export_root_path / "latest"
        
        # Load Model
        logger.info(f"Loading model: {self.base_model}")
        model = YOLO(self.base_model)

        logger.info(f"Starting training for {self.epochs} epochs...")
        
        # Train
        try:
            results = model.train(
                data=str(data_yaml),
                epochs=self.epochs,
                imgsz=IMG_SIZE,
                batch=BATCH,
                device=DEVICE if torch.cuda.is_available() else "cpu",
                workers=WORKERS,
                optimizer="AdamW",
                lr0=0.0015,
                lrf=0.01,
                cos_lr=True,
                patience=PATIENCE,
                cache="disk",
                # Augmentations
                hsv_h=0.02, hsv_s=0.6, hsv_v=0.4,
                degrees=3.0, translate=0.07, scale=0.4,
                fliplr=0.5, mosaic=0.4, close_mosaic=10,
                mixup=0.1, copy_paste=0.1, erasing=0.2,
                name=run_name,
                exist_ok=False,
                deterministic=False
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())
            return

        # Paths
        run_dir = Path(results.save_dir)
        weights_dir = run_dir / "weights"
        best_pt = weights_dir / "best.pt"

        # Copy plots to run-specific folder
        plots_dir = archive_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for plot_file in ["results.png", "confusion_matrix.png", "train_batch0.jpg", "train_batch1.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg"]:
            src_plot = run_dir / plot_file
            if src_plot.exists():
                shutil.copy(src_plot, plots_dir / plot_file)
        logger.info(f"Training plots saved to: {plots_dir}")

        if not best_pt.exists():
            logger.error("best.pt not found. Training failed.")
            return

        logger.info("Exporting FP16 TFLite...")
        # Suppress TF warnings for export
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
        try:
            # Force YOLO export
            export_model = YOLO(best_pt)
            export_model.export(format="tflite", imgsz=IMG_SIZE, half=True)
        except Exception as e:
            logger.error(f"Export failed: {e}")
            logger.error(traceback.format_exc())
            # Continue to save .pt at least

        tflite_files = sorted(weights_dir.rglob("*.tflite"), key=lambda p: str(p))
        best_tflite = tflite_files[0] if tflite_files else None

        # Archive best.pt
        shutil.copy(best_pt, archive_dir / "best.pt")
        
        if best_tflite:
            shutil.copy(best_tflite, archive_dir / "model.tflite")

        # Update Latest
        if latest_dir.exists():
            try:
                shutil.rmtree(latest_dir)
            except Exception as e:
                logger.warning(f"Could not clear latest dir: {e}")
        latest_dir.mkdir(exist_ok=True)

        shutil.copy(best_pt, latest_dir / "best.pt")
        if best_tflite:
            shutil.copy(best_tflite, latest_dir / "model.tflite")
            
        # Copy plots to latest
        latest_plots = latest_dir / "plots"
        if plots_dir.exists():
             shutil.copytree(plots_dir, latest_plots, dirs_exist_ok=True)

        # Labels
        names = model.names
        labels_txt_content = ""
        for i in sorted(names.keys()):
            labels_txt_content += f"{names[i]}\n"
        
        labels_txt = latest_dir / "labels.txt"
        labels_txt.write_text(labels_txt_content, encoding="utf-8")
        shutil.copy(labels_txt, archive_dir / "labels.txt")

        logger.info("TRAINING COMPLETE")
        logger.info(f"Results saved to: {archive_dir}")
        logger.info(f"Latest pointing to: {latest_dir}")

if __name__ == "__main__":
    mp.freeze_support()
    # Test stub
    pass