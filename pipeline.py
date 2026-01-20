import argparse
import sys
from pathlib import Path
import logging

# Ensure local modules are found
sys.path.append(str(Path(__file__).parent))

from annotator import HeadlessAnnotator
from trainer import Trainer

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="End-to-End Training Pipeline: Auto-Label -> Train")
    
    # Default paths are relative to this script
    base_dir = Path(__file__).parent
    default_imgs = base_dir / "FuelImgs"
    default_model = base_dir / "yolo11n.pt"
    
    # Fallback if yolo26n doesn't exist
    if not default_model.exists():
        default_model = "yolo11n.pt"

    parser.add_argument("--input-images", type=str, default=str(default_imgs), help="Path to folder containing images")
    parser.add_argument("--prompt", type=str, default="yellow ball", help="Text prompt for SAM3 auto-labeling")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of images to process")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--model", type=str, default=str(default_model), help="YOLO model to start from")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_images)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return

    # 1. Annotation
    logger.info("Step 1: Auto-Annotation")
    logger.info(f"Input Images: {input_path}")
    logger.info(f"Prompt: {args.prompt}")
    if args.limit:
        logger.info(f"Limit: {args.limit} images")
    
    annotator = HeadlessAnnotator(device_str=args.device)
    
    # Output dataset to a 'datasets' folder within TrainingPipeline
    output_base = base_dir / "datasets"
    
    dataset_path = annotator.process_folder(
        folder_path=str(input_path),
        prompt_text=args.prompt,
        output_dir=str(output_base),
        limit=args.limit
    )
    
    if not dataset_path:
        logger.error("Annotation failed or no images found. Aborting.")
        return

    logger.info(f"Dataset generated at: {dataset_path}")
    data_yaml = dataset_path / "data.yaml"
    
    if not data_yaml.exists():
        logger.error(f"data.yaml missing in {dataset_path}")
        return

    # 2. Training
    logger.info("Step 2: Training")
    logger.info(f"Starting Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    
    trainer = Trainer(base_model=args.model, epochs=args.epochs)
    
    export_root = base_dir / "exports"
    runs_root = base_dir / "runs"
    trainer.train(
        data_yaml_path=str(data_yaml),
        export_root=str(export_root),
        runs_root=str(runs_root)
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()