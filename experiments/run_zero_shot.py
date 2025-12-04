import yaml
from tqdm import tqdm
import numpy as np
import os
import sys
import argparse

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.logger import ExperimentLogger
from src.visualize import visualize_prediction
from src.dataset import CityscapesDataset, CITYSCAPES_IDS
from src.sam_predict import SAMZeroShot
from src.point_sampling import get_center_point
from src.metrics import iou

# Main script for Phase 1: Zero-Shot Segmentation on Cityscapes

def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM zero-shot on Cityscapes")

    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Cityscapes split to evaluate")

    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of images (for debugging)")

    parser.add_argument("--no-vis", action="store_true",
                        help="Disable visualization saving")

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = yaml.safe_load(open("configs/cityscapes.yaml"))

    # Override split
    cfg["image_dir"] = f"leftImg8bit/{args.split}"
    cfg["mask_dir"] = f"gtFine/{args.split}"

    save_visuals = cfg["save_visualizations"] and not args.no_vis

    # Initialize experiment logger (creates timestamped folder)
    logger = ExperimentLogger(cfg["output_dir"])

    # Load dataset
    ds = CityscapesDataset(
        cfg["data_root"],
        cfg["image_dir"],
        cfg["mask_dir"],
        cfg["classes"]
    )

    # Load SAM predictor
    predictor = SAMZeroShot(
        checkpoint=cfg["sam_checkpoint"],
        model_type=cfg["model_type"]
    )

    per_class = {c: [] for c in cfg["classes"]}

    # Limit dataset if requested
    total_images = len(ds) if args.limit is None else min(args.limit, len(ds))

    print(f"📊 Evaluating {total_images} images from split: {args.split}")

    # Main evaluation loop
    for idx in tqdm(range(total_images)):
        image, mask, img_path = ds[idx]

        for cls in cfg["classes"]:
            cid = CITYSCAPES_IDS[cls]
            class_mask = (mask == cid).astype(np.uint8)

            # Generate point prompt
            point = get_center_point(class_mask)
            if point is None:
                continue
            coords, labels = point

            # Run SAM zero-shot prediction
            pred_mask = predictor.predict_with_point(image, coords, labels)
            score = iou(pred_mask, class_mask)

            # Store IoU score
            if score is not None:
                per_class[cls].append(score)

                # Log detailed results
                logger.log_detail(
                    image_name=os.path.basename(img_path),
                    cls=cls,
                    iou=float(score),
                    point_x=int(coords[0][0]),
                    point_y=int(coords[0][1])
                )

            # Save visualization
            if save_visuals:
                vis_name = os.path.basename(img_path).replace(".png", f"_{cls}.png")
                save_path = os.path.join(logger.vis_dir, vis_name)
                visualize_prediction(
                    image, class_mask, pred_mask, cls, save_path
                )

    # After loop: compute and save summary
    mean_iou = np.mean([np.mean(per_class[c]) for c in per_class])

    logger.write_summary(
        per_class_iou={c: float(np.mean(per_class[c])) for c in per_class},
        mean_iou=float(mean_iou)
    )

    # Console output
    print("\n=== RESULTS ===")
    for cls in per_class:
        print(f"{cls:12s}: {np.mean(per_class[cls]):.4f}")

    print("\nMean IoU:", mean_iou)


if __name__ == "__main__":
    main()
