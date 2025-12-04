import yaml
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.dataset import CityscapesDataset, CITYSCAPES_IDS
from src.sam_predict import SAMZeroShot
from src.point_sampling import get_center_point
from src.metrics import iou
from src.visualize import visualize_prediction

def main():
    print("🔍 Running quick SAM sanity check...")

    cfg = yaml.safe_load(open("configs/cityscapes.yaml"))
    ds = CityscapesDataset(
        cfg["data_root"], cfg["image_dir"], cfg["mask_dir"], cfg["classes"]
    )

    predictor = SAMZeroShot(
        checkpoint=cfg["sam_checkpoint"],
        model_type=cfg["model_type"]
    )

    # Limit to a few samples for testing
    NUM_TEST = 3
    for i in range(min(NUM_TEST, len(ds))):
        image, mask, img_path = ds[i]
        print(f"\n🖼  Image: {img_path}")

        for cls in cfg["classes"]:
            cid = CITYSCAPES_IDS[cls]
            class_mask = (mask == cid).astype("uint8")

            point = get_center_point(class_mask)
            if point is None:
                print(f" - Skipping {cls} (no pixels found)")
                continue

            coords, labels = point
            pred_mask = predictor.predict_with_point(image, coords, labels)

            score = iou(pred_mask, class_mask)
            print(f" - {cls:10s} IoU: {score:.4f}")

            # Save visualization to experiments/test_vis
            out_dir = "experiments/test_vis"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"test_{i}_{cls}.png")
            visualize_prediction(image, class_mask, pred_mask, cls, out_path)
            print(f"   Saved visualization to {out_path}")

    print("\n✅ Quick test complete. If no errors appeared, you're ready for full run!")

if __name__ == "__main__":
    main()
