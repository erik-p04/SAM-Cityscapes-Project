import os
import cv2
import numpy as np

def overlay_mask(image, mask, color, alpha=0.5):
    """
    image: H x W x 3 RGB image
    mask: H x W boolean mask
    color: (r, g, b) tuple in [0,255]
    alpha: transparency
    """
    overlay = image.copy()
    overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlay


def visualize_prediction(
    image,
    gt_mask,
    pred_mask,
    class_name,
    save_path,
    show_gt_outline=True
):
    """
    Saves a side-by-side visualization:
    Left  = original image
    Middle = image + SAM mask
    Right = ground-truth class mask
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Choose fixed color for SAM predictions
    sam_color = (0, 255, 0)     # bright green
    gt_color  = (255, 0, 0)     # blue/red highlight

    # Overlay SAM prediction
    sam_vis = overlay_mask(image, pred_mask.astype(bool), sam_color, alpha=0.45)

    # Create GT visualization
    gt_vis = overlay_mask(image, gt_mask.astype(bool), gt_color, alpha=0.45)

    # Combine horizontally
    combined = np.concatenate([image, sam_vis, gt_vis], axis=1)

    # Add text labels
    h, w = image.shape[:2]
    cv2.putText(combined, "Original",        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(combined, f"SAM Prediction ({class_name})", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(combined, f"Ground Truth ({class_name})",   (2*w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    # Save image
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
