import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Simple color map for 5 classes
COLORS = np.array([
    [128, 64,128],   # road
    [244, 35,232],   # sidewalk
    [ 70, 70, 70],   # building
    [220, 20, 60],   # person
    [  0,  0,142],   # car
], dtype=np.uint8)

def decode_segmap(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls in range(len(COLORS)):
        color[mask == cls] = COLORS[cls]

    return color


def visualize_phase2(image, gt_mask, pred_mask, save_path):
    """
    image: [3,H,W] tensor or numpy
    gt_mask: [H,W]
    pred_mask: [H,W]
    """

    if not isinstance(image, np.ndarray):
        image = image.permute(1, 2, 0).cpu().numpy()

    gt_color = decode_segmap(gt_mask.cpu().numpy())
    pred_color = decode_segmap(pred_mask.cpu().numpy())

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image)
    axs[0].set_title("Image")
    axs[1].imshow(gt_color)
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred_color)
    axs[2].set_title("Prediction")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
