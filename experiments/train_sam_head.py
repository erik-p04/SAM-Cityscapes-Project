import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import get_device, set_seed
from src.sam_backbone import SAMBackbone
from src.seg_head import SegmentationHead
from src.cityscapes_semantic import CityscapesSemanticDataset
from src.metrics import compute_iou
from src.visualize import visualize_phase2

# -----------------------------
# Phase 2: Train Semantic Head
# -----------------------------

def main():
    cfg = yaml.safe_load(open("configs/cityscapes.yaml"))

    set_seed(42)
    device = get_device()

    # Dataset & loader
    train_ds = CityscapesSemanticDataset(
        cfg["data_root"],
        "leftImg8bit/train",
        "gtFine/train",
        cfg["classes"],
    )

    val_ds = CityscapesSemanticDataset(
        cfg["data_root"],
        "leftImg8bit/val",
        "gtFine/val",
        cfg["classes"],
    )

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4)

    # Models
    backbone = SAMBackbone(cfg["sam_checkpoint"], cfg["model_type"]).to(device)
    head = SegmentationHead(in_channels=256, num_classes=len(cfg["classes"]))
    head.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    os.makedirs("experiments/phase2", exist_ok=True)

    for epoch in range(5):
        head.train()
        total_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/5"):
            images = images.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                feats = backbone(images)

            logits = head(feats)
            logits = nn.functional.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

        # Validation
        head.eval()
        ious = []

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(device)
                targets = targets.to(device)

                feats = backbone(images)
                logits = head(feats)
                logits = nn.functional.interpolate(
                    logits,
                    size=targets.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                preds = logits.argmax(dim=1)
                iou = compute_iou(preds, targets, num_classes=len(cfg["classes"]))
                ious.append(iou)

                # -----------------------------------------
                # Phase 2 qualitative visualization (FINAL EPOCH ONLY)
                # -----------------------------------------
                if epoch == 4 and batch_idx < 5:
                    os.makedirs("experiments/phase2/visualizations", exist_ok=True)
                    save_path = f"experiments/phase2/visualizations/sample_{batch_idx}.png"

                    visualize_phase2(
                        images[0],
                        targets[0],
                        preds[0],
                        save_path
                    )


        mean_iou = sum(ious) / len(ious)
        print(f"Epoch {epoch+1}: Val mIoU = {mean_iou:.4f}")

        # -----------------------------------------
        # Save Phase 2 summary CSV (final epoch only)
        # -----------------------------------------
        if epoch == 4:
            import csv
            with open("experiments/phase2/summary.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerow(["mIoU", mean_iou])
        # -----------------------------------------


        # Save checkpoint
        torch.save(
            head.state_dict(),
            f"experiments/phase2/seg_head_epoch_{epoch+1}.pth",
        )


if __name__ == "__main__":
    main()
