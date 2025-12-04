import csv
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, output_root):
        # Create timestamped directory
        timestamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        self.exp_dir = os.path.join(output_root, timestamp)
        self.vis_dir = os.path.join(self.exp_dir, "visualizations")

        os.makedirs(self.vis_dir, exist_ok=True)

        # Path to detailed log
        self.detail_csv = os.path.join(self.exp_dir, "detailed_results.csv")
        self.summary_csv = os.path.join(self.exp_dir, "summary.csv")

        # Write headers
        with open(self.detail_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "class", "iou", "point_x", "point_y"])

    def log_detail(self, image_name, cls, iou, point_x, point_y):
        with open(self.detail_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([image_name, cls, iou, point_x, point_y])

    def write_summary(self, per_class_iou, mean_iou):
        with open(self.summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "mean_iou"])

            for cls in per_class_iou:
                writer.writerow([cls, per_class_iou[cls]])

            writer.writerow([])
            writer.writerow(["Overall mIoU", mean_iou])
