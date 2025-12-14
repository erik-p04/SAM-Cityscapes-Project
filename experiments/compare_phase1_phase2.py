import csv
import matplotlib.pyplot as plt
import numpy as np

classes = ["road", "sidewalk", "building", "person", "car"]

def load_csv(path):
    data = {}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                data[row[0]] = float(row[1])
    return data

phase1 = load_csv("experiments/run_*/summary.csv")  # use latest manually
phase2 = load_csv("experiments/phase2/summary.csv")

p1 = [phase1[c] for c in classes]
p2 = [phase2[c] for c in classes]

x = np.arange(len(classes))
w = 0.35

plt.figure(figsize=(10,5))
plt.bar(x - w/2, p1, w, label="Phase 1 (Zero-shot)")
plt.bar(x + w/2, p2, w, label="Phase 2 (Trained Head)")

plt.xticks(x, classes)
plt.ylabel("IoU")
plt.title("Phase 1 vs Phase 2 IoU Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("experiments/phase1_vs_phase2.png")
plt.show()
