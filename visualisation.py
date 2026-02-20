import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
import pandas as pd
import seaborn as sns

# Output dir
out_dir = Path("./visuals")
out_dir.mkdir(exist_ok=True)

# Load data
P = Path("./data/patches")
X_te = np.load(P / "X_test.npy")
y_te = np.load(P / "y_test.npy")

# Load model
model = load_model("cvlstm_blanks.h5", compile=False)

# Predict
y_pred = model.predict(X_te)
y_pred_bin = (y_pred > 0.1).astype(np.uint8)

# Store confusion matrix stats
confusion_stats = []

def plot_sample(idx):
    ndvi  = X_te[idx][-1, ..., 0]
    lst   = X_te[idx][-1, ..., 1]
    wind  = X_te[idx][-1, ..., 2]
    gt    = y_te[idx, ..., 0]
    pred  = y_pred_bin[idx, ..., 0]

    # Confusion metrics
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    confusion_stats.append((tp, fp, fn, tn))

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    axs[0, 0].imshow(ndvi, cmap='Greens', vmin=0, vmax=1); axs[0, 0].set_title("NDVI")
    axs[0, 1].imshow(lst, cmap='coolwarm', vmin=0, vmax=1); axs[0, 1].set_title("LST")
    axs[0, 2].imshow(wind, cmap='Blues', vmin=0, vmax=1); axs[0, 2].set_title("Wind")

    axs[1, 0].imshow(gt, cmap='gray'); axs[1, 0].set_title("GT Fire Mask")
    axs[1, 1].imshow(pred, cmap='Oranges'); axs[1, 1].set_title("Predicted Fire Mask")
    axs[1, 2].imshow(ndvi, cmap='Greens', alpha=0.8)
    axs[1, 2].imshow(pred, cmap='autumn', alpha=0.4)
    axs[1, 2].set_title("Overlay: NDVI + Prediction")

    for ax in axs.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_dir / f"sample_{idx}.png", dpi=200)
    plt.close()

fire_indices = [i for i in range(len(y_te)) if y_te[i].sum() > 0]
for i in fire_indices:
    plot_sample(i)

# Confusion matrix summary
tp_total = sum(tp for tp, _, _, _ in confusion_stats)
fp_total = sum(fp for _, fp, _, _ in confusion_stats)
fn_total = sum(fn for _, _, fn, _ in confusion_stats)
tn_total = sum(tn for _, _, _, tn in confusion_stats)

print("🔥 Confusion Matrix (Total of 5 samples):")
print(f"True Positives:  {tp_total}")
print(f"False Positives: {fp_total}")
print(f"False Negatives: {fn_total}")
print(f"True Negatives:  {tn_total}")

# confusion_stats = [(sample_idx, TP, FP, FN, TN)] from previous run
confusion_stats = []

# Rebuild confusion matrix from predictions
for idx in range(len(y_te)):
    gt = y_te[idx, ..., 0]
    pred = (y_pred[idx, ..., 0] > 0.1).astype(np.uint8)

    if gt.sum() == 0:  # Skip if no fire in GT
        continue

    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()

    confusion_stats.append((idx, tp, fp, fn, tn))

# Build DataFrame
summary = []
for i, (idx, tp, fp, fn, tn) in enumerate(confusion_stats):
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) else 0
    summary.append({
        "Sample #": idx,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "IoU": round(iou, 4),
    })

df_summary = pd.DataFrame(summary)
df_summary.to_csv("confusion_matrix_summary.csv", index=False)

# ────────────────────────────────────────────────────────────────
#  REPLACE EVERYTHING *AFTER* YOU PRINT THE TOTALS WITH THIS
# ────────────────────────────────────────────────────────────────
print("\n🔥 Confusion-matrix totals (fire samples only):")
print(f"TP:{tp_total}  FP:{fp_total}  FN:{fn_total}  TN:{tn_total}")

# ---- 1.   PLOT *COUNTS* CONFUSION MATRIX -----------------------
cm_counts = np.array([[tp_total, fn_total],
                      [fp_total, tn_total]])

labels = ["Fire", "No Fire"]
plt.figure(figsize=(6, 5))
sns.heatmap(cm_counts, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=labels, yticklabels=labels, cbar=False)
plt.title("Confusion Matrix – Counts")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(out_dir / "confusion_matrix_counts.png", dpi=300)
plt.close()

# ---- 2.   PLOT *NORMALISED (%)* CONFUSION MATRIX --------------
cm_percent = cm_counts / cm_counts.sum(axis=1, keepdims=True) * 100
plt.figure(figsize=(6, 5))
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="YlGnBu",
            xticklabels=labels, yticklabels=labels, cbar=False)
plt.title("Confusion Matrix – % of True Class")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(out_dir / "confusion_matrix_percent.png", dpi=300)
plt.close()

print(f"\n✅  Saved heat-maps to  {out_dir.resolve()}")