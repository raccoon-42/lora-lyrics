"""Render per-class classification report + confusion matrix to PNG files."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def save_clf_report(labels, preds, names, out_dir):
    """Save classification_report.png + confusion_matrix.png under `out_dir`.

    Returns the two paths.
    """
    out_dir = Path(out_dir)

    report = classification_report(labels, preds, target_names=names, digits=2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0, 1, report, family="monospace", fontsize=10, va="top")
    report_path = out_dir / "classification_report.png"
    fig.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return report_path, cm_path
