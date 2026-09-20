"""Confusion matrix for the trained fire/smoke detector on the test set.

Re-uses the notebook's own code cells (data loading, model, prediction), loads the saved
checkpoint, predicts on the test set and builds a detection confusion matrix.

Matching rule (same idea as the YOLO confusion matrix): detections at or above the confidence
threshold are matched to ground-truth boxes one-to-one by IoU >= 0.5, highest confidence first,
regardless of class.
  matched pair          -> cell (true class, predicted class)
  unmatched GT box      -> cell (true class, background)   = missed object
  unmatched detection   -> cell (background, predicted class) = false alarm
Needs, in the repository root (the folder above this script): forest_fire_vit_detection.ipynb, the model
outputs/forest_fire_detection_best.pth (run `git lfs pull`), and the D-Fire data in data/ (or set DATA_ROOT).
It can be started from any folder: it changes into the repository root itself.
On Kaggle: add the dataset sayedgamal99/smoke-fire-detection-yolo as an input; the script finds it under /kaggle/input
by itself (or set DATA_ROOT). The model is always read from outputs/, not from /kaggle/working.
"""
import json, os, sys, time
from pathlib import Path
T0 = time.time()
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # repository root
os.chdir(ROOT)                                          # the notebook code uses paths relative to the root
NOTEBOOK, MODEL_FILE = ROOT / "forest_fire_vit_detection.ipynb", ROOT / "outputs" / "forest_fire_detection_best.pth"
if not NOTEBOOK.exists():
    sys.exit(f"Missing {NOTEBOOK.name} in {ROOT}. Clone the whole repository (this script re-uses code from the notebook).")
if not MODEL_FILE.exists() or MODEL_FILE.stat().st_size < 10_000_000:
    sys.exit(f"Model missing or not downloaded: {MODEL_FILE}. Run `git lfs pull` (a ZIP download only contains a small pointer file).")


def find_data_root(search_roots=("/kaggle/input",), max_depth=5):
    """Folder that contains test/images: DATA_ROOT if set, else ./data, else a bounded search (Kaggle inputs)."""
    def ok(d):
        return (Path(d) / "test" / "images").is_dir()
    if os.environ.get("DATA_ROOT"):
        return os.environ["DATA_ROOT"]
    if ok(ROOT / "data"):
        return str(ROOT / "data")
    for base in search_roots:
        if not os.path.isdir(base):
            continue
        base_depth = base.rstrip("/\\").count(os.sep)
        for cur, dirs, _ in os.walk(base):
            if cur.count(os.sep) - base_depth >= max_depth:
                dirs[:] = []
            if ok(cur):
                return cur
    return str(ROOT / "data")


os.environ["DATA_ROOT"] = find_data_root()
if not (Path(os.environ["DATA_ROOT"]) / "test" / "images").is_dir():
    sys.exit(f"Test images not found under DATA_ROOT={os.environ['DATA_ROOT']}. Download D-Fire and set DATA_ROOT to the folder "
             "that contains train/, val/ and test/.")
nb = json.load(open(NOTEBOOK, encoding="utf-8"))
g = globals()
g["display"] = print                                 # notebook cells call IPython's display()
for i in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20):          # setup, config, data, model and metric definitions
    print(f"[{time.time()-T0:.0f}s] running cell {i}", flush=True)
    if nb["cells"][i]["cell_type"] == "code":
        exec(compile("".join(nb["cells"][i]["source"]), f"cell{i}", "exec"), g)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.ops import box_iou

CKPT_PATH = MODEL_FILE          # the notebook code points this at /kaggle/working on Kaggle; always use the repo's model
CKPT = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
MODEL = build_detector(CKPT["cfg"], pretrained=False).to(DEVICE)
MODEL.load_state_dict(CKPT["model"])
CONF = float(CKPT["conf_threshold"]) if "conf_threshold" in CKPT else 0.82
NMS = float(CKPT.get("nms_iou", 0.5))
print(f"checkpoint epoch {CKPT['epoch']} | conf {CONF} | nms {NMS}")

preds, metrics = evaluate(MODEL, TEST_RECS, CKPT["cfg"]["canvas"], nms_thr=NMS)
print(f"test mAP@0.50 = {100 * metrics['mAP50']:.2f}")

names = ["smoke", "fire", "background"]
cm = np.zeros((3, 3), dtype=int)                    # rows: true, columns: predicted
for i, rec in enumerate(TEST_RECS):
    b, s, l = preds[i]
    keep = s >= CONF
    b, s, l = b[keep], s[keep], l[keep]
    gt_b, gt_l = rec["boxes"], rec["labels"]
    used_gt = np.zeros(len(gt_b), bool)
    matched_det = np.zeros(len(b), bool)
    if len(b) and len(gt_b):
        iou = box_iou(torch.as_tensor(b), torch.as_tensor(gt_b)).numpy()
        for d in np.argsort(-s):
            cand = np.where(~used_gt, iou[d], -1)
            j = int(np.argmax(cand))
            if cand[j] >= 0.5:
                used_gt[j], matched_det[d] = True, True
                cm[int(gt_l[j]) - 1, int(l[d]) - 1] += 1
    for j in np.where(~used_gt)[0]:
        cm[int(gt_l[j]) - 1, 2] += 1                # missed
    for d in np.where(~matched_det)[0]:
        cm[2, int(l[d]) - 1] += 1                   # false alarm

os.makedirs("deliverables", exist_ok=True)
df = pd.DataFrame(cm, index=[f"true {n}" for n in names], columns=[f"pred {n}" for n in names])
df.to_csv("deliverables/confusion_matrix.csv")
print(df.to_string())

row_sum = cm.sum(1, keepdims=True)
pct = np.divide(100 * cm, row_sum, out=np.zeros(cm.shape), where=row_sum > 0)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, data, title, fmt in ((axes[0], cm, "Counts", "{:d}"), (axes[1], pct, "Row-normalised (% of true class)", "{:.1f}")):
    show = pct if ax is axes[1] else np.where(np.eye(3, dtype=bool), cm, cm)
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(3)); ax.set_xticklabels(names); ax.set_yticks(range(3)); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for r in range(3):
        for c in range(3):
            if r == 2 and c == 2:
                continue                              # background/background is undefined for detection
            ax.text(c, r, fmt.format(data[r, c]), ha="center", va="center",
                    color="white" if pct[r, c] > 55 else "black", fontsize=12)
fig.suptitle(f"Test-set confusion matrix (IoU 0.5, confidence {CONF}, {len(TEST_RECS)} images)")
plt.tight_layout()
plt.savefig("deliverables/confusion_matrix.png", dpi=150, bbox_inches="tight")
print("saved deliverables/confusion_matrix.png and confusion_matrix.csv", flush=True)
if sys.platform == "win32":                     # ROCm on Windows can spin at 100% CPU while the GPU runtime shuts down
    import ctypes
    ctypes.windll.kernel32.TerminateProcess(ctypes.windll.kernel32.GetCurrentProcess(), 0)
os._exit(0)
