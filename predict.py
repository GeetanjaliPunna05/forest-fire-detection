"""Run the trained fire/smoke detector on an image, no notebook needed.

    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --out result.jpg --conf 0.6

Uses the final checkpoint (default: outputs/forest_fire_detection_best.pth, fetch it with `git lfs pull`).
The checkpoint stores the weights, model config and the tuned thresholds (confidence 0.82, NMS IoU 0.5).
The model code below is copied from the notebook (sections 3.4, 4.2 and 11) so both give identical results.
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

USE_AMP = torch.cuda.is_available()
COLORS = {"smoke": (0, 162, 255), "fire": (255, 59, 31)}       # RGB


def letterbox(img, canvas_hw):
    H, W = canvas_hw
    h, w = img.shape[:2]
    s = min(H / h, W / w)
    nh, nw = min(H, int(round(h * s))), min(W, int(round(w * s)))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.full((H, W, 3), 114, np.uint8)
    out[:nh, :nw] = img
    return out, s


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight, self.bias, self.eps, self.c = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps, c

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), (self.c,), self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class SimpleFeaturePyramid(nn.Module):
    def __init__(self, in_dim, out_dim, strides, patch):
        super().__init__()
        self.levels = nn.ModuleList()
        for s in strides:
            f = patch / s
            if f == 4:
                layers, d = [nn.ConvTranspose2d(in_dim, in_dim // 2, 2, 2), LayerNorm2d(in_dim // 2), nn.GELU(),
                             nn.ConvTranspose2d(in_dim // 2, in_dim // 4, 2, 2)], in_dim // 4
            elif f == 2:
                layers, d = [nn.ConvTranspose2d(in_dim, in_dim // 2, 2, 2)], in_dim // 2
            elif f == 1:
                layers, d = [], in_dim
            elif f == 0.5:
                layers, d = [nn.MaxPool2d(2, 2)], in_dim
            else:
                raise ValueError(f"unsupported stride {s} for patch size {patch}")
            layers += [nn.Conv2d(d, out_dim, 1, bias=False), LayerNorm2d(out_dim),
                       nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False), LayerNorm2d(out_dim)]
            self.levels.append(nn.Sequential(*layers))

    def forward(self, x):
        out = OrderedDict((str(i), lvl(x)) for i, lvl in enumerate(self.levels))
        out["pool"] = F.max_pool2d(list(out.values())[-1], 1, 2, 0)
        return out


class ViTDetBackbone(nn.Module):
    def __init__(self, vit_name, strides, pretrained=True, drop_path=0.1, out_channels=256):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0, global_pool="",
                                     dynamic_img_size=True, drop_path_rate=drop_path)
        cfg = self.vit.pretrained_cfg
        self.mean, self.std = list(cfg.get("mean", (0.5,) * 3)), list(cfg.get("std", (0.5,) * 3))
        self.patch = self.vit.patch_embed.patch_size[0]
        self.neck = SimpleFeaturePyramid(self.vit.embed_dim, out_channels, strides, self.patch)
        self.out_channels = out_channels

    def forward(self, x):
        B, _, H, W = x.shape
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=USE_AMP and x.is_cuda):
            tokens = self.vit.forward_features(x)[:, self.vit.num_prefix_tokens:]
            feats = self.neck(tokens.transpose(1, 2).reshape(B, -1, H // self.patch, W // self.patch))
        return OrderedDict((k, v.float()) for k, v in feats.items())


def build_detector(cfg, pretrained=False):
    backbone = ViTDetBackbone(cfg["vit"], cfg["strides"], pretrained, cfg["drop_path"])
    levels = list(cfg["strides"]) + [cfg["strides"][-1] * 2]
    sizes = tuple(tuple(int(round(k * s)) for k in cfg["anchor_k"]) for s in levels)
    anchors = AnchorGenerator(sizes, (tuple(cfg["anchor_ratios"]),) * len(levels))
    roi_pool = MultiScaleRoIAlign([str(i) for i in range(len(cfg["strides"]))], output_size=7, sampling_ratio=2)
    H, W = cfg["canvas"]
    return FasterRCNN(
        backbone, num_classes=3, min_size=H, max_size=W, image_mean=backbone.mean, image_std=backbone.std,
        rpn_anchor_generator=anchors, box_roi_pool=roi_pool,
        rpn_pre_nms_top_n_train=1000, rpn_post_nms_top_n_train=1000, rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_test=500,
        box_score_thresh=0.001, box_nms_thresh=0.5, box_detections_per_img=100, box_batch_size_per_image=256)


class FireSmokeDetector:
    """detector = FireSmokeDetector(path); detector(image) -> [{'label', 'confidence', 'box': [x1, y1, x2, y2]}, ...]"""

    def __init__(self, checkpoint, device=None, conf=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = build_detector(ck["cfg"], pretrained=False).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.model.roi_heads.nms_thresh = ck["nms_iou"]
        self.canvas, self.conf = ck["cfg"]["canvas"], (conf if conf is not None else ck["conf_threshold"])
        self.names = ck["class_names"]

    @torch.no_grad()
    def __call__(self, image, conf=None):
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise FileNotFoundError(f"could not read image: {image}")
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = np.asarray(image)
        h, w = image.shape[:2]
        img, scale = letterbox(image, self.canvas)
        x = torch.from_numpy(img).permute(2, 0, 1).float().div_(255).to(self.device)
        out = self.model([x])[0]
        thr = self.conf if conf is None else conf
        dets = []
        for b, s, l in zip(out["boxes"].float().cpu().numpy() / scale, out["scores"].float().cpu().numpy(), out["labels"].cpu().numpy()):
            if s >= thr:
                dets.append(dict(label=self.names[int(l)], confidence=round(float(s), 4),
                                 box=[round(float(v), 1) for v in (max(b[0], 0), max(b[1], 0), min(b[2], w), min(b[3], h))]))
        return sorted(dets, key=lambda d: -d["confidence"])


def draw(image_path, dets, out_path):
    img = cv2.imread(str(image_path))
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        r, g, b = COLORS.get(d["label"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), (b, g, r), 3)
        cv2.putText(img, f'{d["label"]} {d["confidence"]:.2f}', (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (b, g, r), 2)
    cv2.imwrite(str(out_path), img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Detect fire and smoke in an image.")
    ap.add_argument("image")
    ap.add_argument("--model", default="outputs/forest_fire_detection_best.pth")
    ap.add_argument("--conf", type=float, default=None, help="confidence threshold (default: tuned value stored in the model)")
    ap.add_argument("--out", default=None, help="save an image with the boxes drawn")
    a = ap.parse_args()
    if not Path(a.model).exists():
        raise SystemExit(f"model not found: {a.model}  (run `git lfs pull` to download it)")
    detections = FireSmokeDetector(a.model, conf=a.conf)(a.image)
    print(json.dumps(detections, indent=2) if detections else "no fire or smoke detected")
    if a.out:
        draw(a.image, detections, a.out)
        print("saved", a.out)
