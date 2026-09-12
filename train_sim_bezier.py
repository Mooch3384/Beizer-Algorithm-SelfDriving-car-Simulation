#!/usr/bin/env python3
"""
train_sim_bezier.py
Fast PyTorch Training Pipeline for BézierLaneNet on AvisEngine Track Data.

Trains CustomResnet to predict:
1. Number of detected lanes (Classification: fc1)
2. Parametric cubic Bézier control points (Regression: fc2)
Saves the optimized model weights to weights/bezier_avis_best.pth.
"""

import sys
import os
import json
import argparse
import time
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "BezierLaneNet"))
from models.custom_resnet import CustomResnet


class AvisBezierDataset(Dataset):
    def __init__(self, data_dir: str, input_w: int = 820, input_h: int = 295, max_lane: int = 4, degree: int = 3):
        self.data_dir = data_dir
        self.input_w = input_w
        self.input_h = input_h
        self.max_lane = max_lane
        self.degree = degree
        self.pts_per_lane = (degree + 1) * 2  # 4 control points * 2 (x,y) = 8

        labels_path = os.path.join(data_dir, "labels.json")
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Annotation file not found: {labels_path}")

        with open(labels_path, "r") as f:
            self.records = json.load(f)

        self.images_dir = os.path.join(data_dir, "images")
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(self.images_dir, rec["image_file"])
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        orig_h, orig_w = bgr.shape[:2]
        resized = cv2.resize(bgr, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_img = (rgb - self.mean) / self.std
        tensor_img = torch.from_numpy(norm_img.transpose(2, 0, 1)).float()

        lanes = rec.get("lanes", [])
        num_lanes = min(len(lanes), self.max_lane)

        # Control points target: shape (max_lane * (degree+1) * 2,)
        target_ctrl = np.zeros((self.max_lane, self.degree + 1, 2), dtype=np.float32)
        for i in range(num_lanes):
            pts = np.array(lanes[i]["control_points"], dtype=np.float32)
            target_ctrl[i, :len(pts)] = pts

        target_ctrl_flat = torch.from_numpy(target_ctrl.flatten()).float()
        target_cls = torch.tensor(num_lanes, dtype=torch.long)

        return tensor_img, target_cls, target_ctrl_flat


def train():
    parser = argparse.ArgumentParser(description="Train BézierLaneNet on AvisEngine data")
    parser.add_argument("--data_dir", type=str, default="dataset/avis_bezier", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_ckpt", type=str, default="weights/bezier_avis_best.pth", help="Checkpoint save path")
    args = parser.parse_args()

    print("=" * 65)
    print("  Training BézierLaneNet for AvisEngine  ")
    print("=" * 65)

    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Dataset directory not found: {args.data_dir}")
        print("Run collect_bezier_dataset.py first to generate training data from AvisEngine.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")

    dataset = AvisBezierDataset(args.data_dir)
    print(f"[INFO] Loaded {len(dataset)} samples from {args.data_dir}")

    should_drop_last = len(dataset) > args.batch_size
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=should_drop_last)

    max_lane = 4
    degree = 3
    num_fc_nodes = (degree + 1) * 2 * max_lane  # 32
    feat_dim = 384

    model = CustomResnet(feat_dim=feat_dim, ckpt='', max_lane=max_lane, num_fc_nodes=num_fc_nodes)
    model.to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.output_ckpt), exist_ok=True)
    best_loss = float('inf')

    print("[INFO] Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0

        t0 = time.time()
        for imgs, cls_targets, reg_targets in dataloader:
            imgs = imgs.to(device)
            cls_targets = cls_targets.to(device)
            reg_targets = reg_targets.to(device)

            optimizer.zero_grad()
            out_cls, out_reg = model(imgs)

            loss_cls = criterion_cls(out_cls, cls_targets)
            loss_reg = criterion_reg(out_reg, reg_targets)
            loss = loss_cls + 0.1 * loss_reg

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        dt = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Loss: {avg_loss:.4f} (Cls: {total_cls_loss/len(dataloader):.4f}, Reg: {total_reg_loss/len(dataloader):.4f}) "
              f"Time: {dt:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.output_ckpt)
            print(f"  --> Saved new best checkpoint: {args.output_ckpt} (Loss: {best_loss:.4f})")

    print("\n[DONE] BézierLaneNet training finished successfully!")
    print(f"[INFO] Trained checkpoint saved at: {args.output_ckpt}")


if __name__ == '__main__':
    train()
