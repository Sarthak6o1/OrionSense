# train.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
from datasets.cod10k_dataset import COD10KDetectionDataset, collate_fn
from models.faster_rcnn import get_fasterrcnn_model
from utils.engine import train_one_epoch, evaluate
import json

# -----------------------------
# Load configuration
# -----------------------------
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Prepare dataset folders
# -----------------------------
def get_subfolders(base_path):
    return {
        'image': os.path.join(base_path, 'Image'),
        'gt_instance': os.path.join(base_path, 'GT_Instance')
    }

train_folders = get_subfolders(os.path.join(cfg["dataset"]["root"], cfg["dataset"]["train"]))
test_folders  = get_subfolders(os.path.join(cfg["dataset"]["root"], cfg["dataset"]["test"]))

# -----------------------------
# Prepare datasets & dataloaders
# -----------------------------
train_dataset = COD10KDetectionDataset(train_folders, transforms=None)
test_dataset  = COD10KDetectionDataset(test_folders, transforms=None)

train_loader = DataLoader(train_dataset,
                          batch_size=cfg["training"]["batch_size"],
                          shuffle=True,
                          num_workers=cfg["training"]["num_workers"],
                          collate_fn=collate_fn)

test_loader = DataLoader(test_dataset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=cfg["training"]["num_workers"],
                         collate_fn=collate_fn)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# -----------------------------
# Initialize model, optimizer, scheduler
# -----------------------------
model = get_fasterrcnn_model(cfg["model"]["num_classes"]).to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params,
                            lr=cfg["training"]["lr"],
                            momentum=cfg["training"]["momentum"],
                            weight_decay=cfg["training"]["weight_decay"])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=cfg["training"]["step_size"],
                                            gamma=cfg["training"]["gamma"])

# -----------------------------
# Create output directories
# -----------------------------
os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
os.makedirs(cfg["output"]["results_dir"], exist_ok=True)

# -----------------------------
# Training loop
# -----------------------------
metrics_log = []

for epoch in range(cfg["training"]["epochs"]):
    print(f"\n===== Epoch {epoch+1}/{cfg['training']['epochs']} =====")

    # Train one epoch
    train_stats = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)

    # Evaluate
    eval_stats = evaluate(model, test_loader, device, cfg={
        "score_threshold": 0.5,
        "seg_metric_mode": "gt_instance",
        "seg_approx_from_boxes": True
    })

    # Step LR scheduler
    scheduler.step()

    # Merge stats
    stats = {"epoch": epoch+1, **train_stats, **eval_stats["detection"]}
    if eval_stats.get("segmentation"):
        stats.update(eval_stats["segmentation"])

    metrics_log.append(stats)

    # Save metrics to JSON
    metrics_file = os.path.join(cfg["output"]["results_dir"], f"metrics_epoch_{epoch+1:03d}.json")
    with open(metrics_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Epoch {epoch+1} metrics saved to {metrics_file}")

    # Save checkpoint
    checkpoint_file = os.path.join(cfg["output"]["checkpoint_dir"], f"epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_file)
    print(f"✅ Model checkpoint saved to {checkpoint_file}")
