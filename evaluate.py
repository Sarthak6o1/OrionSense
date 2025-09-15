# evaluate.py
import os, yaml, torch
from torch.utils.data import DataLoader
from datasets.cod10k_dataset import COD10KDetectionDataset, collate_fn
from models.faster_rcnn import get_fasterrcnn_model
from utils.engine import evaluate

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_subfolders(base_path):
    return {
        'image': os.path.join(base_path, 'Image'),
        'gt_instance': os.path.join(base_path, 'GT_Instance')
    }

test_folders = get_subfolders(os.path.join(cfg["dataset"]["root"], cfg["dataset"]["test"]))
test_dataset = COD10KDetectionDataset(test_folders)
test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False,
                         num_workers=cfg["training"]["num_workers"], collate_fn=collate_fn)

# Load model
model = get_fasterrcnn_model(cfg["model"]["num_classes"]).to(device)
checkpoint_path = os.path.join(cfg["output"]["checkpoint_dir"], cfg["output"]["save_name"])
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Evaluate
metrics = evaluate(model, test_loader, device, cfg={
    "score_threshold": 0.5,
    "seg_metric_mode": "gt_instance",
    "seg_approx_from_boxes": True
})

# Print results
print("\n=== Evaluation Results ===")
print("Detection metrics:", metrics["detection"])
if metrics.get("segmentation"):
    print("Segmentation metrics:", metrics["segmentation"])
