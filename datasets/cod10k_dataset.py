import os, numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class COD10KDetectionDataset(Dataset):
    def __init__(self, folders, transforms=None):
        self.image_dir = folders['image']
        self.instance_dir = folders['gt_instance']
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def _get_boxes_from_instance_mask(self, inst_mask_np):
        boxes, labels, areas, iscrowd = [], [], [], []
        unique_ids = np.unique(inst_mask_np)
        unique_ids = unique_ids[unique_ids != 0]  # ignore background
        for uid in unique_ids:
            ys, xs = np.where(inst_mask_np == uid)
            if ys.size == 0: continue
            xmin, xmax, ymin, ymax = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
            if xmax <= xmin or ymax <= ymin: continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  
            areas.append((xmax - xmin) * (ymax - ymin))
            iscrowd.append(0)
        if len(boxes) == 0:
            return (torch.zeros((0,4),dtype=torch.float32),
                    torch.zeros((0,),dtype=torch.int64),
                    torch.zeros((0,),dtype=torch.float32),
                    torch.zeros((0,),dtype=torch.int64))
        return (torch.as_tensor(boxes,dtype=torch.float32),
                torch.as_tensor(labels,dtype=torch.int64),
                torch.as_tensor(areas,dtype=torch.float32),
                torch.as_tensor(iscrowd,dtype=torch.int64))

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        inst_path = os.path.join(self.instance_dir, img_name.replace('.jpg','.png').replace('.jpeg','.png'))

        img = Image.open(img_path).convert("RGB")
        img = TF.to_tensor(img)

        if os.path.exists(inst_path):
            inst = Image.open(inst_path)
            inst_np = np.array(inst)[...,0] if np.array(inst).ndim == 3 else np.array(inst)
        else:
            inst_np = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)

        boxes, labels, areas, iscrowd = self._get_boxes_from_instance_mask(inst_np)
        target = {
        "boxes": boxes,
        "labels": labels,
        "image_id": torch.tensor([idx]),
        "area": areas,
        "iscrowd": iscrowd,
        "inst_path": inst_path  # path to GT instance mask
}
        if self.transforms:
            img = self.transforms(img)
        return img, target

def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]
