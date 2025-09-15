import os
import yaml
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import gradio as gr
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of classes including background
num_classes = cfg["model"]["num_classes"]

# Load a model pre-trained on COCO
model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)

# Replace the classifier with a new one for our number of classes (if not using pretrained weights)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model = model.to(device)

# Load checkpoint if available
checkpoint_dir = cfg["output"]["checkpoint_dir"]
checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")],
                     key=lambda x: int(x.split("_")[1].split(".")[0]))  # expects checkpoint_XX.pth style

if checkpoints:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

model.eval()

def predict(img, score_threshold=0.5):
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)[0]

    keep = outputs["scores"] > score_threshold
    boxes, scores = outputs["boxes"][keep].cpu(), outputs["scores"][keep].cpu()

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    ax = plt.gca()

    for box, score in zip(boxes, scores):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f"{score:.2f}", color="yellow", fontsize=12)

    plt.axis("off")
    output_path = "prediction_result.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path

iface = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Slider(minimum=0, maximum=1, value=0.5, step=0.05, label="Score Threshold")],
    outputs=gr.Image(type="filepath"),
    title="Custom Faster R-CNN Object Detection",
    description="Upload an image to detect objects using your custom-trained Faster R-CNN."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
