import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_fasterrcnn_model(num_classes=2, pretrained_backbone=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, pretrained_backbone=pretrained_backbone)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
