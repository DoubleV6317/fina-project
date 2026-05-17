import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from config import IMAGENET_MEAN, IMAGENET_STD, IMG_CROP


class DenseNet121(nn.Module):
    """ImageNet-pretrained DenseNet121 with a sigmoid multi-label classifier head."""

    def __init__(self, out_size, pretrained=True):
        super().__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=pretrained)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass returning per-class probabilities."""
        return self.densenet121(x)


class HeatmapGenerator:
    """Class Activation Map (CAM) generator for visualising DenseNet121 attention."""

    def __init__(self, model, device=None):
        self.model = model
        self.model.eval()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = list(self.model.densenet121.classifier[0].parameters())[0].data
        self.transform = transforms.Compose([
            transforms.Resize((IMG_CROP, IMG_CROP)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def generate(self, image, class_idx, original_size=None):
        """Build a CAM heatmap overlaid onto the original image for one class."""
        from PIL import Image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        if original_size is None:
            original_size = image.size
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        feature_maps = []

        def hook_fn(module, _input, output):
            feature_maps.append(output.detach())

        handle = self.model.densenet121.features.register_forward_hook(hook_fn)
        with torch.no_grad():
            self.model(tensor)
        handle.remove()

        feat = feature_maps[0].squeeze(0)
        cam = torch.zeros(feat.shape[1:], device=self.device)
        for i, w in enumerate(self.weights[class_idx]):
            cam += w * feat[i]
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, original_size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        orig = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)
        return overlay, heatmap
