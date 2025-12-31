import os
import io
import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
from matplotlib import cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class BottleneckDec(nn.Module):
    def __init__(self, in_ch, out_ch, bottleneck_ratio=4):
        super().__init__()
        mid = max(out_ch // bottleneck_ratio, 8)

        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.act(out + identity)
        return out


class ResNet50DepthSwAV(nn.Module):
    def __init__(self, encoder_ckpt_path=None):
        super().__init__()

        backbone = models.resnet50(weights=None)

        if encoder_ckpt_path:
            checkpoint = torch.load(encoder_ckpt_path, map_location="cpu")
            state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("model.projection_head") or k.startswith("model.prototypes"):
                    continue
                if k.startswith("module.projection_head") or k.startswith("module.prototypes"):
                    continue
                if k.startswith("model."):
                    k = k[len("model."):]
                elif k.startswith("module."):
                    k = k[len("module."):]
                state_dict[k] = v
            backbone.load_state_dict(state_dict, strict=False)

        self.start = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.dec4 = BottleneckDec(2048, 512)
        self.dec3 = BottleneckDec(512 + 1024, 256)
        self.dec2 = BottleneckDec(256 + 512, 128)
        self.dec1 = BottleneckDec(128 + 256, 64)
        self.dec0 = BottleneckDec(64 + 64, 64, bottleneck_ratio=2)

        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        H, W = x.shape[-2:]

        x0 = self.start(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = self.dec4(x4)

        u3 = F.interpolate(d4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, x3], dim=1))

        u2 = F.interpolate(d3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))

        u1 = F.interpolate(d2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, x1], dim=1))

        d0 = self.dec0(torch.cat([d1, x0], dim=1))

        out = F.interpolate(d0, size=(H, W), mode="bilinear", align_corners=False)
        depth = self.head(out)
        return F.softplus(depth)


_IMAGE_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


def load_model(device, weights_path, encoder_ckpt_path=None):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Не найден файл весов: {weights_path}. "
            f"Положите ваши веса в ./artifacts/model.pt"
        )

    model = ResNet50DepthSwAV(encoder_ckpt_path).to(device)
    raw = torch.load(weights_path, map_location=device)

    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        raise ValueError("Неподдерживаемый формат model.pt")

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def infer_depth_png_and_npy(model, image_bytes, device):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _IMAGE_TF(img).unsqueeze(0).to(device)

    pred = model(x)[0, 0].detach().cpu().numpy().astype(np.float32)

    npy_buf = io.BytesIO()
    np.save(npy_buf, pred)
    depth_npy_bytes = npy_buf.getvalue()

    pmin, pmax = float(pred.min()), float(pred.max())
    den = (pmax - pmin) if (pmax - pmin) > 1e-6 else 1.0
    norm = (pred - pmin) / den

    rgba = (cm.get_cmap("magma")(norm) * 255).astype(np.uint8)[:, :, :3]
    out = Image.fromarray(rgba)

    png_buf = io.BytesIO()
    out.save(png_buf, format="PNG")
    depth_png_bytes = png_buf.getvalue()

    return depth_png_bytes, depth_npy_bytes

