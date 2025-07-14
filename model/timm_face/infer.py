import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import safetensors
import sklearn.preprocessing
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from safetensors.torch import load_file


parser = argparse.ArgumentParser()
parser.add_argument("--ijb_dir", default="ijb")
parser.add_argument("--meta_dir", default="meta")
parser.add_argument("--dataset", required=True, choices=["IJBB", "IJBC"])
parser.add_argument("--model", required=True)
parser.add_argument("--model_kwargs", type=json.loads)
parser.add_argument("--ckpt", default="weights/ReID/face_ReID_vit/model.safetensors")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--channels_last", action="store_true")
parser.add_argument("--compile", action="store_true")
parser.add_argument("--amp_dtype", default="none", choices=["none", "float16", "bfloat16"])
args = parser.parse_args()

if __name__ == '__main__':

    img = Image.fromarray(np.uint8(np.random.rand(112, 112, 3) * 255))
    img = img.convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5  # Normalize to [-1, 1]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 112, 112)

    
    model = timm.create_model(
        "local-dir:weights/ReID/face_ReID_vit",
        pretrained=False
    ).eval().cuda()

    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    if args.compile:
        model.compile()

    if args.ckpt is not None:
        if args.ckpt.endswith(".safetensors"):
            with safetensors.safe_open(args.ckpt, framework="pt") as f:
                state_dict = {k: f.get_tensor(k) for k in f.keys()}
        else:
            state_dict = torch.load(args.ckpt, map_location="cpu")

    # Forward pass
    with torch.no_grad():
        embs = model(img_tensor.cuda())
    embs = F.normalize(embs.float(), dim=1).cpu().numpy()

    # Check output shape
    assert embs.shape == (1, 512)
