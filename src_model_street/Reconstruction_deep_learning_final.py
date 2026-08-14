# -*- coding: utf-8 -*-
# Auto-converted from Reconstruction_deep_learning_final.ipynb
# Markdown cells are preserved as comments. Notebook magics/shell commands are commented.

# %% Cell 1
# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()

# %% Cell 2
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

ambifier_trained_models_path = kagglehub.dataset_download('ambifier/trained-models')
ambifier_resnet_18_mnist_path = kagglehub.dataset_download('ambifier/resnet-18-mnist')
ambifier_resnet_34_mnist_path = kagglehub.dataset_download('ambifier/resnet-34-mnist')
ambifier_resnet_152_mnist_final_path = kagglehub.dataset_download('ambifier/resnet-152-mnist-final')
ambifier_resnet_mnist_final_path = kagglehub.dataset_download('ambifier/resnet-mnist-final')
ambifier_resnet_18_cifar_path = kagglehub.dataset_download('ambifier/resnet-18-cifar')
ambifier_new_dataset_path = kagglehub.dataset_download('ambifier/new-dataset')

print('Data source import complete.')

# %% [markdown] Cell 3
# # Imports

# %% Cell 4
# Imports and helper fn
# === Block 0: Environment & helpers ===
# Paste this first. Installs are optional in Colab:
# !pip install torch torchvision matplotlib pillow tqdm

import torch, time, os, math
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import trange
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torch.nn.functional as F
import torchvision
from skimage.metrics import structural_similarity as ssim
import math
from torchvision.transforms.functional import to_pil_image
import os
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# %% Cell 5
# Parameters (papermill will overwrite these when running non-interactively)
variant = "resnet152"
num_classes = 10
check_point_path = "resnet152_cifar10.pth"
class_id = 9
use_imagenet_pretrained = False
map_location = "cpu"

# %% [markdown] Cell 6
# # General Resnet Architecture from scratch

# %% Cell 7
# === Block 1 (robust): Build ResNet and load either torchvision ImageNet weights OR a local checkpoint ===
# -------------------------
# Resent Architecture from scratch so no size and shape mismatches at checkpoints occur
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# -------------------------
# ResNet Core
# -------------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # CIFAR-style
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# %% [markdown] Cell 8
# # Loading User's Given given model weights

# %% Cell 9
# -------------------------
# ResNet Variants
# -------------------------
def build_resnet(variant="resnet50", num_classes=1000):
    if variant == "resnet18":
        return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)
    elif variant == "resnet34":
        return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
    elif variant == "resnet50":
        return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)
    elif variant == "resnet101":
        return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)
    elif variant == "resnet152":
        return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported ResNet variant: {variant}")

# -------------------------
# Safe checkpoint loader
# -------------------------
def load_weights_safely(model, checkpoint_path):
    # If it's a path, load it
    if isinstance(checkpoint_path, str):
        sd = torch.load(checkpoint_path, map_location="cpu")
    else:
        sd = checkpoint_path  # already a dict

    if isinstance(sd, dict):
        if 'state_dict' in sd: sd = sd['state_dict']
        elif 'model' in sd: sd = sd['model']
    sd = {k.replace("module.", ""): v for k,v in sd.items()}
    model_sd = model.state_dict()
    compatible = {k:v for k,v in sd.items() if k in model_sd and v.shape==model_sd[k].shape}
    skipped = [k for k in model_sd if k not in compatible]
    model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)}/{len(model_sd)} layers from checkpoint")
    if skipped:
        print("Skipped layers:", skipped)


# -------------------------
# Helper: extract model info dynamically
# -------------------------
def get_model_info(model):
    block_type = type(model.layer1[0]).__name__  # 'BasicBlock' or 'Bottleneck'
    layer_channels = [
        l[-1].conv3.out_channels if block_type=="Bottleneck" else l[-1].conv2.out_channels
        for l in [model.layer1, model.layer2, model.layer3, model.layer4]
    ]
    num_classes, embed_dim = model.fc.weight.shape
    return block_type, layer_channels, num_classes, embed_dim

# -------------------------
# === Build and initialize model ===
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
variant = "resnet152"   # choose: resnet18/34/50/101/152
num_classes = 10        # your dataset classes

# Build model
model = build_resnet(variant, num_classes=num_classes).to(device)

# For ImageNet pretrained model weights safely
use_imagenet_pretrained = False
if use_imagenet_pretrained:
    import torchvision.models as tv_models
    try:
        if variant=="resnet18": pretrained_model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        elif variant=="resnet34": pretrained_model = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)
        elif variant=="resnet50": pretrained_model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        elif variant=="resnet101": pretrained_model = tv_models.resnet101(weights=tv_models.ResNet101_Weights.IMAGENET1K_V2)
        elif variant=="resnet152": pretrained_model = tv_models.resnet152(weights=tv_models.ResNet152_Weights.IMAGENET1K_V1)

        # Directly pass the state dict to your safe loader
        load_weights_safely(model, checkpoint_path=pretrained_model.state_dict())

    except Exception as e:
        print("Failed to load ImageNet weights:", e)


# For models pretrained on other small datasets
check_point_path = "/kaggle/input/resnet-152-mnist-final/resnet152_mnist_final.pth"
load_weights_safely(model, check_point_path)

# Freeze all params
model.eval()
for p in model.parameters():
    p.requires_grad = False

# -------------------------
# Extract model info dynamically
# -------------------------
block_type, layer_channels, num_classes, embed_dim = get_model_info(model)

# Print summary
print("ResNet variant:", variant)
print("Block type:", block_type)
print("Layer output channels:", layer_channels)
print("Output FC shape:", model.fc.weight.shape)
print("num_classes:", num_classes, "embed_dim:", embed_dim)

# %% [markdown] Cell 10
# # Collecting BN - stats for the target model

# %% Cell 11
# === Block 3: Collect BatchNorm running stats (map module name -> running_mean/var) ===
bn_stats = {}
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm2d):
        bn_stats[name] = {
            'running_mean': module.running_mean.clone().to(device),
            'running_var': module.running_var.clone().to(device)
        }
print("BN modules found:", len(bn_stats))
# Example: list some BN module names (first 8)
print(list(bn_stats.keys())[:8])

# %% [markdown] Cell 12
# # Creating target embedding for optimization

# %% Cell 13
# === Block 4: Create initial target embedding from FC weights (prototype) ===
# fc.weight: shape (num_classes, embed_dim)
fc_w = model.fc.weight.detach().clone().to(device)  # (C, D)
fc_b = model.fc.bias.detach().clone().to(device) if model.fc.bias is not None else None

# Choose class_id(s) you want to reconstruct
class_id = 9 # change as needed, or loop over multiple classes
print("Using target class:", class_id)

# Simple initial target embedding: normalized classifier weight vector
z0 = fc_w[class_id].detach().clone()
z0 = z0 / (z0.norm() + 1e-8)  # unit vector
print("z0 norm:", z0.norm().item())

# %% [markdown] Cell 14
# # Initial Embedding optimization

# %% Cell 15
# === Block 5: Stage A - Embedding inversion (optimize z) ===
# === Stage A: Embedding Optimization (no image) ===
# We will optimize z (size embed_dim) to be a plausible penultimate embedding for class_id.
# Objective: maximize class logit (W_fc @ z + b), keep z near typical embedding norms, add smoothing regularizer.
# -------------------------
# Hooks to capture activations (optional)
# -------------------------
activations = {}
def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# attach hooks (avgpool for penultimate embedding)
try:
    model.avgpool.register_forward_hook(save_activation('avgpool'))
except Exception as e:
    print("Warning attaching hooks:", e)

# -------------------------
# Fallback embedding extractor (no hooks)
# -------------------------
def get_embedding_no_hooks(x):
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        emb = torch.flatten(x, 1)
    return emb

# -------------------------
# Estimate typical embedding norm
# -------------------------
def estimate_embed_norms(n=8):
    norms = []
    for _ in range(n):
        r = torch.randn(1, 3, 224, 224, device=device)
        _ = model(r)  # populate hooks if attached
        if 'avgpool' in activations:
            emb = activations['avgpool'].reshape(1, -1)
        else:
            emb = get_embedding_no_hooks(r)
        norms.append(float(emb.norm().item()))
    return float(np.mean(norms)), float(np.std(norms))

est_mean, est_std = estimate_embed_norms(n=6)
print("Estimated embedding norm mean,std:", est_mean, est_std)

# -------------------------
# Initial target embedding z0
# -------------------------
class_id = 0  # target class
fc_w = model.fc.weight.detach().clone().to(device)
fc_b = model.fc.bias.detach().clone().to(device) if model.fc.bias is not None else None

# Use class prototype from FC weights
z0 = fc_w[class_id].clone()
z0 = z0 / (z0.norm() + 1e-8)

# -------------------------
# Gradient-based optimization of z
# -------------------------
z = z0.clone().detach().requires_grad_(True).to(device)
optimizer_z = optim.Adam([z], lr=1e-2)
num_steps_z = 800
lambda_norm = 1e-1
lambda_l2z = 1e-4

for it in range(1, num_steps_z + 1):
    optimizer_z.zero_grad()

    # Compute class logit
    logits = fc_w @ z
    if fc_b is not None:
        logits += fc_b
    loss_logits = -logits[class_id]  # maximize target class logit

    # Regularize norm of embedding
    loss_norm = (z.norm() - est_mean)**2
    loss_l2 = (z**2).mean()

    # Total loss
    loss = loss_logits + lambda_norm * loss_norm + lambda_l2z * loss_l2
    loss.backward()  # compute gradient
    torch.nn.utils.clip_grad_norm_([z], 1.0)
    optimizer_z.step()  # gradient update

    # Renormalize every 200 iterations
    if it % 200 == 0:
        with torch.no_grad():
            z /= (z.norm() + 1e-8)

    if it % 200 == 0 or it == 1:
        with torch.no_grad():
            current_logit = (fc_w @ z + (fc_b if fc_b is not None else 0))[class_id].item()
            print(f"[Stage A] iter {it}/{num_steps_z}, loss={loss.item():.4f}, logit={current_logit:.4f}, norm={z.norm().item():.4f}")

# Final optimized embedding
z_opt = z.detach().clone()
print("Finished Stage A embedding optimization. z_opt norm:", z_opt.norm().item())

# %% [markdown] Cell 16
# # Creating Random Image to be further optimized for generating reconstruction

# %% Cell 17
# === Block 6: Stage B - initialize learnable image and helpers ===
# ============================
# === STAGE B — IMAGE OPTIM ===
# ============================
# better initialization (improves cosine similarity a LOT)
x = torch.randn(1, 3, 32, 32, device=device) * 0.01
x.requires_grad_(True)
optimizer_x = optim.Adam([x], lr=0.03)
mse = nn.MSELoss()

# ---------- helper: forward pass ----------
def forward_get_emb_logits(img):
    activations.clear()
    logits = model(img)
    emb = activations['avgpool'].reshape(img.shape[0], -1)
    return emb, logits

# %% [markdown] Cell 18
# # Main Image Reconstruction Using cosine similarity and other losses

# %% Cell 19
# === Stage B: Image Reconstruction Cosine Prior ===
# === Block 6: Stage B - initialize learnable image and helpers ===
# === Block 7: Optimize image to match z_opt ===
# ---------- priors ----------
def total_variation(img):
    return (
        (img[:,:,1:,:]-img[:,:,:-1,:]).pow(2).mean() +
        (img[:,:,:,1:]-img[:,:,:,:-1]).pow(2).mean()
    )

# ---------- show image ----------
def show_tensor_img(tensor, title=""):
    img = tensor.detach().cpu().clone()
    img = img.squeeze(0)        # (3,H,W)
    img = T.ToPILImage()(img)
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()


# ======== weights for losses ========
lambda_emb   = 1.0
lambda_cos   = 0.5         # ★ give cosine some weight (improves similarity)
lambda_tv    = 1e-5
lambda_l2img = 1e-4
lambda_bn    = 1e-2
lambda_logit = 0.1         # ★ improve logit → improves cosine

num_steps_x = 1200       # ★ run a bit more, cosine improves

# ================== optimization ==================
for it in range(1, num_steps_x+1):

    optimizer_x.zero_grad()

    emb, logits = forward_get_emb_logits(x)

    # MSE loss
    loss_emb = mse(emb, z_opt.unsqueeze(0))

    # cosine loss (very helpful)
    cos_sim = torch.nn.functional.cosine_similarity(emb, z_opt.unsqueeze(0))
    loss_cos = 1.0 - cos_sim.mean()

    # BN-stat matching (corrected)
    loss_bn = 0.0
    if lambda_bn > 0:
        if 'layer4' in activations:
            act = activations['layer4']             # (1,2048,H,W)
            bn_layer = model.layer4[2].bn3          # correct BN
            rm = bn_layer.running_mean.to(device)
            rv = bn_layer.running_var.to(device)

            act_mean = act.mean(dim=(0,2,3))
            act_var  = act.var(dim=(0,2,3), unbiased=False)

            loss_bn = mse(act_mean, rm) + mse(act_var, rv)

    # image priors
    loss_tv  = total_variation(x)
    loss_l2  = (x**2).mean()

    # logit boosting
    loss_logit = -logits[:, class_id].mean()

    # total loss
    loss = (
        lambda_emb * loss_emb +
        lambda_cos * loss_cos +
        lambda_tv  * loss_tv +
        lambda_l2img * loss_l2 +
        lambda_bn  * loss_bn +
        lambda_logit * loss_logit
    )

    loss.backward()
    optimizer_x.step()

    # keep pixels in range
    with torch.no_grad():
        x.clamp_(0, 1)

    # --- logging ---
    if it % 200 == 0 or it == 1:
        with torch.no_grad():
            emb_curr, logits_curr = forward_get_emb_logits(x)
            cos_now = torch.nn.functional.cosine_similarity(
                emb_curr, z_opt.unsqueeze(0)
            ).item()

            print(f"[Stage-B] {it}/{num_steps_x} "
                  f"loss={loss.item():.4f}  "
                  f"emb_MSE={loss_emb.item():.4f}  "
                  f"cos={cos_now:.4f}  "
                  f"logit={logits_curr[0,class_id].item():.4f}")

            show_tensor_img(x, title=f"iter {it} — cos {cos_now:.3f}")

# %% [markdown] Cell 20
# # Refinement Stage only for Bigger Resnets(50,101,152)

# %% Cell 21
# === Block 8: Refinement - multi-scale & multi-layer matching [needed for resnet50,101 and 152 only] ===
# --- Rebuild BN stats (required!) ---
bn_stats = {}
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm2d):
        bn_stats[name] = {
            "running_mean": module.running_mean.detach().clone(),
            "running_var":  module.running_var.detach().clone(),
        }

# --- BN lookup for layer3 and layer4 ---
bn_layer3 = model.layer3[2].bn3   # 1024-d
bn_layer4 = model.layer4[2].bn3   # 2048-d

mse = nn.MSELoss()

optimizer_x = optim.Adam([x], lr=1e-3)
num_refine = 800

lambda_tv = 5e-4
lambda_l2img = 5e-4
lambda_bn = 5e-2
lambda_layer = 1e-3
lambda_logit_boost = 0.02

def total_variation(img):
    b,c,h,w = img.shape
    tv = ((img[:,:,1:,:] - img[:,:,:h-1,:])**2).mean() + \
         ((img[:,:,:,1:] - img[:,:,:,:w-1])**2).mean()
    return tv

for it in range(1, num_refine+1):

    optimizer_x.zero_grad()
    emb, logits = forward_get_emb_logits(x)

    loss_emb = mse(emb, z_opt.unsqueeze(0))

    # ---- Multi-layer feature losses (L2 on activations) ----
    loss_layers = torch.tensor(0., device=device)
    loss_bn = torch.tensor(0., device=device)

    # Layer 3
    if "layer3" in activations:
        act = activations["layer3"]   # (1,1024,H,W)
        act_mean = act.mean(dim=(0,2,3))
        act_var  = act.var(dim=(0,2,3), unbiased=False)

        rm = bn_layer3.running_mean.to(device)
        rv = bn_layer3.running_var.to(device)

        loss_bn  += mse(act_mean, rm) + mse(act_var, rv)
        loss_layers += act.pow(2).mean()

    # Layer 4
    if "layer4" in activations:
        act = activations["layer4"]   # (1,2048,H,W)
        act_mean = act.mean(dim=(0,2,3))
        act_var  = act.var(dim=(0,2,3), unbiased=False)

        rm = bn_layer4.running_mean.to(device)
        rv = bn_layer4.running_var.to(device)

        loss_bn  += mse(act_mean, rm) + mse(act_var, rv)
        loss_layers += act.pow(2).mean()

    # ---- Natural image priors ----
    loss_tv = total_variation(x)
    loss_l2 = (x**2).mean()

    # ---- Logit boost ----
    loss_logit = -logits[:, class_id].mean()

    # ---- Total loss ----
    loss = (
        1.0 * loss_emb +
        lambda_layer * loss_layers +
        lambda_tv * loss_tv +
        lambda_l2img * loss_l2 +
        lambda_bn * loss_bn +
        lambda_logit_boost * loss_logit
    )

    loss.backward()
    optimizer_x.step()

    with torch.no_grad():
        x.clamp_(0,1)

    if it % 200 == 0 or it == 1:
        emb_curr, logits_curr = forward_get_emb_logits(x)
        cos_now = torch.nn.functional.cosine_similarity(
            emb_curr, z_opt.unsqueeze(0)
        ).item()

        print(f"[refine] it {it}/{num_refine} "
              f"loss={loss.item():.4f} "
              f"emb_mse={loss_emb.item():.4f} "
              f"cos={cos_now:.4f} "
              f"logit={logits_curr[0,class_id].item():.4f}")

        show_tensor_img(x, title=f"refine {it} cos {cos_now:.3f}")

# %% [markdown] Cell 22
# # Saving Final Reconstructed Image

# %% Cell 23
# === Block 9: Save final images, embeddings, and diagnostics ===
out_dir = "/kaggle/working/recon_output"
os.makedirs(out_dir, exist_ok=True)

# save image
final_img = (x.detach().cpu().clamp(0,1).squeeze(0).permute(1,2,0).numpy() * 255).astype(np.uint8)
Image.fromarray(final_img).save(os.path.join(out_dir, f"recon_class{class_id}.png"))
# save embedding
torch.save(z_opt.cpu(), os.path.join(out_dir, f"z_opt_class{class_id}.pt"))

# print final metrics
with torch.no_grad():
    emb_final, logits_final = forward_get_emb_logits(x)
    cos_final = torch.nn.functional.cosine_similarity(emb_final, z_opt.unsqueeze(0)).item()
    topk = torch.topk(torch.softmax(logits_final, dim=1), k=5)
    print("Final cosine with z_opt:", cos_final)
    print("Top-5 predicted classes:", topk.indices.cpu().numpy(), "probs:", topk.values.cpu().numpy())
print("Saved outputs to", out_dir)

# %% [markdown] Cell 24
# # Recsontructed vs Nearest Original using SSIM index ( CIFAR10 pretrained models)

# %% Cell 25
#Load CIFAR-10 original dataset ===
import torchvision
import torchvision.transforms as transforms

transform_raw = transforms.ToTensor()   # no normalization

trainset = torchvision.datasets.CIFAR10(
    root="/kaggle/working/cifar10_dataset",
    train=True,
    download=True,
    transform=transform_raw
)

print("Loaded CIFAR-10:", len(trainset), "images")

# %% Cell 26
# === Block 10: Nearest Image Search + Metrics (SSIM, PSNR, Cosine, MSE) ===
# ------------------------
# Helper: PSNR
# ------------------------
def compute_psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


# ------------------------
# 1. Prepare reconstruction
# ------------------------
device = next(model.parameters()).device      # model device (cpu or cuda)

recon = x.detach().clamp(0,1)                # tensor [1,3,H,W]
recon = recon.to(device)                      # to same device as model

# For SSIM (needs numpy CPU image)
recon_np = recon.detach().cpu().squeeze(0).permute(1,2,0).numpy()


# ------------------------
# 2. Search best match
# ------------------------
best_ssim = -1
best_idx = None
best_metrics = None

print("Searching nearest CIFAR-10 image...")

for idx in range(len(trainset)):
    img_raw, _ = trainset[idx]               # raw tensor [3,32,32] on CPU
    img_t = img_raw.unsqueeze(0).to(device)  # [1,3,32,32] to model device

    # SSIM computations must be on CPU numpy
    img_np = img_raw.permute(1,2,0).numpy()

    # ----- SSIM -----
    ssim_val = ssim(recon_np, img_np, channel_axis=2, data_range=1.0)

    if ssim_val > best_ssim:
        best_ssim = ssim_val
        best_idx = idx

        # Compute embedding-level cosine similarity
        with torch.no_grad():
            emb_r, _ = forward_get_emb_logits(recon)
            emb_o, _ = forward_get_emb_logits(img_t)
            cos_val = F.cosine_similarity(emb_r, emb_o).item()

        # Pixel-level metrics (MSE, PSNR)
        mse_val = F.mse_loss(recon, img_t).item()
        psnr_val = compute_psnr(recon, img_t)

        best_metrics = {
            "SSIM": float(best_ssim),
            "PSNR": float(psnr_val),
            "Cosine": float(cos_val),
            "MSE": float(mse_val),
        }

print("Nearest image index:", best_idx)
print("Best Metrics:", best_metrics)


# ------------------------
# 3. Visualization
# ------------------------
nearest_raw, _ = trainset[best_idx]    # CPU tensor [3,32,32]

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Reconstructed")
plt.imshow(to_pil_image(recon.detach().cpu().squeeze(0)))
plt.axis("off")

plt.subplot(1,2,2)
plt.title(f"Nearest CIFAR-10 (idx={best_idx})")
plt.imshow(to_pil_image(nearest_raw))
plt.axis("off")

plt.show()

# %% [markdown] Cell 27
# # Recsontructed vs Nearest Original using SSIM index ( Imagenet pretrained models)

# %% Cell 28
import os
# Kaggle automatically unzips into /kaggle/input/<dataset-name>
dataset_path = "/kaggle/input/new-dataset/train"   # already unzipped

# Standard ImageNet preprocessing
transform_imagenet_raw = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Load dataset directly
imagenet_train = torchvision.datasets.ImageFolder(
    root=dataset_path,      # no "train/train", just the actual folder
    transform=transform_imagenet_raw
)

print("Total images:", len(imagenet_train))
print("Classes:", imagenet_train.classes)


# %% Cell 29
def compute_psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


device = next(model.parameters()).device

# ----- Load reconstructed image -----
recon = x.detach().clamp(0,1).to(device)        # [1,3,H,W]

# Resize to 224×224 because ImageNet images are 224×224
recon_224 = F.interpolate(recon, size=(224,224), mode="bilinear", align_corners=False)
recon_np = recon_224.squeeze(0).permute(1,2,0).cpu().numpy()

best_ssim = -1
best_idx = None
best_metrics = None

print("Searching nearest ImageNet image...")

for idx in range(len(imagenet_train)):
    img_t, _ = imagenet_train[idx]
    img_t = img_t.unsqueeze(0).to(device)       # [1,3,224,224]

    img_np = img_t.squeeze(0).permute(1,2,0).cpu().numpy()

    # ===== SSIM =====
    ssim_val = ssim(recon_np, img_np, channel_axis=2, data_range=1.0)

    if ssim_val > best_ssim:
        best_ssim = ssim_val
        best_idx = idx

        with torch.no_grad():
            emb_r, _ = forward_get_emb_logits(recon_224)
            emb_o, _ = forward_get_emb_logits(img_t)
            cos_val = F.cosine_similarity(emb_r, emb_o).item()

        mse_val = F.mse_loss(recon_224, img_t).item()
        psnr_val = compute_psnr(recon_224, img_t)

        best_metrics = {
            "SSIM": ssim_val,
            "PSNR": psnr_val,
            "Cosine": cos_val,
            "MSE": mse_val,
        }

print("Nearest ImageNet index:", best_idx)
print("Metrics:", best_metrics)

nearest_raw, _ = imagenet_train[best_idx]

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Reconstructed")
plt.imshow(to_pil_image(recon_224.squeeze(0).cpu()))
plt.axis("off")

plt.subplot(1,2,2)
plt.title(f"Nearest IMAGENET idx={best_idx}")
plt.imshow(to_pil_image(nearest_raw))
plt.axis("off")
plt.show()

# %% [markdown] Cell 30
# # Recsontructed vs Nearest Original using SSIM index ( MNIST pretrained models)

# %% Cell 31
# Keep MNIST in original format (28x28, 1 channel)
transform_mnist_raw = transforms.Compose([
    transforms.ToTensor()
])

mnist_train = torchvision.datasets.MNIST(
    root="/kaggle/working/mnist_data",
    train=True,
    download=True,
    transform=transform_mnist_raw
)

print("MNIST loaded:", len(mnist_train), "images")

# %% Cell 32
# ------------------------------
# Load MNIST (28x28, 1-channel)
# ------------------------------
mnist_train = torchvision.datasets.MNIST(
    root="/kaggle/working/mnist_data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

print("MNIST loaded:", len(mnist_train))

# ------------------------------
# Helper metric
# ------------------------------
def compute_psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))

device = next(model.parameters()).device

# ------------------------------
# Your reconstruction (32x32x3)
# ------------------------------
recon = x.detach().clamp(0,1).to(device)   # [1,3,32,32]
recon_np = recon.squeeze(0).permute(1,2,0).cpu().numpy()  # (32,32,3)

best_ssim = -1
best_idx = None
best_metrics = None

print("Searching nearest MNIST neighbor...")

for idx in range(len(mnist_train)):
    img_t, _ = mnist_train[idx]   # [1,28,28]

    # ---------- Convert MNIST => (32,32,3) ----------
    img_28 = img_t.squeeze(0).cpu().numpy()  # (28,28)

    # Resize to 32x32
    img_32 = cv2.resize(img_28, (32,32), interpolation=cv2.INTER_LINEAR)   # (32,32)

    # Convert to 3 channels
    img_np = np.stack([img_32]*3, axis=2)   # (32,32,3)

    # ---------- SSIM ----------
    ssim_val = ssim(recon_np, img_np, channel_axis=2, data_range=1.0)

    if ssim_val > best_ssim:
        best_ssim = ssim_val
        best_idx = idx

        img_t32 = torch.tensor(img_np).permute(2,0,1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            emb_r, _ = forward_get_emb_logits(recon)
            emb_o, _ = forward_get_emb_logits(img_t32)
            cos_val = F.cosine_similarity(emb_r, emb_o).item()

        mse_val = F.mse_loss(recon, img_t32).item()
        psnr_val = compute_psnr(recon, img_t32)

        best_metrics = {
            "SSIM": ssim_val,
            "PSNR": psnr_val,
            "Cosine": cos_val,
            "MSE": mse_val
        }

print("\nNearest MNIST index:", best_idx)
print("Metrics:", best_metrics)

# ------------------------------
# Visualization
# ------------------------------
nearest_raw, _ = mnist_train[best_idx]
nearest_np = nearest_raw.squeeze(0).numpy()
nearest_np32 = cv2.resize(nearest_np, (32,32))
nearest_np32_3 = np.stack([nearest_np32]*3, axis=2)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Reconstructed")
plt.imshow(recon_np)
plt.axis("off")

plt.subplot(1,2,2)
plt.title(f"Nearest MNIST idx={best_idx}")
plt.imshow(nearest_np32_3, cmap="gray")
plt.axis("off")

plt.show()

# %% [markdown] Cell 33
# # Plots for model depth vs reconstruction quality and also across different dataaets

# %% Cell 34
import matplotlib.pyplot as plt

# =============================
# Data Provided
# =============================

depths = [18, 34, 50, 101, 152]

# CIFAR pretrained
cifar_ssim  = [0.0683, 0.1331, 0.07856, 0.09385, 0.08018]
cifar_psnr  = [7.2493, 7.2814, 6.1192, 6.4624, 6.5322]
cifar_cos   = [0.4213, 0.4102, 0.4053, 0.7613, 0.5074]
cifar_mse   = [0.18839, 0.18700, 0.24438, 0.22581, 0.22221]

# ImageNet pretrained
imagenet_ssim = [0.07622, 0.07697, 0.18359, 0.14606, 0.11332]
imagenet_psnr = [8.2373, 8.6348, 12.2082, 13.1758, 10.5478]
imagenet_cos  = [0.95927, 0.96604, 0.98354, 0.96739, 0.95196]
imagenet_mse  = [0.15006, 0.13693, 0.06014, 0.04813, 0.08814]

# MNIST pretrained
mnist_ssim = [0.03189, 0.09549, None, None, 0.12932]
mnist_psnr = [5.7608, 5.3962, None, None, 3.9432]
mnist_cos  = [0.99408, 0.90524, None, None, 0.99951]
mnist_mse  = [0.26540, 0.28865, None, None, 0.40334]

# Replace missing values with NaN for plotting
import numpy as np
mnist_ssim = [np.nan if v is None else v for v in mnist_ssim]
mnist_psnr = [np.nan if v is None else v for v in mnist_psnr]
mnist_cos  = [np.nan if v is None else v for v in mnist_cos]
mnist_mse  = [np.nan if v is None else v for v in mnist_mse]

# =============================
# Helper Function
# =============================
def plot_metrics(depths, ssim, psnr, cosine, mse, title):
    plt.figure(figsize=(8, 6))
    plt.plot(depths, ssim, marker='o', label='SSIM')
    plt.plot(depths, psnr, marker='o', label='PSNR')
    plt.plot(depths, cosine, marker='o', label='Cosine Similarity')
    plt.plot(depths, mse, marker='o', label='MSE')
    plt.title(title)
    plt.xlabel("ResNet Depth[18,34,50,101,152]")
    plt.ylabel("Metric Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================
# Plot 1: CIFAR pretrained
# =============================
plot_metrics(depths, cifar_ssim, cifar_psnr, cifar_cos, cifar_mse,
             "Reconstruction Metrics vs Depth (CIFAR Pretrained ResNets)")

# =============================
# Plot 2: ImageNet pretrained
# =============================
plot_metrics(depths, imagenet_ssim, imagenet_psnr, imagenet_cos, imagenet_mse,
             "Reconstruction Metrics vs Depth (ImageNet Pretrained ResNets)")

# =============================
# Plot 3: MNIST pretrained
# =============================
plot_metrics(depths, mnist_ssim, mnist_psnr, mnist_cos, mnist_mse,
             "Reconstruction Metrics vs Depth (MNIST Pretrained ResNets)")

