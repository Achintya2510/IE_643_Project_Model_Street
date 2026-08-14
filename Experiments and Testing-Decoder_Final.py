# -*- coding: utf-8 -*-
# Auto-converted from Experiments and Testing-Decoder_Final.ipynb
# Markdown cells are preserved as comments. Notebook shell commands are commented.

# %% Cell 1
#imports
import zipfile
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
import torchvision.utils as vutils
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import tarfile

zip_path = "/content/train.zip"  # path to your zip
extract_to = "/content/imagenet_images" # folder to extract images

# Make sure folder exists
os.makedirs(extract_to, exist_ok=True)

# Unzip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"Unzipped images to {extract_to}")

# %% Cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Cell 3
#Decoder definition
# ----------------- Decoder with skip connections -----------------
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Decoder body
        self.up1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(64, 3, 3, padding=1)
        self.relu = nn.ReLU()

        # Deep adapters: small vs large
        self.deep_adapter_small = nn.Conv2d(512, 512, 1)   # for ResNet18/34
        self.deep_adapter_large = nn.Conv2d(2048, 512, 1)  # for ResNet50/101/152

        # Skip adapters: small vs large
        self.skip_adapters_small = nn.ModuleList([
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(128, 128, 1),
            nn.Conv2d(64, 64, 1)
        ])
        self.skip_adapters_large = nn.ModuleList([
            nn.Conv2d(1024, 256, 1),
            nn.Conv2d(512, 128, 1),
            nn.Conv2d(256, 64, 1)
        ])

    def forward(self, x, skips):
     """
     x: final output from encoder [B, C, H, W]
     skips: list of skip features [layer3, layer2, layer1], can be None
     Automatically chooses adapter based on input channels.
     """
     # Determine if small or large ResNet by channel count
     if x.shape[1] == 512:
        # Small ResNet (18/34)
        deep_adapter = self.deep_adapter_small
        skip_adapter_list = self.skip_adapters_small
     elif x.shape[1] == 2048:
        # Large ResNet (50/101/152)
        deep_adapter = self.deep_adapter_large
        skip_adapter_list = self.skip_adapters_large
     else:
        raise ValueError(f"Unexpected embedding channels: {x.shape[1]}")

     # Apply deep adapter
     x = self.relu(deep_adapter(x))

     # Upsample + skip connections
     x = self.up1(x)
     if skips[0] is not None:
        x = x + skip_adapter_list[0](skips[0])

     x = self.up2(x)
     if skips[1] is not None:
        x = x + skip_adapter_list[1](skips[1])

     x = self.up3(x)
     if skips[2] is not None:
        x = x + skip_adapter_list[2](skips[2])

     # Final convolution to RGB
     x = torch.sigmoid(self.final_conv(x))
     return x

decoder = Decoder().to(device)

# %% Cell 4
# ----------------- Helper: encoder forward with skip connections -----------------
def encoder_forward(x):
    skips = []
    x = encoder_backbone.conv1(x)
    x = encoder_backbone.bn1(x)
    x = encoder_backbone.relu(x)
    x = encoder_backbone.maxpool(x)

    x = encoder_backbone.layer1(x)
    skips.append(x)   # layer1
    x = encoder_backbone.layer2(x)
    skips.append(x)   # layer2
    x = encoder_backbone.layer3(x)
    skips.append(x)   # layer3
    x = encoder_backbone.layer4(x)
    return x, skips[::-1]  # reverse: layer3, layer2, layer1

# %% Cell 5
#Loading your already trained decoder
decoder = Decoder().to(device)  # or 'cpu' depending on setup

# Path to your saved weights
weights_path = "/content/decoder_trained.pth"

# Load the state dict
state_dict = torch.load(weights_path, map_location='cuda')

# Handle different checkpoint structures
if isinstance(state_dict, dict) and 'state_dict' in state_dict:
    decoder.load_state_dict(state_dict['state_dict'])
elif isinstance(state_dict, dict) and 'decoder' in state_dict:
    decoder.load_state_dict(state_dict['decoder'])
else:
    decoder.load_state_dict(state_dict)

# Set to evaluation mode
decoder.eval()

# %% [markdown] Cell 6
# #Testing on resnet 18 which was pre trained image net

# %% Cell 7
# ------------------- Imports -------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os

# ------------------- User Settings -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = "deepinv_outputs"
os.makedirs(out_dir, exist_ok=True)

# Decoder must be already defined and trained
# from previous code: decoder = ...

large = False  # True -> ResNet50/101/152, False -> ResNet18/34
B = 1        # batch size

# Embedding & skip shapes
if large:
    z_shape = (B, 2048, 7, 7)
    skip_shapes = [(B, 1024, 14, 14), (B, 512, 28, 28), (B, 256, 56, 56)]
else:
    z_shape = (B, 512, 7, 7)
    skip_shapes = [(B, 256, 14, 14), (B, 128, 28, 28), (B, 64, 56, 56)]

# Optimization params
num_steps = 300
save_every = 50
lr = 0.1
l2_weight = 1.0
tv_weight = 1e-4
bn_weight = 1.0
z_l2_weight = 1e-6
optimize_skips = False
use_image_denorm = False
n_images = 5  # Number of reconstructions

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

# ------------------- Dataset Settings -------------------
dataset_root = "/content/imagenet_images"   # ImageFolder-style
dataset_batch = 64
feature_cache_path = os.path.join(out_dir, "dataset_features.pt")
compare_distance = "cosine"
save_pairs_dir = os.path.join(out_dir, "recon_vs_nearest")
os.makedirs(save_pairs_dir, exist_ok=True)

preprocess_for_encoder = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
display_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
dataset_for_features = datasets.ImageFolder(dataset_root, transform=preprocess_for_encoder)
dataset_for_display  = datasets.ImageFolder(dataset_root, transform=display_transform)
dataset_loader = DataLoader(dataset_for_features, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)
display_loader = DataLoader(dataset_for_display, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)

# ------------------- Build Frozen Encoder for BN Loss -------------------
if large:
    enc_model_full = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
else:
    enc_model_full = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone_modules = list(enc_model_full.children())[:-2]  # remove avgpool + fc
encoder_for_bn = nn.Sequential(*backbone_modules).to(device).eval()
for p in encoder_for_bn.parameters():
    p.requires_grad = False

print("Using encoder_for_bn with final channels =", z_shape[1])

# ------------------- Helper Functions -------------------
def denorm_if_needed(img_tensor):
    t = img_tensor
    if use_image_denorm:
        t = (t * IMAGENET_STD) + IMAGENET_MEAN
    return t.clamp(0,1)

def save_and_show(tensor_img, path, show=True):
    t = denorm_if_needed(tensor_img.detach().cpu())
    vutils.save_image(t, path)
    if show:
        B = t.shape[0]
        for i in range(B):
            np_img = t[i].permute(1,2,0).numpy()
            plt.figure(figsize=(4,4))
            plt.imshow(np_img)
            plt.axis('off')
            plt.show()

def tv_loss(img):
    dy = torch.mean(torch.abs(img[:,:,1:,:] - img[:,:,:-1,:]))
    dx = torch.mean(torch.abs(img[:,:,:,1:] - img[:,:,:,:-1]))
    return dx + dy

def bn_stat_loss_from_encoder(decoder_out, encoder_model, device):
    bn_features = []
    hooks = []

    def hook_fn(module, inp, out):
        bn_features.append(out)

    for module in encoder_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(hook_fn))

    _ = encoder_model(decoder_out)

    losses = []
    bn_modules = [m for m in encoder_model.modules() if isinstance(m, nn.BatchNorm2d)]
    for out_bn, module in zip(bn_features, bn_modules):
        mean_batch = out_bn.mean(dim=[0,2,3])
        var_batch  = out_bn.var(dim=[0,2,3], unbiased=False)
        losses.append(((mean_batch - module.running_mean.to(device))**2).mean())
        losses.append(((var_batch  - module.running_var.to(device))**2).mean())

    for h in hooks:
        h.remove()

    return sum(losses)

# ------------------- Extract / Cache Dataset Features -------------------
def extract_and_cache_dataset_features(encoder, loader, cache_path, device):
    if os.path.exists(cache_path):
        print("Loading cached dataset features:", cache_path)
        cached = torch.load(cache_path, map_location='cpu')
        features = cached['features']
        paths    = cached['paths']
        return features, paths

    encoder.eval()
    all_feats, all_paths = [], []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            feats = encoder(x)
            feats_gap = feats.mean(dim=[2,3])
            all_feats.append(feats_gap.cpu())
            start_idx = batch_idx * loader.batch_size
            end_idx = start_idx + feats_gap.shape[0]
            for i in range(start_idx, min(end_idx, len(dataset_for_features.samples))):
                all_paths.append(dataset_for_features.samples[i][0])
    features = torch.cat(all_feats, dim=0)
    torch.save({'features': features, 'paths': all_paths}, cache_path)
    print("Cached dataset features to:", cache_path)
    return features, all_paths

dataset_features, dataset_paths = extract_and_cache_dataset_features(encoder_for_bn, dataset_loader, feature_cache_path, device)
dataset_features = dataset_features.to(device)
print("Dataset size:", dataset_features.shape[0], "feature-dim:", dataset_features.shape[1])

# ------------------- Projection Layer if Needed -------------------
with torch.no_grad():
    sample_input = torch.randn(1,3,224,224, device=device)
    sample_feat  = encoder_for_bn(sample_input).mean(dim=[2,3])
feat_dim = sample_feat.shape[1]
dataset_feat_dim = dataset_features.shape[1]

if dataset_feat_dim != feat_dim:
    print(f"Projecting dataset features from {dataset_feat_dim} -> {feat_dim}")
    proj_layer = nn.Linear(dataset_feat_dim, feat_dim, bias=False).to(device)
    dataset_features_proj = proj_layer(dataset_features)
else:
    proj_layer = None
    dataset_features_proj = dataset_features

# ------------------- Helper: get feature vector for reconstructed image -------------------
# ---------- Projection Layer if Needed (create dataset -> encoder projection only) ----------
with torch.no_grad():
    sample_input = torch.randn(1,3,224,224, device=device)
    sample_feat  = encoder_for_bn(sample_input).mean(dim=[2,3])
feat_dim = sample_feat.shape[1]            # encoder output dim (e.g. 2048)
dataset_feat_dim = dataset_features.shape[1]  # cached features dim (e.g. 512)

proj_layer = None
dataset_features_proj = dataset_features

if dataset_feat_dim != feat_dim:
    print(f"Projecting dataset features from {dataset_feat_dim} -> {feat_dim}")
    # projection maps dataset_dim -> encoder_dim
    proj_layer = nn.Linear(dataset_feat_dim, feat_dim, bias=False).to(device)
    # apply projection only to the dataset features (N, dataset_feat_dim) -> (N, feat_dim)
    with torch.no_grad():
        dataset_features_proj = proj_layer(dataset_features)  # (N, feat_dim)
else:
    print("No projection needed: dataset features and encoder features match.")
def get_feature_for_img_tensor(img_tensor, encoder, device, proj_layer=None, dataset_feat_dim=None):
    """
    img_tensor: (B,3,H,W) in [0,1]
    Returns: feats (B, feat_dim) matching encoder output dim.
    If proj_layer is provided, will APPLY IT ONLY if recon feats have dimension == proj_layer.in_features.
    """
    encoder.eval()
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    imgs_resized = torch.zeros((img_tensor.shape[0],3,224,224), device=device)
    for i in range(img_tensor.shape[0]):
        pil = transforms.ToPILImage()(img_tensor[i].cpu())
        proc = preprocess(pil)
        t = transforms.ToTensor()(proc).to(device)
        t = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(t)
        imgs_resized[i] = t

    with torch.no_grad():
        feats = encoder(imgs_resized).mean(dim=[2,3])  # (B, feat_dim_encoder)

        # If proj_layer exists, only use it *if* feats currently match dataset_feat_dim (i.e. need projection).
        # But typically we want: dataset_features_proj has shape (N, feat_dim_encoder), so we should NOT project recon feats.
        # We therefore apply proj_layer only when the recon feats' dim equals proj_layer.in_features.
        if proj_layer is not None:
            # infer in_features from proj_layer.weight shape
            in_features = proj_layer.weight.shape[1]
            if feats.shape[1] == in_features:
                feats = proj_layer(feats)
            else:
                # feats already in encoder dimension — do NOT project
                pass

    return feats

# ------------------- Distance function -------------------
def compute_nearest_dataset_idx(query_feats, dataset_feats, metric="euclidean"):
    if metric=="euclidean":
        dists = torch.cdist(query_feats, dataset_feats)
        idx = torch.argmin(dists, dim=1)
        return idx, dists
    elif metric=="cosine":
        qn = torch.nn.functional.normalize(query_feats, dim=1)
        dn = torch.nn.functional.normalize(dataset_feats, dim=1)
        sims = qn @ dn.t()
        idx = torch.argmax(sims, dim=1)
        return idx, sims
    else:
        raise ValueError("Unsupported metric")

# ------------------- Main Reconstruction Loop -------------------
global_saved = 0
for img_idx in range(n_images):
    print(f"\n--- Generating reconstruction {img_idx+1}/{n_images} ---")

    # Initialize embeddings
    z = torch.randn(z_shape, device=device, requires_grad=True)
    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]
    optimizer_emb = optim.Adam(opt_params, lr=lr)

    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])
        loss = l2_weight*torch.mean(out**2) + tv_weight*tv_loss(out) + z_l2_weight*torch.mean(z**2)
        if bn_weight>0:
            loss += bn_weight*bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

        if step % 50==0 or step==1:
            print(f"step {step}/{num_steps}  loss={loss.item():.6f}")
            save_path = os.path.join(out_dir, f"deepinv_step{step:04d}.png")
            save_and_show(out, save_path, show=False)

    # Final reconstruction
    recon = denorm_if_needed(out.detach().cpu())
    recon_for_enc = recon.to(device)
    recon_feats = get_feature_for_img_tensor(recon_for_enc, encoder_for_bn, device, proj_layer)
    idxs, dists = compute_nearest_dataset_idx(recon_feats, dataset_features_proj, metric=compare_distance)

    for b in range(recon.shape[0]):
        ds_idx = idxs[b].item()
        ds_path = dataset_paths[ds_idx]
        img_ds = Image.open(ds_path).convert('RGB')
        img_ds = display_transform(img_ds)
        img_recon = transforms.ToTensor()(transforms.Resize(224)(transforms.ToPILImage()(recon[b])))

        fig, axs = plt.subplots(1,2,figsize=(6,3))
        axs[0].imshow(img_recon.permute(1,2,0).numpy()); axs[0].set_title("Reconstruction"); axs[0].axis('off')
        axs[1].imshow(img_ds.permute(1,2,0).numpy()); axs[1].set_title(f"Nearest (idx={ds_idx})"); axs[1].axis('off')
        plt.suptitle(f"Recon {img_idx:03d} -- distance: {dists[b,ds_idx].item():.4f}")
        pair_save = os.path.join(save_pairs_dir, f"recon_{img_idx:03d}_nearest_{ds_idx:05d}.png")
        plt.savefig(pair_save, bbox_inches='tight', dpi=150)
        plt.show()
        plt.close(fig)

        vutils.save_image(img_recon, os.path.join(save_pairs_dir, f"recon_{img_idx:03d}.png"))
        vutils.save_image(img_ds,    os.path.join(save_pairs_dir, f"nearest_{img_idx:03d}.png"))
        print(f"Saved pair: {pair_save} -- dataset path: {ds_path}")
        global_saved += 1

print(f"\nDone. Saved {global_saved} recon vs nearest pairs to: {save_pairs_dir}")


# %% [markdown] Cell 8
# #Testing on resnet 152 which was pre trained image net
#
#
#

# %% Cell 9
# ------------------- Imports -------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os

# ------------------- User Settings -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = "deepinv_outputs"
os.makedirs(out_dir, exist_ok=True)

# Decoder must be already defined and trained
# from previous code: decoder = ...

large = True  # True -> ResNet50/101/152, False -> ResNet18/34
B = 1        # batch size

# Embedding & skip shapes
if large:
    z_shape = (B, 2048, 7, 7)
    skip_shapes = [(B, 1024, 14, 14), (B, 512, 28, 28), (B, 256, 56, 56)]
else:
    z_shape = (B, 512, 7, 7)
    skip_shapes = [(B, 256, 14, 14), (B, 128, 28, 28), (B, 64, 56, 56)]

# Optimization params
num_steps = 300
save_every = 50
lr = 0.1
l2_weight = 1.0
tv_weight = 1e-4
bn_weight = 1.0
z_l2_weight = 1e-6
optimize_skips = False
use_image_denorm = False
n_images = 5  # Number of reconstructions

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

# ------------------- Dataset Settings -------------------
dataset_root = "/content/imagenet_images"   # ImageFolder-style
dataset_batch = 64
feature_cache_path = os.path.join(out_dir, "dataset_features.pt")
compare_distance = "cosine"
save_pairs_dir = os.path.join(out_dir, "recon_vs_nearest")
os.makedirs(save_pairs_dir, exist_ok=True)

preprocess_for_encoder = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
display_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
dataset_for_features = datasets.ImageFolder(dataset_root, transform=preprocess_for_encoder)
dataset_for_display  = datasets.ImageFolder(dataset_root, transform=display_transform)
dataset_loader = DataLoader(dataset_for_features, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)
display_loader = DataLoader(dataset_for_display, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)

# ------------------- Build Frozen Encoder for BN Loss -------------------
if large:
    enc_model_full = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
else:
    enc_model_full = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone_modules = list(enc_model_full.children())[:-2]  # remove avgpool + fc
encoder_for_bn = nn.Sequential(*backbone_modules).to(device).eval()
for p in encoder_for_bn.parameters():
    p.requires_grad = False

print("Using encoder_for_bn with final channels =", z_shape[1])

# ------------------- Helper Functions -------------------
def denorm_if_needed(img_tensor):
    t = img_tensor
    if use_image_denorm:
        t = (t * IMAGENET_STD) + IMAGENET_MEAN
    return t.clamp(0,1)

def save_and_show(tensor_img, path, show=True):
    t = denorm_if_needed(tensor_img.detach().cpu())
    vutils.save_image(t, path)
    if show:
        B = t.shape[0]
        for i in range(B):
            np_img = t[i].permute(1,2,0).numpy()
            plt.figure(figsize=(4,4))
            plt.imshow(np_img)
            plt.axis('off')
            plt.show()

def tv_loss(img):
    dy = torch.mean(torch.abs(img[:,:,1:,:] - img[:,:,:-1,:]))
    dx = torch.mean(torch.abs(img[:,:,:,1:] - img[:,:,:,:-1]))
    return dx + dy

def bn_stat_loss_from_encoder(decoder_out, encoder_model, device):
    bn_features = []
    hooks = []

    def hook_fn(module, inp, out):
        bn_features.append(out)

    for module in encoder_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(hook_fn))

    _ = encoder_model(decoder_out)

    losses = []
    bn_modules = [m for m in encoder_model.modules() if isinstance(m, nn.BatchNorm2d)]
    for out_bn, module in zip(bn_features, bn_modules):
        mean_batch = out_bn.mean(dim=[0,2,3])
        var_batch  = out_bn.var(dim=[0,2,3], unbiased=False)
        losses.append(((mean_batch - module.running_mean.to(device))**2).mean())
        losses.append(((var_batch  - module.running_var.to(device))**2).mean())

    for h in hooks:
        h.remove()

    return sum(losses)

# ------------------- Extract / Cache Dataset Features -------------------
def extract_and_cache_dataset_features(encoder, loader, cache_path, device):
    if os.path.exists(cache_path):
        print("Loading cached dataset features:", cache_path)
        cached = torch.load(cache_path, map_location='cpu')
        features = cached['features']
        paths    = cached['paths']
        return features, paths

    encoder.eval()
    all_feats, all_paths = [], []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            feats = encoder(x)
            feats_gap = feats.mean(dim=[2,3])
            all_feats.append(feats_gap.cpu())
            start_idx = batch_idx * loader.batch_size
            end_idx = start_idx + feats_gap.shape[0]
            for i in range(start_idx, min(end_idx, len(dataset_for_features.samples))):
                all_paths.append(dataset_for_features.samples[i][0])
    features = torch.cat(all_feats, dim=0)
    torch.save({'features': features, 'paths': all_paths}, cache_path)
    print("Cached dataset features to:", cache_path)
    return features, all_paths

dataset_features, dataset_paths = extract_and_cache_dataset_features(encoder_for_bn, dataset_loader, feature_cache_path, device)
dataset_features = dataset_features.to(device)
print("Dataset size:", dataset_features.shape[0], "feature-dim:", dataset_features.shape[1])

# ------------------- Projection Layer if Needed -------------------
with torch.no_grad():
    sample_input = torch.randn(1,3,224,224, device=device)
    sample_feat  = encoder_for_bn(sample_input).mean(dim=[2,3])
feat_dim = sample_feat.shape[1]
dataset_feat_dim = dataset_features.shape[1]

if dataset_feat_dim != feat_dim:
    print(f"Projecting dataset features from {dataset_feat_dim} -> {feat_dim}")
    proj_layer = nn.Linear(dataset_feat_dim, feat_dim, bias=False).to(device)
    dataset_features_proj = proj_layer(dataset_features)
else:
    proj_layer = None
    dataset_features_proj = dataset_features

# ------------------- Helper: get feature vector for reconstructed image -------------------
# ---------- Projection Layer if Needed (create dataset -> encoder projection only) ----------
with torch.no_grad():
    sample_input = torch.randn(1,3,224,224, device=device)
    sample_feat  = encoder_for_bn(sample_input).mean(dim=[2,3])
feat_dim = sample_feat.shape[1]            # encoder output dim (e.g. 2048)
dataset_feat_dim = dataset_features.shape[1]  # cached features dim (e.g. 512)

proj_layer = None
dataset_features_proj = dataset_features

if dataset_feat_dim != feat_dim:
    print(f"Projecting dataset features from {dataset_feat_dim} -> {feat_dim}")
    # projection maps dataset_dim -> encoder_dim
    proj_layer = nn.Linear(dataset_feat_dim, feat_dim, bias=False).to(device)
    # apply projection only to the dataset features (N, dataset_feat_dim) -> (N, feat_dim)
    with torch.no_grad():
        dataset_features_proj = proj_layer(dataset_features)  # (N, feat_dim)
else:
    print("No projection needed: dataset features and encoder features match.")
def get_feature_for_img_tensor(img_tensor, encoder, device, proj_layer=None, dataset_feat_dim=None):
    """
    img_tensor: (B,3,H,W) in [0,1]
    Returns: feats (B, feat_dim) matching encoder output dim.
    If proj_layer is provided, will APPLY IT ONLY if recon feats have dimension == proj_layer.in_features.
    """
    encoder.eval()
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    imgs_resized = torch.zeros((img_tensor.shape[0],3,224,224), device=device)
    for i in range(img_tensor.shape[0]):
        pil = transforms.ToPILImage()(img_tensor[i].cpu())
        proc = preprocess(pil)
        t = transforms.ToTensor()(proc).to(device)
        t = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(t)
        imgs_resized[i] = t

    with torch.no_grad():
        feats = encoder(imgs_resized).mean(dim=[2,3])  # (B, feat_dim_encoder)

        # If proj_layer exists, only use it *if* feats currently match dataset_feat_dim (i.e. need projection).
        # But typically we want: dataset_features_proj has shape (N, feat_dim_encoder), so we should NOT project recon feats.
        # We therefore apply proj_layer only when the recon feats' dim equals proj_layer.in_features.
        if proj_layer is not None:
            # infer in_features from proj_layer.weight shape
            in_features = proj_layer.weight.shape[1]
            if feats.shape[1] == in_features:
                feats = proj_layer(feats)
            else:
                # feats already in encoder dimension — do NOT project
                pass

    return feats

# ------------------- Distance function -------------------
def compute_nearest_dataset_idx(query_feats, dataset_feats, metric="euclidean"):
    if metric=="euclidean":
        dists = torch.cdist(query_feats, dataset_feats)
        idx = torch.argmin(dists, dim=1)
        return idx, dists
    elif metric=="cosine":
        qn = torch.nn.functional.normalize(query_feats, dim=1)
        dn = torch.nn.functional.normalize(dataset_feats, dim=1)
        sims = qn @ dn.t()
        idx = torch.argmax(sims, dim=1)
        return idx, sims
    else:
        raise ValueError("Unsupported metric")

# ------------------- Main Reconstruction Loop -------------------
global_saved = 0
for img_idx in range(n_images):
    print(f"\n--- Generating reconstruction {img_idx+1}/{n_images} ---")

    # Initialize embeddings
    z = torch.randn(z_shape, device=device, requires_grad=True)
    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]
    optimizer_emb = optim.Adam(opt_params, lr=lr)

    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])
        loss = l2_weight*torch.mean(out**2) + tv_weight*tv_loss(out) + z_l2_weight*torch.mean(z**2)
        if bn_weight>0:
            loss += bn_weight*bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

        if step % 50==0 or step==1:
            print(f"step {step}/{num_steps}  loss={loss.item():.6f}")
            save_path = os.path.join(out_dir, f"deepinv_step{step:04d}.png")
            save_and_show(out, save_path, show=False)

    # Final reconstruction
    recon = denorm_if_needed(out.detach().cpu())
    recon_for_enc = recon.to(device)
    recon_feats = get_feature_for_img_tensor(recon_for_enc, encoder_for_bn, device, proj_layer)
    idxs, dists = compute_nearest_dataset_idx(recon_feats, dataset_features_proj, metric=compare_distance)

    for b in range(recon.shape[0]):
        ds_idx = idxs[b].item()
        ds_path = dataset_paths[ds_idx]
        img_ds = Image.open(ds_path).convert('RGB')
        img_ds = display_transform(img_ds)
        img_recon = transforms.ToTensor()(transforms.Resize(224)(transforms.ToPILImage()(recon[b])))

        fig, axs = plt.subplots(1,2,figsize=(6,3))
        axs[0].imshow(img_recon.permute(1,2,0).numpy()); axs[0].set_title("Reconstruction"); axs[0].axis('off')
        axs[1].imshow(img_ds.permute(1,2,0).numpy()); axs[1].set_title(f"Nearest (idx={ds_idx})"); axs[1].axis('off')
        plt.suptitle(f"Recon {img_idx:03d} -- distance: {dists[b,ds_idx].item():.4f}")
        pair_save = os.path.join(save_pairs_dir, f"recon_{img_idx:03d}_nearest_{ds_idx:05d}.png")
        plt.savefig(pair_save, bbox_inches='tight', dpi=150)
        plt.show()
        plt.close(fig)

        vutils.save_image(img_recon, os.path.join(save_pairs_dir, f"recon_{img_idx:03d}.png"))
        vutils.save_image(img_ds,    os.path.join(save_pairs_dir, f"nearest_{img_idx:03d}.png"))
        print(f"Saved pair: {pair_save} -- dataset path: {ds_path}")
        global_saved += 1

print(f"\nDone. Saved {global_saved} recon vs nearest pairs to: {save_pairs_dir}")


# %% [markdown] Cell 10
# #Testing on resnet 34 which was pre trained image net

# %% Cell 11
# ------------------- Imports -------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os

# ------------------- User Settings -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = "deepinv_outputs"
os.makedirs(out_dir, exist_ok=True)

# Decoder must be already defined and trained
# from previous code: decoder = ...

large = False  # True -> ResNet50/101/152, False -> ResNet18/34
B = 1        # batch size

# Embedding & skip shapes
if large:
    z_shape = (B, 2048, 7, 7)
    skip_shapes = [(B, 1024, 14, 14), (B, 512, 28, 28), (B, 256, 56, 56)]
else:
    z_shape = (B, 512, 7, 7)
    skip_shapes = [(B, 256, 14, 14), (B, 128, 28, 28), (B, 64, 56, 56)]

# Optimization params
num_steps = 300
save_every = 50
lr = 0.1
l2_weight = 1.0
tv_weight = 1e-4
bn_weight = 1.0
z_l2_weight = 1e-6
optimize_skips = False
use_image_denorm = False
n_images = 5  # Number of reconstructions

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

# ------------------- Dataset Settings -------------------
dataset_root = "/content/imagenet_images"   # ImageFolder-style
dataset_batch = 64
feature_cache_path = os.path.join(out_dir, "dataset_features.pt")
compare_distance = "cosine"
save_pairs_dir = os.path.join(out_dir, "recon_vs_nearest")
os.makedirs(save_pairs_dir, exist_ok=True)

preprocess_for_encoder = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
display_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
dataset_for_features = datasets.ImageFolder(dataset_root, transform=preprocess_for_encoder)
dataset_for_display  = datasets.ImageFolder(dataset_root, transform=display_transform)
dataset_loader = DataLoader(dataset_for_features, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)
display_loader = DataLoader(dataset_for_display, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)

# ------------------- Build Frozen Encoder for BN Loss -------------------
if large:
    enc_model_full = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
else:
    enc_model_full = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
backbone_modules = list(enc_model_full.children())[:-2]  # remove avgpool + fc
encoder_for_bn = nn.Sequential(*backbone_modules).to(device).eval()
for p in encoder_for_bn.parameters():
    p.requires_grad = False

print("Using encoder_for_bn with final channels =", z_shape[1])

# ------------------- Helper Functions -------------------
def denorm_if_needed(img_tensor):
    t = img_tensor
    if use_image_denorm:
        t = (t * IMAGENET_STD) + IMAGENET_MEAN
    return t.clamp(0,1)

def save_and_show(tensor_img, path, show=True):
    t = denorm_if_needed(tensor_img.detach().cpu())
    vutils.save_image(t, path)
    if show:
        B = t.shape[0]
        for i in range(B):
            np_img = t[i].permute(1,2,0).numpy()
            plt.figure(figsize=(4,4))
            plt.imshow(np_img)
            plt.axis('off')
            plt.show()

def tv_loss(img):
    dy = torch.mean(torch.abs(img[:,:,1:,:] - img[:,:,:-1,:]))
    dx = torch.mean(torch.abs(img[:,:,:,1:] - img[:,:,:,:-1]))
    return dx + dy

def bn_stat_loss_from_encoder(decoder_out, encoder_model, device):
    bn_features = []
    hooks = []

    def hook_fn(module, inp, out):
        bn_features.append(out)

    for module in encoder_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(hook_fn))

    _ = encoder_model(decoder_out)

    losses = []
    bn_modules = [m for m in encoder_model.modules() if isinstance(m, nn.BatchNorm2d)]
    for out_bn, module in zip(bn_features, bn_modules):
        mean_batch = out_bn.mean(dim=[0,2,3])
        var_batch  = out_bn.var(dim=[0,2,3], unbiased=False)
        losses.append(((mean_batch - module.running_mean.to(device))**2).mean())
        losses.append(((var_batch  - module.running_var.to(device))**2).mean())

    for h in hooks:
        h.remove()

    return sum(losses)

# ------------------- Extract / Cache Dataset Features -------------------
def extract_and_cache_dataset_features(encoder, loader, cache_path, device):
    if os.path.exists(cache_path):
        print("Loading cached dataset features:", cache_path)
        cached = torch.load(cache_path, map_location='cpu')
        features = cached['features']
        paths    = cached['paths']
        return features, paths

    encoder.eval()
    all_feats, all_paths = [], []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            feats = encoder(x)
            feats_gap = feats.mean(dim=[2,3])
            all_feats.append(feats_gap.cpu())
            start_idx = batch_idx * loader.batch_size
            end_idx = start_idx + feats_gap.shape[0]
            for i in range(start_idx, min(end_idx, len(dataset_for_features.samples))):
                all_paths.append(dataset_for_features.samples[i][0])
    features = torch.cat(all_feats, dim=0)
    torch.save({'features': features, 'paths': all_paths}, cache_path)
    print("Cached dataset features to:", cache_path)
    return features, all_paths

dataset_features, dataset_paths = extract_and_cache_dataset_features(encoder_for_bn, dataset_loader, feature_cache_path, device)
dataset_features = dataset_features.to(device)
print("Dataset size:", dataset_features.shape[0], "feature-dim:", dataset_features.shape[1])

# ------------------- Projection Layer if Needed -------------------
with torch.no_grad():
    sample_input = torch.randn(1,3,224,224, device=device)
    sample_feat  = encoder_for_bn(sample_input).mean(dim=[2,3])
feat_dim = sample_feat.shape[1]
dataset_feat_dim = dataset_features.shape[1]

if dataset_feat_dim != feat_dim:
    print(f"Projecting dataset features from {dataset_feat_dim} -> {feat_dim}")
    proj_layer = nn.Linear(dataset_feat_dim, feat_dim, bias=False).to(device)
    dataset_features_proj = proj_layer(dataset_features)
else:
    proj_layer = None
    dataset_features_proj = dataset_features

# ------------------- Helper: get feature vector for reconstructed image -------------------
# ---------- Projection Layer if Needed (create dataset -> encoder projection only) ----------
with torch.no_grad():
    sample_input = torch.randn(1,3,224,224, device=device)
    sample_feat  = encoder_for_bn(sample_input).mean(dim=[2,3])
feat_dim = sample_feat.shape[1]            # encoder output dim (e.g. 2048)
dataset_feat_dim = dataset_features.shape[1]  # cached features dim (e.g. 512)

proj_layer = None
dataset_features_proj = dataset_features

if dataset_feat_dim != feat_dim:
    print(f"Projecting dataset features from {dataset_feat_dim} -> {feat_dim}")
    # projection maps dataset_dim -> encoder_dim
    proj_layer = nn.Linear(dataset_feat_dim, feat_dim, bias=False).to(device)
    # apply projection only to the dataset features (N, dataset_feat_dim) -> (N, feat_dim)
    with torch.no_grad():
        dataset_features_proj = proj_layer(dataset_features)  # (N, feat_dim)
else:
    print("No projection needed: dataset features and encoder features match.")
def get_feature_for_img_tensor(img_tensor, encoder, device, proj_layer=None, dataset_feat_dim=None):
    """
    img_tensor: (B,3,H,W) in [0,1]
    Returns: feats (B, feat_dim) matching encoder output dim.
    If proj_layer is provided, will APPLY IT ONLY if recon feats have dimension == proj_layer.in_features.
    """
    encoder.eval()
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    imgs_resized = torch.zeros((img_tensor.shape[0],3,224,224), device=device)
    for i in range(img_tensor.shape[0]):
        pil = transforms.ToPILImage()(img_tensor[i].cpu())
        proc = preprocess(pil)
        t = transforms.ToTensor()(proc).to(device)
        t = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(t)
        imgs_resized[i] = t

    with torch.no_grad():
        feats = encoder(imgs_resized).mean(dim=[2,3])  # (B, feat_dim_encoder)

        # If proj_layer exists, only use it *if* feats currently match dataset_feat_dim (i.e. need projection).
        # But typically we want: dataset_features_proj has shape (N, feat_dim_encoder), so we should NOT project recon feats.
        # We therefore apply proj_layer only when the recon feats' dim equals proj_layer.in_features.
        if proj_layer is not None:
            # infer in_features from proj_layer.weight shape
            in_features = proj_layer.weight.shape[1]
            if feats.shape[1] == in_features:
                feats = proj_layer(feats)
            else:
                # feats already in encoder dimension — do NOT project
                pass

    return feats

# ------------------- Distance function -------------------
def compute_nearest_dataset_idx(query_feats, dataset_feats, metric="euclidean"):
    if metric=="euclidean":
        dists = torch.cdist(query_feats, dataset_feats)
        idx = torch.argmin(dists, dim=1)
        return idx, dists
    elif metric=="cosine":
        qn = torch.nn.functional.normalize(query_feats, dim=1)
        dn = torch.nn.functional.normalize(dataset_feats, dim=1)
        sims = qn @ dn.t()
        idx = torch.argmax(sims, dim=1)
        return idx, sims
    else:
        raise ValueError("Unsupported metric")

# ------------------- Main Reconstruction Loop -------------------
global_saved = 0
for img_idx in range(n_images):
    print(f"\n--- Generating reconstruction {img_idx+1}/{n_images} ---")

    # Initialize embeddings
    z = torch.randn(z_shape, device=device, requires_grad=True)
    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]
    optimizer_emb = optim.Adam(opt_params, lr=lr)

    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])
        loss = l2_weight*torch.mean(out**2) + tv_weight*tv_loss(out) + z_l2_weight*torch.mean(z**2)
        if bn_weight>0:
            loss += bn_weight*bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

        if step % 50==0 or step==1:
            print(f"step {step}/{num_steps}  loss={loss.item():.6f}")
            save_path = os.path.join(out_dir, f"deepinv_step{step:04d}.png")
            save_and_show(out, save_path, show=False)

    # Final reconstruction
    recon = denorm_if_needed(out.detach().cpu())
    recon_for_enc = recon.to(device)
    recon_feats = get_feature_for_img_tensor(recon_for_enc, encoder_for_bn, device, proj_layer)
    idxs, dists = compute_nearest_dataset_idx(recon_feats, dataset_features_proj, metric=compare_distance)

    for b in range(recon.shape[0]):
        ds_idx = idxs[b].item()
        ds_path = dataset_paths[ds_idx]
        img_ds = Image.open(ds_path).convert('RGB')
        img_ds = display_transform(img_ds)
        img_recon = transforms.ToTensor()(transforms.Resize(224)(transforms.ToPILImage()(recon[b])))

        fig, axs = plt.subplots(1,2,figsize=(6,3))
        axs[0].imshow(img_recon.permute(1,2,0).numpy()); axs[0].set_title("Reconstruction"); axs[0].axis('off')
        axs[1].imshow(img_ds.permute(1,2,0).numpy()); axs[1].set_title(f"Nearest (idx={ds_idx})"); axs[1].axis('off')
        plt.suptitle(f"Recon {img_idx:03d} -- distance: {dists[b,ds_idx].item():.4f}")
        pair_save = os.path.join(save_pairs_dir, f"recon_{img_idx:03d}_nearest_{ds_idx:05d}.png")
        plt.savefig(pair_save, bbox_inches='tight', dpi=150)
        plt.show()
        plt.close(fig)

        vutils.save_image(img_recon, os.path.join(save_pairs_dir, f"recon_{img_idx:03d}.png"))
        vutils.save_image(img_ds,    os.path.join(save_pairs_dir, f"nearest_{img_idx:03d}.png"))
        print(f"Saved pair: {pair_save} -- dataset path: {ds_path}")
        global_saved += 1

print(f"\nDone. Saved {global_saved} recon vs nearest pairs to: {save_pairs_dir}")


# %% [markdown] Cell 12
# #Testing on resnet 101 which was pretrained imagenet
#

# %% Cell 13
# ------------------- Imports -------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os

# ------------------- User Settings -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = "deepinv_outputs"
os.makedirs(out_dir, exist_ok=True)

# Decoder must be already defined and trained
# from previous code: decoder = ...

large = True  # True -> ResNet50/101/152, False -> ResNet18/34
B = 1        # batch size

# Embedding & skip shapes
if large:
    z_shape = (B, 2048, 7, 7)
    skip_shapes = [(B, 1024, 14, 14), (B, 512, 28, 28), (B, 256, 56, 56)]
else:
    z_shape = (B, 512, 7, 7)
    skip_shapes = [(B, 256, 14, 14), (B, 128, 28, 28), (B, 64, 56, 56)]

# Optimization params
num_steps = 300
save_every = 50
lr = 0.1
l2_weight = 1.0
tv_weight = 1e-4
bn_weight = 1.0
z_l2_weight = 1e-6
optimize_skips = False
use_image_denorm = False
n_images = 5  # Number of reconstructions

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

# ------------------- Dataset Settings -------------------
dataset_root = "/content/imagenet_images"   # ImageFolder-style
dataset_batch = 64
feature_cache_path = os.path.join(out_dir, "dataset_features.pt")
compare_distance = "cosine"
save_pairs_dir = os.path.join(out_dir, "recon_vs_nearest")
os.makedirs(save_pairs_dir, exist_ok=True)

preprocess_for_encoder = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
display_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
dataset_for_features = datasets.ImageFolder(dataset_root, transform=preprocess_for_encoder)
dataset_for_display  = datasets.ImageFolder(dataset_root, transform=display_transform)
dataset_loader = DataLoader(dataset_for_features, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)
display_loader = DataLoader(dataset_for_display, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)

# ------------------- Build Frozen Encoder for BN Loss -------------------
if large:
    enc_model_full = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
else:
    enc_model_full = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone_modules = list(enc_model_full.children())[:-2]  # remove avgpool + fc
encoder_for_bn = nn.Sequential(*backbone_modules).to(device).eval()
for p in encoder_for_bn.parameters():
    p.requires_grad = False

print("Using encoder_for_bn with final channels =", z_shape[1])

# ------------------- Helper Functions -------------------
def denorm_if_needed(img_tensor):
    t = img_tensor
    if use_image_denorm:
        t = (t * IMAGENET_STD) + IMAGENET_MEAN
    return t.clamp(0,1)

def save_and_show(tensor_img, path, show=True):
    t = denorm_if_needed(tensor_img.detach().cpu())
    vutils.save_image(t, path)
    if show:
        B = t.shape[0]
        for i in range(B):
            np_img = t[i].permute(1,2,0).numpy()
            plt.figure(figsize=(4,4))
            plt.imshow(np_img)
            plt.axis('off')
            plt.show()

def tv_loss(img):
    dy = torch.mean(torch.abs(img[:,:,1:,:] - img[:,:,:-1,:]))
    dx = torch.mean(torch.abs(img[:,:,:,1:] - img[:,:,:,:-1]))
    return dx + dy

def bn_stat_loss_from_encoder(decoder_out, encoder_model, device):
    bn_features = []
    hooks = []

    def hook_fn(module, inp, out):
        bn_features.append(out)

    for module in encoder_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(hook_fn))

    _ = encoder_model(decoder_out)

    losses = []
    bn_modules = [m for m in encoder_model.modules() if isinstance(m, nn.BatchNorm2d)]
    for out_bn, module in zip(bn_features, bn_modules):
        mean_batch = out_bn.mean(dim=[0,2,3])
        var_batch  = out_bn.var(dim=[0,2,3], unbiased=False)
        losses.append(((mean_batch - module.running_mean.to(device))**2).mean())
        losses.append(((var_batch  - module.running_var.to(device))**2).mean())

    for h in hooks:
        h.remove()

    return sum(losses)

# ------------------- Extract / Cache Dataset Features -------------------
def extract_and_cache_dataset_features(encoder, loader, cache_path, device):
    if os.path.exists(cache_path):
        print("Loading cached dataset features:", cache_path)
        cached = torch.load(cache_path, map_location='cpu')
        features = cached['features']
        paths    = cached['paths']
        return features, paths

    encoder.eval()
    all_feats, all_paths = [], []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            feats = encoder(x)
            feats_gap = feats.mean(dim=[2,3])
            all_feats.append(feats_gap.cpu())
            start_idx = batch_idx * loader.batch_size
            end_idx = start_idx + feats_gap.shape[0]
            for i in range(start_idx, min(end_idx, len(dataset_for_features.samples))):
                all_paths.append(dataset_for_features.samples[i][0])
    features = torch.cat(all_feats, dim=0)
    torch.save({'features': features, 'paths': all_paths}, cache_path)
    print("Cached dataset features to:", cache_path)
    return features, all_paths

dataset_features, dataset_paths = extract_and_cache_dataset_features(encoder_for_bn, dataset_loader, feature_cache_path, device)
dataset_features = dataset_features.to(device)
print("Dataset size:", dataset_features.shape[0], "feature-dim:", dataset_features.shape[1])

# ------------------- Projection Layer if Needed -------------------
with torch.no_grad():
    sample_input = torch.randn(1,3,224,224, device=device)
    sample_feat  = encoder_for_bn(sample_input).mean(dim=[2,3])
feat_dim = sample_feat.shape[1]
dataset_feat_dim = dataset_features.shape[1]

if dataset_feat_dim != feat_dim:
    print(f"Projecting dataset features from {dataset_feat_dim} -> {feat_dim}")
    proj_layer = nn.Linear(dataset_feat_dim, feat_dim, bias=False).to(device)
    dataset_features_proj = proj_layer(dataset_features)
else:
    proj_layer = None
    dataset_features_proj = dataset_features

# ------------------- Helper: get feature vector for reconstructed image -------------------
# ---------- Projection Layer if Needed (create dataset -> encoder projection only) ----------
with torch.no_grad():
    sample_input = torch.randn(1,3,224,224, device=device)
    sample_feat  = encoder_for_bn(sample_input).mean(dim=[2,3])
feat_dim = sample_feat.shape[1]            # encoder output dim (e.g. 2048)
dataset_feat_dim = dataset_features.shape[1]  # cached features dim (e.g. 512)

proj_layer = None
dataset_features_proj = dataset_features

if dataset_feat_dim != feat_dim:
    print(f"Projecting dataset features from {dataset_feat_dim} -> {feat_dim}")
    # projection maps dataset_dim -> encoder_dim
    proj_layer = nn.Linear(dataset_feat_dim, feat_dim, bias=False).to(device)
    # apply projection only to the dataset features (N, dataset_feat_dim) -> (N, feat_dim)
    with torch.no_grad():
        dataset_features_proj = proj_layer(dataset_features)  # (N, feat_dim)
else:
    print("No projection needed: dataset features and encoder features match.")
def get_feature_for_img_tensor(img_tensor, encoder, device, proj_layer=None, dataset_feat_dim=None):
    """
    img_tensor: (B,3,H,W) in [0,1]
    Returns: feats (B, feat_dim) matching encoder output dim.
    If proj_layer is provided, will APPLY IT ONLY if recon feats have dimension == proj_layer.in_features.
    """
    encoder.eval()
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    imgs_resized = torch.zeros((img_tensor.shape[0],3,224,224), device=device)
    for i in range(img_tensor.shape[0]):
        pil = transforms.ToPILImage()(img_tensor[i].cpu())
        proc = preprocess(pil)
        t = transforms.ToTensor()(proc).to(device)
        t = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(t)
        imgs_resized[i] = t

    with torch.no_grad():
        feats = encoder(imgs_resized).mean(dim=[2,3])  # (B, feat_dim_encoder)

        # If proj_layer exists, only use it *if* feats currently match dataset_feat_dim (i.e. need projection).
        # But typically we want: dataset_features_proj has shape (N, feat_dim_encoder), so we should NOT project recon feats.
        # We therefore apply proj_layer only when the recon feats' dim equals proj_layer.in_features.
        if proj_layer is not None:
            # infer in_features from proj_layer.weight shape
            in_features = proj_layer.weight.shape[1]
            if feats.shape[1] == in_features:
                feats = proj_layer(feats)
            else:
                # feats already in encoder dimension — do NOT project
                pass

    return feats

# ------------------- Distance function -------------------
def compute_nearest_dataset_idx(query_feats, dataset_feats, metric="euclidean"):
    if metric=="euclidean":
        dists = torch.cdist(query_feats, dataset_feats)
        idx = torch.argmin(dists, dim=1)
        return idx, dists
    elif metric=="cosine":
        qn = torch.nn.functional.normalize(query_feats, dim=1)
        dn = torch.nn.functional.normalize(dataset_feats, dim=1)
        sims = qn @ dn.t()
        idx = torch.argmax(sims, dim=1)
        return idx, sims
    else:
        raise ValueError("Unsupported metric")

# ------------------- Main Reconstruction Loop -------------------
global_saved = 0
for img_idx in range(n_images):
    print(f"\n--- Generating reconstruction {img_idx+1}/{n_images} ---")

    # Initialize embeddings
    z = torch.randn(z_shape, device=device, requires_grad=True)
    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]
    optimizer_emb = optim.Adam(opt_params, lr=lr)

    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])
        loss = l2_weight*torch.mean(out**2) + tv_weight*tv_loss(out) + z_l2_weight*torch.mean(z**2)
        if bn_weight>0:
            loss += bn_weight*bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

        if step % 50==0 or step==1:
            print(f"step {step}/{num_steps}  loss={loss.item():.6f}")
            save_path = os.path.join(out_dir, f"deepinv_step{step:04d}.png")
            save_and_show(out, save_path, show=False)

    # Final reconstruction
    recon = denorm_if_needed(out.detach().cpu())
    recon_for_enc = recon.to(device)
    recon_feats = get_feature_for_img_tensor(recon_for_enc, encoder_for_bn, device, proj_layer)
    idxs, dists = compute_nearest_dataset_idx(recon_feats, dataset_features_proj, metric=compare_distance)

    for b in range(recon.shape[0]):
        ds_idx = idxs[b].item()
        ds_path = dataset_paths[ds_idx]
        img_ds = Image.open(ds_path).convert('RGB')
        img_ds = display_transform(img_ds)
        img_recon = transforms.ToTensor()(transforms.Resize(224)(transforms.ToPILImage()(recon[b])))

        fig, axs = plt.subplots(1,2,figsize=(6,3))
        axs[0].imshow(img_recon.permute(1,2,0).numpy()); axs[0].set_title("Reconstruction"); axs[0].axis('off')
        axs[1].imshow(img_ds.permute(1,2,0).numpy()); axs[1].set_title(f"Nearest (idx={ds_idx})"); axs[1].axis('off')
        plt.suptitle(f"Recon {img_idx:03d} -- distance: {dists[b,ds_idx].item():.4f}")
        pair_save = os.path.join(save_pairs_dir, f"recon_{img_idx:03d}_nearest_{ds_idx:05d}.png")
        plt.savefig(pair_save, bbox_inches='tight', dpi=150)
        plt.show()
        plt.close(fig)

        vutils.save_image(img_recon, os.path.join(save_pairs_dir, f"recon_{img_idx:03d}.png"))
        vutils.save_image(img_ds,    os.path.join(save_pairs_dir, f"nearest_{img_idx:03d}.png"))
        print(f"Saved pair: {pair_save} -- dataset path: {ds_path}")
        global_saved += 1

print(f"\nDone. Saved {global_saved} recon vs nearest pairs to: {save_pairs_dir}")


# %% [markdown] Cell 14
# #Testing on resnet 50 which was pretrained on imagenet

# %% Cell 15
# ------------------- Imports -------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os

# ------------------- User Settings -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = "deepinv_outputs"
os.makedirs(out_dir, exist_ok=True)

# Decoder must be already defined and trained
# from previous code: decoder = ...

large = True  # True -> ResNet50/101/152, False -> ResNet18/34
B = 1        # batch size

# Embedding & skip shapes
if large:
    z_shape = (B, 2048, 7, 7)
    skip_shapes = [(B, 1024, 14, 14), (B, 512, 28, 28), (B, 256, 56, 56)]
else:
    z_shape = (B, 512, 7, 7)
    skip_shapes = [(B, 256, 14, 14), (B, 128, 28, 28), (B, 64, 56, 56)]

# Optimization params
num_steps = 300
save_every = 50
lr = 0.1
l2_weight = 1.0
tv_weight = 1e-4
bn_weight = 1.0
z_l2_weight = 1e-6
optimize_skips = False
use_image_denorm = False
n_images = 5  # Number of reconstructions

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

# ------------------- Dataset Settings -------------------
dataset_root = "/content/imagenet_images"   # ImageFolder-style
dataset_batch = 64
feature_cache_path = os.path.join(out_dir, "dataset_features.pt")
compare_distance = "cosine"
save_pairs_dir = os.path.join(out_dir, "recon_vs_nearest")
os.makedirs(save_pairs_dir, exist_ok=True)

preprocess_for_encoder = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
display_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
dataset_for_features = datasets.ImageFolder(dataset_root, transform=preprocess_for_encoder)
dataset_for_display  = datasets.ImageFolder(dataset_root, transform=display_transform)
dataset_loader = DataLoader(dataset_for_features, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)
display_loader = DataLoader(dataset_for_display, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)

# ------------------- Build Frozen Encoder for BN Loss -------------------
if large:
    enc_model_full = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
else:
    enc_model_full = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone_modules = list(enc_model_full.children())[:-2]  # remove avgpool + fc
encoder_for_bn = nn.Sequential(*backbone_modules).to(device).eval()
for p in encoder_for_bn.parameters():
    p.requires_grad = False

print("Using encoder_for_bn with final channels =", z_shape[1])

# ------------------- Helper Functions -------------------
def denorm_if_needed(img_tensor):
    t = img_tensor
    if use_image_denorm:
        t = (t * IMAGENET_STD) + IMAGENET_MEAN
    return t.clamp(0,1)

def save_and_show(tensor_img, path, show=True):
    t = denorm_if_needed(tensor_img.detach().cpu())
    vutils.save_image(t, path)
    if show:
        B = t.shape[0]
        for i in range(B):
            np_img = t[i].permute(1,2,0).numpy()
            plt.figure(figsize=(4,4))
            plt.imshow(np_img)
            plt.axis('off')
            plt.show()

def tv_loss(img):
    dy = torch.mean(torch.abs(img[:,:,1:,:] - img[:,:,:-1,:]))
    dx = torch.mean(torch.abs(img[:,:,:,1:] - img[:,:,:,:-1]))
    return dx + dy

def bn_stat_loss_from_encoder(decoder_out, encoder_model, device):
    bn_features = []
    hooks = []

    def hook_fn(module, inp, out):
        bn_features.append(out)

    for module in encoder_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(hook_fn))

    _ = encoder_model(decoder_out)

    losses = []
    bn_modules = [m for m in encoder_model.modules() if isinstance(m, nn.BatchNorm2d)]
    for out_bn, module in zip(bn_features, bn_modules):
        mean_batch = out_bn.mean(dim=[0,2,3])
        var_batch  = out_bn.var(dim=[0,2,3], unbiased=False)
        losses.append(((mean_batch - module.running_mean.to(device))**2).mean())
        losses.append(((var_batch  - module.running_var.to(device))**2).mean())

    for h in hooks:
        h.remove()

    return sum(losses)

# ------------------- Extract / Cache Dataset Features -------------------
def extract_and_cache_dataset_features(encoder, loader, cache_path, device):
    if os.path.exists(cache_path):
        print("Loading cached dataset features:", cache_path)
        cached = torch.load(cache_path, map_location='cpu')
        features = cached['features']
        paths    = cached['paths']
        return features, paths

    encoder.eval()
    all_feats, all_paths = [], []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            feats = encoder(x)
            feats_gap = feats.mean(dim=[2,3])
            all_feats.append(feats_gap.cpu())
            start_idx = batch_idx * loader.batch_size
            end_idx = start_idx + feats_gap.shape[0]
            for i in range(start_idx, min(end_idx, len(dataset_for_features.samples))):
                all_paths.append(dataset_for_features.samples[i][0])
    features = torch.cat(all_feats, dim=0)
    torch.save({'features': features, 'paths': all_paths}, cache_path)
    print("Cached dataset features to:", cache_path)
    return features, all_paths

dataset_features, dataset_paths = extract_and_cache_dataset_features(encoder_for_bn, dataset_loader, feature_cache_path, device)
dataset_features = dataset_features.to(device)
print("Dataset size:", dataset_features.shape[0], "feature-dim:", dataset_features.shape[1])

# ------------------- Projection Layer if Needed -------------------
with torch.no_grad():
    sample_input = torch.randn(1,3,224,224, device=device)
    sample_feat  = encoder_for_bn(sample_input).mean(dim=[2,3])
feat_dim = sample_feat.shape[1]
dataset_feat_dim = dataset_features.shape[1]

if dataset_feat_dim != feat_dim:
    print(f"Projecting dataset features from {dataset_feat_dim} -> {feat_dim}")
    proj_layer = nn.Linear(dataset_feat_dim, feat_dim, bias=False).to(device)
    dataset_features_proj = proj_layer(dataset_features)
else:
    proj_layer = None
    dataset_features_proj = dataset_features

# ------------------- Helper: get feature vector for reconstructed image -------------------
# ---------- Projection Layer if Needed (create dataset -> encoder projection only) ----------
with torch.no_grad():
    sample_input = torch.randn(1,3,224,224, device=device)
    sample_feat  = encoder_for_bn(sample_input).mean(dim=[2,3])
feat_dim = sample_feat.shape[1]            # encoder output dim (e.g. 2048)
dataset_feat_dim = dataset_features.shape[1]  # cached features dim (e.g. 512)

proj_layer = None
dataset_features_proj = dataset_features

if dataset_feat_dim != feat_dim:
    print(f"Projecting dataset features from {dataset_feat_dim} -> {feat_dim}")
    # projection maps dataset_dim -> encoder_dim
    proj_layer = nn.Linear(dataset_feat_dim, feat_dim, bias=False).to(device)
    # apply projection only to the dataset features (N, dataset_feat_dim) -> (N, feat_dim)
    with torch.no_grad():
        dataset_features_proj = proj_layer(dataset_features)  # (N, feat_dim)
else:
    print("No projection needed: dataset features and encoder features match.")
def get_feature_for_img_tensor(img_tensor, encoder, device, proj_layer=None, dataset_feat_dim=None):
    """
    img_tensor: (B,3,H,W) in [0,1]
    Returns: feats (B, feat_dim) matching encoder output dim.
    If proj_layer is provided, will APPLY IT ONLY if recon feats have dimension == proj_layer.in_features.
    """
    encoder.eval()
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    imgs_resized = torch.zeros((img_tensor.shape[0],3,224,224), device=device)
    for i in range(img_tensor.shape[0]):
        pil = transforms.ToPILImage()(img_tensor[i].cpu())
        proc = preprocess(pil)
        t = transforms.ToTensor()(proc).to(device)
        t = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(t)
        imgs_resized[i] = t

    with torch.no_grad():
        feats = encoder(imgs_resized).mean(dim=[2,3])  # (B, feat_dim_encoder)

        # If proj_layer exists, only use it *if* feats currently match dataset_feat_dim (i.e. need projection).
        # But typically we want: dataset_features_proj has shape (N, feat_dim_encoder), so we should NOT project recon feats.
        # We therefore apply proj_layer only when the recon feats' dim equals proj_layer.in_features.
        if proj_layer is not None:
            # infer in_features from proj_layer.weight shape
            in_features = proj_layer.weight.shape[1]
            if feats.shape[1] == in_features:
                feats = proj_layer(feats)
            else:
                # feats already in encoder dimension — do NOT project
                pass

    return feats

# ------------------- Distance function -------------------
def compute_nearest_dataset_idx(query_feats, dataset_feats, metric="euclidean"):
    if metric=="euclidean":
        dists = torch.cdist(query_feats, dataset_feats)
        idx = torch.argmin(dists, dim=1)
        return idx, dists
    elif metric=="cosine":
        qn = torch.nn.functional.normalize(query_feats, dim=1)
        dn = torch.nn.functional.normalize(dataset_feats, dim=1)
        sims = qn @ dn.t()
        idx = torch.argmax(sims, dim=1)
        return idx, sims
    else:
        raise ValueError("Unsupported metric")

# ------------------- Main Reconstruction Loop -------------------
global_saved = 0
for img_idx in range(n_images):
    print(f"\n--- Generating reconstruction {img_idx+1}/{n_images} ---")

    # Initialize embeddings
    z = torch.randn(z_shape, device=device, requires_grad=True)
    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]
    optimizer_emb = optim.Adam(opt_params, lr=lr)

    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])
        loss = l2_weight*torch.mean(out**2) + tv_weight*tv_loss(out) + z_l2_weight*torch.mean(z**2)
        if bn_weight>0:
            loss += bn_weight*bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

        if step % 50==0 or step==1:
            print(f"step {step}/{num_steps}  loss={loss.item():.6f}")
            save_path = os.path.join(out_dir, f"deepinv_step{step:04d}.png")
            save_and_show(out, save_path, show=False)

    # Final reconstruction
    recon = denorm_if_needed(out.detach().cpu())
    recon_for_enc = recon.to(device)
    recon_feats = get_feature_for_img_tensor(recon_for_enc, encoder_for_bn, device, proj_layer)
    idxs, dists = compute_nearest_dataset_idx(recon_feats, dataset_features_proj, metric=compare_distance)

    for b in range(recon.shape[0]):
        ds_idx = idxs[b].item()
        ds_path = dataset_paths[ds_idx]
        img_ds = Image.open(ds_path).convert('RGB')
        img_ds = display_transform(img_ds)
        img_recon = transforms.ToTensor()(transforms.Resize(224)(transforms.ToPILImage()(recon[b])))

        fig, axs = plt.subplots(1,2,figsize=(6,3))
        axs[0].imshow(img_recon.permute(1,2,0).numpy()); axs[0].set_title("Reconstruction"); axs[0].axis('off')
        axs[1].imshow(img_ds.permute(1,2,0).numpy()); axs[1].set_title(f"Nearest (idx={ds_idx})"); axs[1].axis('off')
        plt.suptitle(f"Recon {img_idx:03d} -- distance: {dists[b,ds_idx].item():.4f}")
        pair_save = os.path.join(save_pairs_dir, f"recon_{img_idx:03d}_nearest_{ds_idx:05d}.png")
        plt.savefig(pair_save, bbox_inches='tight', dpi=150)
        plt.show()
        plt.close(fig)

        vutils.save_image(img_recon, os.path.join(save_pairs_dir, f"recon_{img_idx:03d}.png"))
        vutils.save_image(img_ds,    os.path.join(save_pairs_dir, f"nearest_{img_idx:03d}.png"))
        print(f"Saved pair: {pair_save} -- dataset path: {ds_path}")
        global_saved += 1

print(f"\nDone. Saved {global_saved} recon vs nearest pairs to: {save_pairs_dir}")


# %% [markdown] Cell 16
# #Resnet 18 which was pretrained on cifar10

# %% Cell 17
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def make_resnet_from_user_model():
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=10)

# %% Cell 18
# ---------------- User parameters ----------------
device = device
out_dir = "deepinv_outputs"
os.makedirs(out_dir, exist_ok=True)

large = True  # True -> ResNet50/101/152, False -> ResNet18/34
B = 1
n_images = 5         # total reconstructed images
num_steps = 300
save_every = 100
lr = 0.1
l2_weight = 1.0
tv_weight = 1e-4
bn_weight = 1.0
perc_weight = 0.0
z_l2_weight = 1e-6
optimize_skips = False
use_image_denorm = False

# shapes for z & skips
if large:
    z_shape = (B, 2048, 7, 7)
    skip_shapes = [(B, 1024, 14, 14), (B, 512, 28, 28), (B, 256, 56, 56)]
else:
    z_shape = (B, 512, 7, 7)
    skip_shapes = [(B, 256, 14, 14), (B, 128, 28, 28), (B, 64, 56, 56)]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

# ---------------- Encoder for BN loss ----------------
enc_model_full = make_resnet_from_user_model()
enc_model_full.load_state_dict(torch.load("/content/drive/MyDrive/Pre_trained_model_weights/resnet18_cifar10_trained.pth"))
encoder_for_bn = torch.nn.Sequential(*list(enc_model_full.children())[:-2]).to(device).eval()
for p in encoder_for_bn.parameters():
    p.requires_grad = False

# ---------------- Helper functions ----------------
def tv_loss(img):
    dy = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    dx = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return dx + dy

def bn_stat_loss_from_encoder(decoder_out, encoder_model, device):
    bn_features = []
    def hook_fn(module, inp, out):
        bn_features.append(out)
    hooks = [m.register_forward_hook(hook_fn) for m in encoder_model.modules() if isinstance(m, nn.BatchNorm2d)]
    _ = encoder_model(decoder_out)
    losses = []
    bn_modules = [m for m in encoder_model.modules() if isinstance(m, nn.BatchNorm2d)]
    for out_bn, module in zip(bn_features, bn_modules):
        mean_batch = out_bn.mean(dim=[0,2,3])
        var_batch  = out_bn.var(dim=[0,2,3], unbiased=False)
        losses.append(((mean_batch - module.running_mean.to(device))**2).mean())
        losses.append(((var_batch  - module.running_var.to(device))**2).mean())
    for h in hooks:
        h.remove()
    return sum(losses)

def denorm_if_needed(img_tensor):
    if use_image_denorm:
        return ((img_tensor * IMAGENET_STD) + IMAGENET_MEAN).clamp(0,1)
    else:
        return img_tensor.clamp(0,1)

# ---------------- Main generation loop ----------------
for img_idx in range(n_images):
    print(f"\n=== Generating image {img_idx+1}/{n_images} ===")
    z = torch.randn(z_shape, device=device, requires_grad=True)

    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]

    optimizer_emb = torch.optim.Adam(opt_params, lr=lr)

    # Optimize latent embeddings
    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])

        loss_l2 = torch.mean(out**2)
        loss = l2_weight * loss_l2 + tv_weight * tv_loss(out) + z_l2_weight * torch.mean(z**2)
        if bn_weight > 0:
            loss += bn_weight * bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

        if step % 50 == 0:
            print(f"step {step}/{num_steps}  loss={loss.item():.6f}")

    # save final output
    with torch.no_grad():
        final_img = denorm_if_needed(out)
        save_path = os.path.join(out_dir, f"deepinv_img_{img_idx:03d}.png")
        vutils.save_image(final_img, save_path)
        print(f"✅ Saved: {save_path}")

print(f"\nAll {n_images} images saved in: {out_dir}")
# ================== Nearest-dataset-image comparison + display ==================
import os, torch, math, numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils

# ---------- User settings ----------
# ---------- User settings (CIFAR-10) ----------
# Where to download/store CIFAR-10
cifar_root = "./data"            # change if you want another path
dataset_batch = 64
feature_cache_path = os.path.join(out_dir, "dataset_features.pt")  # cached features
compare_distance = "cosine"  # "euclidean" or "cosine"
save_pairs_dir = os.path.join(out_dir, "recon_vs_nearest")
os.makedirs(save_pairs_dir, exist_ok=True)

# ---------- Helper: get feature vector for reconstructed image ----------
def get_feature_for_img_tensor(img_tensor, encoder, device):
    encoder.eval()
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    imgs_resized = torch.zeros((img_tensor.shape[0], 3, 224, 224), device=device)
    for i in range(img_tensor.shape[0]):
        pil = transforms.ToPILImage()(img_tensor[i].cpu())
        proc = preprocess(pil)
        t = transforms.ToTensor()(proc).to(device)
        t = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(t)
        imgs_resized[i] = t
    with torch.no_grad():
        feats = encoder(imgs_resized)                # (B, C_encoder, Hf, Wf)
        feats_gap = feats.mean(dim=[2,3]) # (B, C_encoder)

    return feats_gap

# ---------- Helper: compute nearest dataset index ----------
def compute_nearest_dataset_idx(query_feats, dataset_feats, metric="cosine"):
    if metric == "euclidean" and query_feats.shape[1] != dataset_feats.shape[1]:
        print("Warning: feature dims mismatch, using cosine instead")
        metric = "cosine"

    if metric == "euclidean":
        dists = torch.cdist(query_feats, dataset_feats)
        idx = torch.argmin(dists, dim=1)
        return idx, dists
    elif metric == "cosine":
        qn = torch.nn.functional.normalize(query_feats, dim=1)
        dn = torch.nn.functional.normalize(dataset_feats, dim=1)
        sims = qn @ dn.t()
        idx = torch.argmax(sims, dim=1)
        return idx, sims
    else:
        raise ValueError("Unsupported metric")

# ---------- Helper: extract / cache dataset features ----------
def extract_and_cache_dataset_features(encoder, loader, cache_path, device, dataset_obj):
    """
    Extracts global-average-pooled features for the dataset in `loader` using `encoder`,
    and caches them along with corresponding dataset paths.
    ✅ Fixed: avoids index overshoot by recording exact dataset indices per batch.
    """
    if os.path.exists(cache_path):
        print("Loading cached dataset features:", cache_path)
        cached = torch.load(cache_path, map_location='cpu')
        return cached['features'], cached['all_paths']

    encoder.eval()
    all_feats = []
    all_paths = []

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            feats = encoder(x)
            feats_gap = feats.mean(dim=[2, 3])  # global average pooling
            all_feats.append(feats_gap.cpu())

            # Record exact dataset indices per batch (no overshoot)
            start_idx = batch_idx * loader.batch_size
            for i_in_batch in range(feats_gap.shape[0]):
                dataset_index = start_idx + i_in_batch
                if dataset_index >= len(dataset_obj):
                    break
                all_paths.append(dataset_index)

    features = torch.cat(all_feats, dim=0)

    # Safety trim (in case of any mismatch)
    n = min(features.shape[0], len(all_paths))
    features = features[:n]
    all_paths = all_paths[:n]

    torch.save({'features': features, 'all_paths': all_paths}, cache_path)
    print("✅ Cached dataset features to:", cache_path)
    return features, all_paths



# ---------- Transforms ----------
preprocess_for_encoder = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

display_transform = transforms.Compose([
    transforms.ToTensor(),
])

# ---------- CIFAR-10 datasets & loaders ----------
dataset_for_features = datasets.CIFAR10(root=cifar_root, train=False, transform=preprocess_for_encoder, download=True)
dataset_for_display  = datasets.CIFAR10(root=cifar_root, train=False, transform=display_transform, download=False)

dataset_loader = DataLoader(dataset_for_features, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)
display_loader = DataLoader(dataset_for_display, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)

# ---------- Step 1: Extract / cache dataset features ----------
import os
if os.path.exists(feature_cache_path):
    os.remove(feature_cache_path)
    print("Old cache deleted, regenerating...")

dataset_features, dataset_paths = extract_and_cache_dataset_features(
    encoder_for_bn, dataset_loader, feature_cache_path, device, dataset_for_features
)
dataset_features = dataset_features.to(device)
dataset_feat_dim = dataset_features.shape[1]

# ---------- Optional: projection if feature dims mismatch ----------
with torch.no_grad():
    sample_input = torch.randn(1,3,32,32, device=device)
    sample_feat  = encoder_for_bn(sample_input)
encoder_feat_dim = sample_feat.mean(dim=[2,3]).shape[1]


proj_layer = None
dataset_features_proj = dataset_features

if dataset_feat_dim != encoder_feat_dim:
    print(f"Projecting dataset features from {dataset_feat_dim} -> {encoder_feat_dim}")
    # Create a projection layer from dataset_feat_dim to encoder_feat_dim
    proj_layer = nn.Linear(dataset_feat_dim, encoder_feat_dim, bias=False).to(device)
    with torch.no_grad():
         # Project dataset features to the encoder's feature dimension
         dataset_features_proj = proj_layer(dataset_features)
else:
    print("No projection needed: dataset features and encoder features match.")


# ---------- Step 2: Main reconstruction loop ----------
n_images_to_generate = n_images
global_saved = 0

for img_idx in range(n_images_to_generate):
    print(f"\n--- Generating reconstruction {img_idx+1}/{n_images_to_generate} ---")

    # Initialize embeddings
    z = torch.randn(z_shape, device=device, requires_grad=True)
    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]

    optimizer_emb = torch.optim.Adam(opt_params, lr=lr)

    # Optimize latent embeddings
    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])
        loss = (l2_weight * torch.mean(out**2) +
                tv_weight * tv_loss(out) +
                z_l2_weight * torch.mean(z**2))
        if bn_weight > 0:
            loss += bn_weight * bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

    recon = denorm_if_needed(out.detach().cpu())

    # ---------- Compute features of reconstruction ----------
    recon_for_enc = recon.to(device)
    recon_feats = get_feature_for_img_tensor(recon_for_enc, encoder_for_bn, device)

    # No projection needed for recon_feats, as dataset_features_proj is already
    # projected to the encoder_feat_dim if necessary.


    # ---------- Find nearest CIFAR-10 image ----------
    # Use the recon features directly for comparison with projected dataset features
    idxs, sims = compute_nearest_dataset_idx(recon_feats, dataset_features_proj, metric="cosine")
    nearest_idx = idxs[0].item()

    # Get nearest CIFAR-10 image
    nearest_img, _ = dataset_for_display[nearest_idx]
    nearest_img_resized = transforms.Resize(224)(transforms.ToPILImage()(nearest_img))
    recon_img_resized = transforms.Resize(224)(transforms.ToPILImage()(recon[0]))

    # ---------- Display side by side ----------
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(recon_img_resized); axs[0].set_title("Reconstruction"); axs[0].axis('off')
    axs[1].imshow(nearest_img_resized); axs[1].set_title(f"Nearest CIFAR-10 (idx={nearest_idx})"); axs[1].axis('off')
    plt.suptitle(f"Recon {img_idx:03d}  -- cosine similarity: {sims[0, nearest_idx]:.4f}")
    pair_save = os.path.join(save_pairs_dir, f"recon_{img_idx:03d}_nearest.png")
    plt.savefig(pair_save, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)

    # Save individual images
    vutils.save_image(transforms.ToTensor()(recon_img_resized), os.path.join(save_pairs_dir, f"recon_{img_idx:03d}.png"))
    vutils.save_image(transforms.ToTensor()(nearest_img_resized), os.path.join(save_pairs_dir, f"nearest_{img_idx:03d}.png"))
    print(f"Saved pair: {pair_save}  -- dataset idx: {nearest_idx}")
    global_saved += 1

print(f"\nDone. Saved {global_saved} recon vs nearest CIFAR-10 pairs to: {save_pairs_dir}")

# %% [markdown] Cell 19
# #Resnet 101 which was pretrained on cifar10

# %% Cell 20
import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetCIFAR101(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3,4,23,3], num_classes=10):
        super(ResNetCIFAR101, self).__init__()
        self.in_channels = 64
        # CIFAR-style stem: 3x3 conv, stride=1, no maxpool
        self.conv1 = conv3x3(3, 64, stride=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        # residual layers (standard widths but CIFAR stem)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x

def make_resnet101_cifar(num_classes=10):
    return ResNetCIFAR101(num_classes=num_classes)


# %% Cell 21
# ---------------- ResNet-101 (CIFAR-style) test + nearest-cosine comparison ----------------
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ---------- Assumptions: you already have `device` and `decoder` defined ----------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# decoder = <your decoder instance loaded with weights>
# ---------- Build encoder_for_bn from ResNet-101 CIFAR and load checkpoint ----------
resnet101_model = make_resnet101_cifar(num_classes=10).to(device)
ckpt_path = "/content/drive/MyDrive/Pre_trained_model_weights/resnet101_cifar10_final_state_dict.pth"   # <-- change to your checkpoint filepath

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

sd = torch.load(ckpt_path, map_location='cpu')
# handle common checkpoint wrappers
if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
    sd = sd['state_dict']

try:
    resnet101_model.load_state_dict(sd, strict=True)
    print("ResNet-101 (CIFAR) checkpoint loaded (strict=True).")
except Exception as e:
    print("Strict load failed:", e)
    print("Retrying with strict=False (will ignore missing/unexpected keys).")
    resnet101_model.load_state_dict(sd, strict=False)

# Build encoder_for_bn = everything up to (and excluding) avgpool + fc
encoder_for_bn = torch.nn.Sequential(*list(resnet101_model.children())[:-2]).to(device).eval()
for p in encoder_for_bn.parameters():
    p.requires_grad = False

# ---------------- User parameters (kept same as your earlier pipeline) ----------------
out_dir = "deepinv_outputs_resnet101_cifar"
os.makedirs(out_dir, exist_ok=True)

large = True  # True -> ResNet50/101/152 style z dims (we keep your existing shapes)
B = 1
n_images = 5         # total reconstructed images
num_steps = 300
save_every = 100
lr = 0.1
l2_weight = 1.0
tv_weight = 1e-4
bn_weight = 1.0
perc_weight = 0.0
z_l2_weight = 1e-6
optimize_skips = False
use_image_denorm = False

# shapes for z & skips
if large:
    z_shape = (B, 2048, 7, 7)    # matches ResNet101 CIFAR final channels = 512*4 = 2048
    skip_shapes = [(B, 1024, 14, 14), (B, 512, 28, 28), (B, 256, 56, 56)]
else:
    z_shape = (B, 512, 7, 7)
    skip_shapes = [(B, 256, 14, 14), (B, 128, 28, 28), (B, 64, 56, 56)]

# CIFAR normalization (used for dataset feature extraction and encoding)
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)
CIFAR_STD  = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1,3,1,1)

def tv_loss(img):
    dy = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    dx = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return dx + dy

def bn_stat_loss_from_encoder(decoder_out, encoder_model, device):
    # register forward hooks on BN layers of encoder_model and compare batch stats to running stats
    bn_features = []
    def hook_fn(module, inp, out):
        bn_features.append(out)
    bn_modules = [m for m in encoder_model.modules() if isinstance(m, nn.BatchNorm2d)]
    hooks = [m.register_forward_hook(hook_fn) for m in bn_modules]
    # run forward
    _ = encoder_model(decoder_out)
    losses = []
    # zip outputs with bn_modules (should match in count & order for models built consistently)
    for out_bn, module in zip(bn_features, bn_modules):
        mean_batch = out_bn.mean(dim=[0,2,3])
        var_batch  = out_bn.var(dim=[0,2,3], unbiased=False)
        losses.append(((mean_batch - module.running_mean.to(device))**2).mean())
        losses.append(((var_batch  - module.running_var.to(device))**2).mean())
    for h in hooks:
        h.remove()
    return sum(losses)

def denorm_if_needed(img_tensor):
    # decoder outputs are expected in [0,1] range (optionally normalized during training)
    if use_image_denorm:
        return ((img_tensor * CIFAR_STD) + CIFAR_MEAN).clamp(0,1)
    else:
        return img_tensor.clamp(0,1)

# ---------------- Main generation loop (same logic as your ResNet-18 pipeline) ----------------
for img_idx in range(n_images):
    print(f"\n=== Generating image {img_idx+1}/{n_images} ===")
    z = torch.randn(z_shape, device=device, requires_grad=True)

    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]

    optimizer_emb = torch.optim.Adam(opt_params, lr=lr)

    # Optimize latent embeddings
    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])            # out in (B,3,H,W)
        loss = (l2_weight * torch.mean(out**2) +
                tv_weight * tv_loss(out) +
                z_l2_weight * torch.mean(z**2))
        if bn_weight > 0:
            loss += bn_weight * bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

        if step % 50 == 0:
            print(f"step {step}/{num_steps}  loss={loss.item():.6f}")

    # save final output
    with torch.no_grad():
        final_img = denorm_if_needed(out)
        save_path = os.path.join(out_dir, f"deepinv_resnet101_cifar_img_{img_idx:03d}.png")
        vutils.save_image(final_img, save_path)
        print(f"✅ Saved: {save_path}")

print(f"\nAll {n_images} images saved in: {out_dir}")

# ================== Nearest-dataset-image comparison + display ==================
# ---------- User settings ----------
cifar_root = "./data"            # change if you want another path
dataset_batch = 64
feature_cache_path = os.path.join(out_dir, "dataset_features_resnet101_cifar.pt")
compare_distance = "cosine"  # "euclidean" or "cosine"
save_pairs_dir = os.path.join(out_dir, "recon_vs_nearest")
os.makedirs(save_pairs_dir, exist_ok=True)

# ---------- Helper: get feature vector for reconstructed image ----------
def get_feature_for_img_tensor_cifar(img_tensor, encoder, device):
    # img_tensor: (B,3,H,W) in [0,1] (already denormed/clamped)
    encoder.eval()
    # Ensure input size 32x32 (CIFAR)
    imgs_proc = torch.zeros((img_tensor.shape[0], 3, 32, 32), device=device)
    for i in range(img_tensor.shape[0]):
        pil = transforms.ToPILImage()(img_tensor[i].cpu())
        proc = pil.resize((32,32), Image.BILINEAR)
        t = transforms.ToTensor()(proc).to(device)
        t = transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010])(t)
        imgs_proc[i] = t
    with torch.no_grad():
        feats = encoder(imgs_proc)                # (B, C_encoder, Hf, Wf)
        feats_gap = feats.mean(dim=[2,3]) # (B, C_encoder)
    return feats_gap

# ---------- Helper: compute nearest dataset index ----------
def compute_nearest_dataset_idx(query_feats, dataset_feats, metric="cosine"):
    if metric == "euclidean" and query_feats.shape[1] != dataset_feats.shape[1]:
        print("Warning: feature dims mismatch, using cosine instead")
        metric = "cosine"

    if metric == "euclidean":
        dists = torch.cdist(query_feats, dataset_feats)
        idx = torch.argmin(dists, dim=1)
        return idx, dists
    elif metric == "cosine":
        qn = torch.nn.functional.normalize(query_feats, dim=1)
        dn = torch.nn.functional.normalize(dataset_feats, dim=1)
        sims = qn @ dn.t()   # similarity matrix
        idx = torch.argmax(sims, dim=1)
        return idx, sims
    else:
        raise ValueError("Unsupported metric")

# ---------- Helper: extract / cache dataset features ----------
# ---------- Helper: extract / cache dataset features ----------
def extract_and_cache_dataset_features(encoder, loader, cache_path, device, dataset_for_features):
    if os.path.exists(cache_path):
        print("Loading cached dataset features:", cache_path)
        cached = torch.load(cache_path, map_location='cpu')
        return cached['features'], cached['paths']

    encoder.eval()
    all_feats, all_paths = [], []

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)   # x already normalized for CIFAR in loader
            feats = encoder(x)                # (B, C, Hf, Wf)
            feats_gap = feats.mean(dim=[2,3]) # (B, C)
            all_feats.append(feats_gap.cpu())

            # store dataset indices as "paths"
            start_idx = batch_idx * loader.batch_size
            end_idx   = start_idx + feats_gap.shape[0]
            for i in range(start_idx, min(end_idx, len(dataset_for_features))):
                all_paths.append(i)

    features = torch.cat(all_feats, dim=0)
    torch.save({'features': features, 'paths': all_paths}, cache_path)
    print("Cached dataset features to:", cache_path)
    return features, all_paths

# ---------- Transforms ----------
preprocess_for_encoder = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

display_transform = transforms.Compose([
    transforms.ToTensor(),
])

# ---------- CIFAR-10 datasets & loaders ----------
dataset_for_features = datasets.CIFAR10(root=cifar_root, train=False, transform=preprocess_for_encoder, download=True)
dataset_for_display  = datasets.CIFAR10(root=cifar_root, train=False, transform=display_transform, download=False)

dataset_loader = DataLoader(dataset_for_features, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)
display_loader = DataLoader(dataset_for_display, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)

# ---------- Step 1: Extract / cache dataset features ----------
dataset_features, dataset_paths = extract_and_cache_dataset_features(encoder_for_bn, dataset_loader, feature_cache_path, device, dataset_for_features)
dataset_features = dataset_features.to(device)
dataset_feat_dim = dataset_features.shape[1]

# ---------- Step 2: Main reconstruction -> nearest neighbor comparison ----------
n_images_to_generate = n_images
global_saved = 0

# If your encoder_for_bn produces feat dim different than cached features, you could optionally project.
# Here we assume they match (both produced by encoder_for_bn).
for img_idx in range(n_images_to_generate):
    print(f"\n--- Producing nearest-match display for reconstruction {img_idx+1}/{n_images_to_generate} ---")

    # Recreate same reconstruction process (to ensure recon is available)
    z = torch.randn(z_shape, device=device, requires_grad=True)
    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]

    optimizer_emb = torch.optim.Adam(opt_params, lr=lr)
    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])
        loss = (l2_weight * torch.mean(out**2) +
                tv_weight * tv_loss(out) +
                z_l2_weight * torch.mean(z**2))
        if bn_weight > 0:
            loss += bn_weight * bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

    recon = denorm_if_needed(out.detach().cpu())    # (B,3,H,W) cpu

    # Compute features of reconstruction (using CIFAR-sized preprocessing)
    recon_for_enc = recon.to(device)
    recon_feats = get_feature_for_img_tensor_cifar(recon_for_enc, encoder_for_bn, device)  # (B, C)

    # Find nearest CIFAR-10 image (cosine)
    idxs, sims = compute_nearest_dataset_idx(recon_feats, dataset_features, metric="cosine")
    nearest_idx = idxs[0].item()

    # Get nearest CIFAR-10 image for display
    nearest_img, _ = dataset_for_display[nearest_idx]   # (3,32,32) tensor in [0,1]
    # Resize both to 32x32 -> for display we keep 32 but can upscale for plotting clarity
    recon_img_display = transforms.ToPILImage()(recon[0])
    nearest_img_display = transforms.ToPILImage()(nearest_img)

    # ---------- Display side by side ----------
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(recon_img_display); axs[0].set_title("Reconstruction"); axs[0].axis('off')
    axs[1].imshow(nearest_img_display); axs[1].set_title(f"Nearest CIFAR-10 (idx={nearest_idx})"); axs[1].axis('off')
    sim_val = float(sims[0, nearest_idx].cpu()) if compare_distance == "cosine" else float(sims[0, nearest_idx].cpu())
    plt.suptitle(f"Recon {img_idx:03d}  -- cosine similarity: {sim_val:.4f}")
    pair_save = os.path.join(save_pairs_dir, f"recon_{img_idx:03d}_nearest.png")
    plt.savefig(pair_save, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)

    # Save individual images (as in your pipeline)
    vutils.save_image(transforms.ToTensor()(recon_img_display), os.path.join(save_pairs_dir, f"recon_{img_idx:03d}.png"))
    vutils.save_image(transforms.ToTensor()(nearest_img_display), os.path.join(save_pairs_dir, f"nearest_{img_idx:03d}.png"))
    print(f"Saved pair: {pair_save}  -- dataset idx: {nearest_idx}")
    global_saved += 1

print(f"\nDone. Saved {global_saved} recon vs nearest CIFAR-10 pairs to: {save_pairs_dir}")


# %% [markdown] Cell 22
# #Resnet50 which was pretrained on cifar10

# %% Cell 23
# resnet50_deepinv_cifar.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ----------------- Model definitions (same as your ResNetCIFAR101) -----------------
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetCIFAR101(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3,4,23,3], num_classes=10):
        super(ResNetCIFAR101, self).__init__()
        self.in_channels = 64
        # CIFAR-style stem: 3x3 conv, stride=1, no maxpool
        self.conv1 = conv3x3(3, 64, stride=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        # residual layers (standard widths but CIFAR stem)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x

def make_resnet50_cifar(num_classes=10):
    return ResNetCIFAR101(block=Bottleneck, layers=[3,4,6,3], num_classes=num_classes)

# ----------------- Environment assumptions (uncomment / set as needed) -----------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# decoder = <your decoder instance loaded with weights>

# ----------------- Build model and load checkpoint -----------------
resnet50_model = make_resnet50_cifar(num_classes=10).to(device)
ckpt_path = "/content/drive/MyDrive/Pre_trained_model_weights/resnet50_cifar10 .pth"   # <-- change to your checkpoint filepath

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

sd = torch.load(ckpt_path, map_location='cpu')
if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
    sd = sd['state_dict']

try:
    resnet50_model.load_state_dict(sd, strict=True)
    print("ResNet-50 (CIFAR) checkpoint loaded (strict=True).")
except Exception as e:
    print("Strict load failed:", e)
    print("Retrying with strict=False (will ignore missing/unexpected keys).")
    resnet50_model.load_state_dict(sd, strict=False)

# Build encoder_for_bn = everything up to (and excluding) avgpool + fc
encoder_for_bn = torch.nn.Sequential(*list(resnet50_model.children())[:-2]).to(device).eval()
for p in encoder_for_bn.parameters():
    p.requires_grad = False

# ----------------- Reconstruction / deepinv params (same as your pipeline) -----------------
out_dir = "deepinv_outputs_resnet50_cifar"
os.makedirs(out_dir, exist_ok=True)

large = True
B = 1
n_images = 5
num_steps = 300
save_every = 100
lr = 0.1
l2_weight = 1.0
tv_weight = 1e-4
bn_weight = 1.0
perc_weight = 0.0
z_l2_weight = 1e-6
optimize_skips = False
use_image_denorm = False

if large:
    z_shape = (B, 2048, 7, 7)
    skip_shapes = [(B, 1024, 14, 14), (B, 512, 28, 28), (B, 256, 56, 56)]
else:
    z_shape = (B, 512, 7, 7)
    skip_shapes = [(B, 256, 14, 14), (B, 128, 28, 28), (B, 64, 56, 56)]

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)
CIFAR_STD  = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1,3,1,1)

def tv_loss(img):
    dy = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    dx = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return dx + dy

def bn_stat_loss_from_encoder(decoder_out, encoder_model, device):
    bn_features = []
    def hook_fn(module, inp, out):
        bn_features.append(out)
    bn_modules = [m for m in encoder_model.modules() if isinstance(m, nn.BatchNorm2d)]
    hooks = [m.register_forward_hook(hook_fn) for m in bn_modules]
    _ = encoder_model(decoder_out)
    losses = []
    for out_bn, module in zip(bn_features, bn_modules):
        mean_batch = out_bn.mean(dim=[0,2,3])
        var_batch  = out_bn.var(dim=[0,2,3], unbiased=False)
        losses.append(((mean_batch - module.running_mean.to(device))**2).mean())
        losses.append(((var_batch  - module.running_var.to(device))**2).mean())
    for h in hooks:
        h.remove()
    return sum(losses)

def denorm_if_needed(img_tensor):
    if use_image_denorm:
        return ((img_tensor * CIFAR_STD) + CIFAR_MEAN).clamp(0,1)
    else:
        return img_tensor.clamp(0,1)

# ----------------- Main generation loop -----------------
for img_idx in range(n_images):
    print(f"\n=== Generating image {img_idx+1}/{n_images} ===")
    z = torch.randn(z_shape, device=device, requires_grad=True)

    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]

    optimizer_emb = torch.optim.Adam(opt_params, lr=lr)

    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])            # out in (B,3,H,W)
        loss = (l2_weight * torch.mean(out**2) +
                tv_weight * tv_loss(out) +
                z_l2_weight * torch.mean(z**2))
        if bn_weight > 0:
            loss += bn_weight * bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

        if step % 50 == 0:
            print(f"step {step}/{num_steps}  loss={loss.item():.6f}")

    with torch.no_grad():
        final_img = denorm_if_needed(out)
        save_path = os.path.join(out_dir, f"deepinv_resnet50_cifar_img_{img_idx:03d}.png")
        vutils.save_image(final_img, save_path)
        print(f"✅ Saved: {save_path}")

print(f"\nAll {n_images} images saved in: {out_dir}")

# ----------------- Nearest-dataset-image comparison + display -----------------
cifar_root = "./data"
dataset_batch = 64
feature_cache_path = os.path.join(out_dir, "dataset_features_resnet50_cifar.pt")
compare_distance = "cosine"
save_pairs_dir = os.path.join(out_dir, "recon_vs_nearest")
os.makedirs(save_pairs_dir, exist_ok=True)

def get_feature_for_img_tensor_cifar(img_tensor, encoder, device):
    encoder.eval()
    imgs_proc = torch.zeros((img_tensor.shape[0], 3, 32, 32), device=device)
    for i in range(img_tensor.shape[0]):
        pil = transforms.ToPILImage()(img_tensor[i].cpu())
        proc = pil.resize((32,32), Image.BILINEAR)
        t = transforms.ToTensor()(proc).to(device)
        t = transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010])(t)
        imgs_proc[i] = t
    with torch.no_grad():
        feats = encoder(imgs_proc)
        feats_gap = feats.mean(dim=[2,3])
    return feats_gap

def compute_nearest_dataset_idx(query_feats, dataset_feats, metric="cosine"):
    if metric == "euclidean" and query_feats.shape[1] != dataset_feats.shape[1]:
        print("Warning: feature dims mismatch, using cosine instead")
        metric = "cosine"

    if metric == "euclidean":
        dists = torch.cdist(query_feats, dataset_feats)
        idx = torch.argmin(dists, dim=1)
        return idx, dists
    elif metric == "cosine":
        qn = torch.nn.functional.normalize(query_feats, dim=1)
        dn = torch.nn.functional.normalize(dataset_feats, dim=1)
        sims = qn @ dn.t()
        idx = torch.argmax(sims, dim=1)
        return idx, sims
    else:
        raise ValueError("Unsupported metric")

def extract_and_cache_dataset_features(encoder, loader, cache_path, device, dataset_for_features):
    if os.path.exists(cache_path):
        print("Loading cached dataset features:", cache_path)
        cached = torch.load(cache_path, map_location='cpu')
        return cached['features'], cached['paths']

    encoder.eval()
    all_feats, all_paths = [], []

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            feats = encoder(x)
            feats_gap = feats.mean(dim=[2,3])
            all_feats.append(feats_gap.cpu())
            start_idx = batch_idx * loader.batch_size
            end_idx   = start_idx + feats_gap.shape[0]
            for i in range(start_idx, min(end_idx, len(dataset_for_features))):
                all_paths.append(i)

    features = torch.cat(all_feats, dim=0)
    torch.save({'features': features, 'paths': all_paths}, cache_path)
    print("Cached dataset features to:", cache_path)
    return features, all_paths

preprocess_for_encoder = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

display_transform = transforms.Compose([transforms.ToTensor()])

dataset_for_features = datasets.CIFAR10(root=cifar_root, train=False, transform=preprocess_for_encoder, download=True)
dataset_for_display  = datasets.CIFAR10(root=cifar_root, train=False, transform=display_transform, download=False)

dataset_loader = DataLoader(dataset_for_features, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)
display_loader = DataLoader(dataset_for_display, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)

dataset_features, dataset_paths = extract_and_cache_dataset_features(encoder_for_bn, dataset_loader, feature_cache_path, device, dataset_for_features)
dataset_features = dataset_features.to(device)

n_images_to_generate = n_images
global_saved = 0

for img_idx in range(n_images_to_generate):
    print(f"\n--- Producing nearest-match display for reconstruction {img_idx+1}/{n_images_to_generate} ---")
    z = torch.randn(z_shape, device=device, requires_grad=True)
    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]

    optimizer_emb = torch.optim.Adam(opt_params, lr=lr)
    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])
        loss = (l2_weight * torch.mean(out**2) +
                tv_weight * tv_loss(out) +
                z_l2_weight * torch.mean(z**2))
        if bn_weight > 0:
            loss += bn_weight * bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

    recon = denorm_if_needed(out.detach().cpu())
    recon_for_enc = recon.to(device)
    recon_feats = get_feature_for_img_tensor_cifar(recon_for_enc, encoder_for_bn, device)
    idxs, sims = compute_nearest_dataset_idx(recon_feats, dataset_features, metric="cosine")
    nearest_idx = idxs[0].item()
    nearest_img, _ = dataset_for_display[nearest_idx]
    recon_img_display = transforms.ToPILImage()(recon[0])
    nearest_img_display = transforms.ToPILImage()(nearest_img)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(recon_img_display); axs[0].set_title("Reconstruction"); axs[0].axis('off')
    axs[1].imshow(nearest_img_display); axs[1].set_title(f"Nearest CIFAR-10 (idx={nearest_idx})"); axs[1].axis('off')
    sim_val = float(sims[0, nearest_idx].cpu()) if compare_distance == "cosine" else float(sims[0, nearest_idx].cpu())
    plt.suptitle(f"Recon {img_idx:03d}  -- cosine similarity: {sim_val:.4f}")
    pair_save = os.path.join(save_pairs_dir, f"recon_{img_idx:03d}_nearest.png")
    plt.savefig(pair_save, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)

    vutils.save_image(transforms.ToTensor()(recon_img_display), os.path.join(save_pairs_dir, f"recon_{img_idx:03d}.png"))
    vutils.save_image(transforms.ToTensor()(nearest_img_display), os.path.join(save_pairs_dir, f"nearest_{img_idx:03d}.png"))
    print(f"Saved pair: {pair_save}  -- dataset idx: {nearest_idx}")
    global_saved += 1

print(f"\nDone. Saved {global_saved} recon vs nearest CIFAR-10 pairs to: {save_pairs_dir}")

# %% [markdown] Cell 24
# #Resnet152 which was pretrained on cifar10

# %% Cell 25
# resnet152_deepinv_cifar.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ----------------- Model definitions (same as your ResNetCIFAR101) -----------------
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetCIFAR101(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3,4,23,3], num_classes=10):
        super(ResNetCIFAR101, self).__init__()
        self.in_channels = 64
        # CIFAR-style stem: 3x3 conv, stride=1, no maxpool
        self.conv1 = conv3x3(3, 64, stride=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        # residual layers (standard widths but CIFAR stem)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x

def make_resnet152_cifar(num_classes=10):
    return ResNetCIFAR101(block=Bottleneck, layers=[3,8,36,3], num_classes=num_classes)

# ----------------- Environment assumptions (uncomment / set as needed) -----------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# decoder = <your decoder instance loaded with weights>

# ----------------- Build model and load checkpoint -----------------
resnet152_model = make_resnet152_cifar(num_classes=10).to(device)
ckpt_path = "/content/drive/MyDrive/Pre_trained_model_weights/resnet152_cifar10.pth"   # <-- change to your checkpoint filepath

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

sd = torch.load(ckpt_path, map_location='cpu')
if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
    sd = sd['state_dict']

try:
    resnet152_model.load_state_dict(sd, strict=True)
    print("ResNet-152 (CIFAR) checkpoint loaded (strict=True).")
except Exception as e:
    print("Strict load failed:", e)
    print("Retrying with strict=False (will ignore missing/unexpected keys).")
    resnet152_model.load_state_dict(sd, strict=False)

# Build encoder_for_bn = everything up to (and excluding) avgpool + fc
encoder_for_bn = torch.nn.Sequential(*list(resnet152_model.children())[:-2]).to(device).eval()
for p in encoder_for_bn.parameters():
    p.requires_grad = False

# ----------------- Reconstruction / deepinv params (same as your pipeline) -----------------
out_dir = "deepinv_outputs_resnet152_cifar"
os.makedirs(out_dir, exist_ok=True)

large = True
B = 1
n_images = 5
num_steps = 300
save_every = 100
lr = 0.1
l2_weight = 1.0
tv_weight = 1e-4
bn_weight = 1.0
perc_weight = 0.0
z_l2_weight = 1e-6
optimize_skips = False
use_image_denorm = False

if large:
    z_shape = (B, 2048, 7, 7)
    skip_shapes = [(B, 1024, 14, 14), (B, 512, 28, 28), (B, 256, 56, 56)]
else:
    z_shape = (B, 512, 7, 7)
    skip_shapes = [(B, 256, 14, 14), (B, 128, 28, 28), (B, 64, 56, 56)]

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)
CIFAR_STD  = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1,3,1,1)

def tv_loss(img):
    dy = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    dx = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return dx + dy

def bn_stat_loss_from_encoder(decoder_out, encoder_model, device):
    """
    Compute BN running-stat matching loss by feeding the encoder the same
    normalized images it was trained on. Returns a scalar loss (torch.Tensor)
    which gradients can flow back into decoder_out.
    """
    # 1) Ensure decoder_out is in [0,1] (heuristic: map from [-1,1] if necessary)
    proc = decoder_out
    if proc.min() < -0.5:
        # likely tanh output in [-1,1]
        proc = (proc + 1.0) / 2.0
    proc = proc.clamp(0.0, 1.0)

    # 2) Normalize using CIFAR stats (move to same device)
    mean = CIFAR_MEAN.to(device)
    std  = CIFAR_STD.to(device)
    normalized = (proc - mean) / std

    # 3) Collect (module, output) pairs with hooks so ordering & pairing is exact
    bn_pairs = []
    def make_hook(mod):
        def hook(module, inp, out):
            # capture the module instance and its output tensor (not detached)
            bn_pairs.append((module, out))
        return hook

    bn_modules = [m for m in encoder_model.modules() if isinstance(m, nn.BatchNorm2d)]
    hooks = [m.register_forward_hook(make_hook(m)) for m in bn_modules]

    # 4) Run encoder in eval mode but without disabling grad, so gradients flow to decoder
    encoder_model.eval()
    _ = encoder_model(normalized)  # gradients will propagate back to `normalized`/decoder_out

    # 5) Compute per-module mean/var matching loss (safe to handle channel mismatches)
    losses = []
    mismatch_warned = False
    for module, out_bn in bn_pairs:
        # out_bn shape: (B, C_out, H, W)
        mean_batch = out_bn.mean(dim=[0, 2, 3])
        var_batch  = out_bn.var(dim=[0, 2, 3], unbiased=False)

        running_mean = module.running_mean.to(out_bn.device)
        running_var  = module.running_var.to(out_bn.device)

        if running_mean.shape[0] != mean_batch.shape[0]:
            # Very rare: sizes don't match. Slice to min channels to avoid crash and warn once.
            min_c = min(running_mean.shape[0], mean_batch.shape[0])
            if not mismatch_warned:
                print(f"⚠️ BN-channel mismatch encountered (module {module}): "
                      f"running_mean has {running_mean.shape[0]} ch, output has {mean_batch.shape[0]} ch. "
                      f"Slicing to {min_c} channels for loss computation.")
                mismatch_warned = True
            mean_batch = mean_batch[:min_c]
            var_batch  = var_batch[:min_c]
            running_mean = running_mean[:min_c]
            running_var  = running_var[:min_c]

        losses.append(((mean_batch - running_mean)**2).mean())
        losses.append(((var_batch  - running_var )**2).mean())

    # 6) Remove hooks
    for h in hooks:
        h.remove()

    if len(losses) == 0:
        # fallback: no bn modules found (shouldn't happen), return zero loss tensor on correct device
        return torch.tensor(0.0, device=device)

    return sum(losses)


def denorm_if_needed(img_tensor):
    """
    Convert decoder output to displayable range [0,1].
    Handles these common decoder output conventions:
      - if values look like tanh in [-1,1] -> map (x+1)/2 to [0,1]
      - if values already in [0,1] -> clamp
      - if values are normalized (mean/std) with CIFAR stats -> invert that normalization
    We use CIFAR_MEAN/CIFAR_STD tensors (on same device) defined earlier.
    """
    # quick stats (work on CPU tensors if needed)
    mn = float(img_tensor.min().detach().cpu())
    mx = float(img_tensor.max().detach().cpu())

    # case: tanh outputs in [-1,1]
    if mn < -0.5 and mx <= 1.0:
        img = (img_tensor + 1.0) / 2.0
        return img.clamp(0.0, 1.0)

    # case: already in [0,1]
    if mn >= -0.01 and mx <= 1.01:
        return img_tensor.clamp(0.0, 1.0)

    # case: appears to be normalized with CIFAR mean/std (rough heuristic)
    # i.e., values roughly centered near zero with std about ~1
    if abs(mn) < 5 and abs(mx) < 7:
        # invert normalization: x_pixel = x_norm * std + mean
        # CIFAR_MEAN / CIFAR_STD are (1,3,1,1) tensors on device
        img = img_tensor * CIFAR_STD + CIFAR_MEAN
        return img.clamp(0.0, 1.0)

    # fallback: clamp
    return img_tensor.clamp(0.0, 1.0)


# ----------------- Main generation loop -----------------
for img_idx in range(n_images):
    print(f"\n=== Generating image {img_idx+1}/{n_images} ===")
    z = torch.randn(z_shape, device=device, requires_grad=True)

    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]

    optimizer_emb = torch.optim.Adam(opt_params, lr=lr)

    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])            # out in (B,3,H,W)
        loss = (l2_weight * torch.mean(out**2) +
                tv_weight * tv_loss(out) +
                z_l2_weight * torch.mean(z**2))
        if bn_weight > 0:
            loss += bn_weight * bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

        if step % 50 == 0:
            print(f"step {step}/{num_steps}  loss={loss.item():.6f}")

    with torch.no_grad():
        final_img = denorm_if_needed(out)
        save_path = os.path.join(out_dir, f"deepinv_resnet152_cifar_img_{img_idx:03d}.png")
        vutils.save_image(final_img, save_path)
        print(f"✅ Saved: {save_path}")

print(f"\nAll {n_images} images saved in: {out_dir}")

# ----------------- Nearest-dataset-image comparison + display -----------------
cifar_root = "./data"
dataset_batch = 64
feature_cache_path = os.path.join(out_dir, "dataset_features_resnet152_cifar.pt")
compare_distance = "cosine"
save_pairs_dir = os.path.join(out_dir, "recon_vs_nearest")
os.makedirs(save_pairs_dir, exist_ok=True)

def get_feature_for_img_tensor_cifar(img_tensor, encoder, device):
    encoder.eval()
    imgs_proc = torch.zeros((img_tensor.shape[0], 3, 32, 32), device=device)
    for i in range(img_tensor.shape[0]):
        pil = transforms.ToPILImage()(img_tensor[i].cpu())
        proc = pil.resize((32,32), Image.BILINEAR)
        t = transforms.ToTensor()(proc).to(device)
        t = transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010])(t)
        imgs_proc[i] = t
    with torch.no_grad():
        feats = encoder(imgs_proc)
        feats_gap = feats.mean(dim=[2,3])
    return feats_gap

def compute_nearest_dataset_idx(query_feats, dataset_feats, metric="cosine"):
    if metric == "euclidean" and query_feats.shape[1] != dataset_feats.shape[1]:
        print("Warning: feature dims mismatch, using cosine instead")
        metric = "cosine"

    if metric == "euclidean":
        dists = torch.cdist(query_feats, dataset_feats)
        idx = torch.argmin(dists, dim=1)
        return idx, dists
    elif metric == "cosine":
        qn = torch.nn.functional.normalize(query_feats, dim=1)
        dn = torch.nn.functional.normalize(dataset_feats, dim=1)
        sims = qn @ dn.t()
        idx = torch.argmax(sims, dim=1)
        return idx, sims
    else:
        raise ValueError("Unsupported metric")

def extract_and_cache_dataset_features(encoder, loader, cache_path, device, dataset_for_features):
    if os.path.exists(cache_path):
        print("Loading cached dataset features:", cache_path)
        cached = torch.load(cache_path, map_location='cpu')
        return cached['features'], cached['paths']

    encoder.eval()
    all_feats, all_paths = [], []

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            feats = encoder(x)
            feats_gap = feats.mean(dim=[2,3])
            all_feats.append(feats_gap.cpu())
            start_idx = batch_idx * loader.batch_size
            end_idx   = start_idx + feats_gap.shape[0]
            for i in range(start_idx, min(end_idx, len(dataset_for_features))):
                all_paths.append(i)

    features = torch.cat(all_feats, dim=0)
    torch.save({'features': features, 'paths': all_paths}, cache_path)
    print("Cached dataset features to:", cache_path)
    return features, all_paths

preprocess_for_encoder = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

display_transform = transforms.Compose([transforms.ToTensor()])

dataset_for_features = datasets.CIFAR10(root=cifar_root, train=False, transform=preprocess_for_encoder, download=True)
dataset_for_display  = datasets.CIFAR10(root=cifar_root, train=False, transform=display_transform, download=False)

dataset_loader = DataLoader(dataset_for_features, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)
display_loader = DataLoader(dataset_for_display, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)

dataset_features, dataset_paths = extract_and_cache_dataset_features(encoder_for_bn, dataset_loader, feature_cache_path, device, dataset_for_features)
dataset_features = dataset_features.to(device)

n_images_to_generate = n_images
global_saved = 0

for img_idx in range(n_images_to_generate):
    print(f"\n--- Producing nearest-match display for reconstruction {img_idx+1}/{n_images_to_generate} ---")
    z = torch.randn(z_shape, device=device, requires_grad=True)
    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]

    optimizer_emb = torch.optim.Adam(opt_params, lr=lr)
    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])
        loss = (l2_weight * torch.mean(out**2) +
                tv_weight * tv_loss(out) +
                z_l2_weight * torch.mean(z**2))
        if bn_weight > 0:
            loss += bn_weight * bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

    recon = denorm_if_needed(out.detach().cpu())
    recon_for_enc = recon.to(device)
    recon_feats = get_feature_for_img_tensor_cifar(recon_for_enc, encoder_for_bn, device)
    idxs, sims = compute_nearest_dataset_idx(recon_feats, dataset_features, metric="cosine")
    nearest_idx = idxs[0].item()
    nearest_img, _ = dataset_for_display[nearest_idx]
    recon_img_display = transforms.ToPILImage()(recon[0])
    nearest_img_display = transforms.ToPILImage()(nearest_img)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(recon_img_display); axs[0].set_title("Reconstruction"); axs[0].axis('off')
    axs[1].imshow(nearest_img_display); axs[1].set_title(f"Nearest CIFAR-10 (idx={nearest_idx})"); axs[1].axis('off')
    sim_val = float(sims[0, nearest_idx].cpu()) if compare_distance == "cosine" else float(sims[0, nearest_idx].cpu())
    plt.suptitle(f"Recon {img_idx:03d}  -- cosine similarity: {sim_val:.4f}")
    pair_save = os.path.join(save_pairs_dir, f"recon_{img_idx:03d}_nearest.png")
    plt.savefig(pair_save, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)

    vutils.save_image(transforms.ToTensor()(recon_img_display), os.path.join(save_pairs_dir, f"recon_{img_idx:03d}.png"))
    vutils.save_image(transforms.ToTensor()(nearest_img_display), os.path.join(save_pairs_dir, f"nearest_{img_idx:03d}.png"))
    print(f"Saved pair: {pair_save}  -- dataset idx: {nearest_idx}")
    global_saved += 1

print(f"\nDone. Saved {global_saved} recon vs nearest CIFAR-10 pairs to: {save_pairs_dir}")

# %% [markdown] Cell 26
# #Resnet 34 which was pretrained on cifar10

# %% Cell 27
# resnet34_deepinv_cifar.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ----------------- Model definitions (CIFAR-style ResNet-34) -----------------
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetCIFAR(nn.Module):
    def __init__(self, block=BasicBlock, layers=[3,4,6,3], num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.in_channels = 64
        # CIFAR-style stem: 3x3 conv, stride=1, no maxpool
        self.conv1 = conv3x3(3, 64, stride=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        # residual layers
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x

def make_resnet34_cifar(num_classes=10):
    return ResNetCIFAR(block=BasicBlock, layers=[3,4,6,3], num_classes=num_classes)

# ----------------- Environment assumptions (uncomment / set as needed) -----------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# decoder = <your decoder instance loaded with weights>

# ----------------- Build model and load checkpoint -----------------
resnet34_model = make_resnet34_cifar(num_classes=10).to(device)
ckpt_path = "/content/drive/MyDrive/Pre_trained_model_weights/resnet34_cifar10.pth"   # <-- change to your checkpoint filepath

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

sd = torch.load(ckpt_path, map_location='cpu')
if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
    sd = sd['state_dict']

try:
    resnet34_model.load_state_dict(sd, strict=True)
    print("ResNet-34 (CIFAR) checkpoint loaded (strict=True).")
except Exception as e:
    print("Strict load failed:", e)
    print("Retrying with strict=False (will ignore missing/unexpected keys).")
    resnet34_model.load_state_dict(sd, strict=False)

# Build encoder_for_bn = everything up to (and excluding) avgpool + fc
encoder_for_bn = torch.nn.Sequential(*list(resnet34_model.children())[:-2]).to(device).eval()
for p in encoder_for_bn.parameters():
    p.requires_grad = False

# ----------------- Reconstruction / deepinv params (adapted for BasicBlock) -----------------
out_dir = "deepinv_outputs_resnet34_cifar"
os.makedirs(out_dir, exist_ok=True)

# For BasicBlock (expansion=1) final channels = 512, so latent shapes are smaller than Bottleneck-based nets.
B = 1
n_images = 5
num_steps = 300
save_every = 100
lr = 0.1
l2_weight = 1.0
tv_weight = 1e-4
bn_weight = 1.0
perc_weight = 0.0
z_l2_weight = 1e-6
optimize_skips = False
use_image_denorm = False

# shapes for z & skips adapted for ResNet-34 (final channels 512)
z_shape = (B, 512, 7, 7)
skip_shapes = [(B, 256, 14, 14), (B, 128, 28, 28), (B, 64, 56, 56)]

# CIFAR normalization (used for dataset feature extraction and encoding)
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1,3,1,1)
CIFAR_STD  = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1,3,1,1)

def tv_loss(img):
    dy = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    dx = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return dx + dy

def bn_stat_loss_from_encoder(decoder_out, encoder_model, device):
    # register forward hooks on BN layers of encoder_model and compare batch stats to running stats
    bn_features = []
    def hook_fn(module, inp, out):
        bn_features.append(out)
    bn_modules = [m for m in encoder_model.modules() if isinstance(m, nn.BatchNorm2d)]
    hooks = [m.register_forward_hook(hook_fn) for m in bn_modules]
    # run forward
    _ = encoder_model(decoder_out)
    losses = []
    # zip outputs with bn_modules (should match in count & order for models built consistently)
    for out_bn, module in zip(bn_features, bn_modules):
        mean_batch = out_bn.mean(dim=[0,2,3])
        var_batch  = out_bn.var(dim=[0,2,3], unbiased=False)
        losses.append(((mean_batch - module.running_mean.to(device))**2).mean())
        losses.append(((var_batch  - module.running_var.to(device))**2).mean())
    for h in hooks:
        h.remove()
    return sum(losses)

def denorm_if_needed(img_tensor):
    # decoder outputs are expected in [0,1] range (optionally normalized during training)
    if use_image_denorm:
        return ((img_tensor * CIFAR_STD) + CIFAR_MEAN).clamp(0,1)
    else:
        return img_tensor.clamp(0,1)

# ----------------- Main generation loop -----------------
for img_idx in range(n_images):
    print(f"\n=== Generating image {img_idx+1}/{n_images} ===")
    z = torch.randn(z_shape, device=device, requires_grad=True)

    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]

    optimizer_emb = torch.optim.Adam(opt_params, lr=lr)

    # Optimize latent embeddings
    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])            # out in (B,3,H,W)
        loss = (l2_weight * torch.mean(out**2) +
                tv_weight * tv_loss(out) +
                z_l2_weight * torch.mean(z**2))
        if bn_weight > 0:
            loss += bn_weight * bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

        if step % 50 == 0:
            print(f"step {step}/{num_steps}  loss={loss.item():.6f}")

    # save final output
    with torch.no_grad():
        final_img = denorm_if_needed(out)
        save_path = os.path.join(out_dir, f"deepinv_resnet34_cifar_img_{img_idx:03d}.png")
        vutils.save_image(final_img, save_path)
        print(f"✅ Saved: {save_path}")

print(f"\nAll {n_images} images saved in: {out_dir}")

# ================== Nearest-dataset-image comparison + display ==================
# ---------- User settings ----------
cifar_root = "./data"
dataset_batch = 64
feature_cache_path = os.path.join(out_dir, "dataset_features_resnet34_cifar.pt")
compare_distance = "cosine"  # "euclidean" or "cosine"
save_pairs_dir = os.path.join(out_dir, "recon_vs_nearest")
os.makedirs(save_pairs_dir, exist_ok=True)

# ---------- Helper: get feature vector for reconstructed image ----------
def get_feature_for_img_tensor_cifar(img_tensor, encoder, device):
    # img_tensor: (B,3,H,W) in [0,1] (already denormed/clamped)
    encoder.eval()
    # Ensure input size 32x32 (CIFAR)
    imgs_proc = torch.zeros((img_tensor.shape[0], 3, 32, 32), device=device)
    for i in range(img_tensor.shape[0]):
        pil = transforms.ToPILImage()(img_tensor[i].cpu())
        proc = pil.resize((32,32), Image.BILINEAR)
        t = transforms.ToTensor()(proc).to(device)
        t = transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010])(t)
        imgs_proc[i] = t
    with torch.no_grad():
        feats = encoder(imgs_proc)                # (B, C_encoder, Hf, Wf)
        feats_gap = feats.mean(dim=[2,3]) # (B, C_encoder)
    return feats_gap

# ---------- Helper: compute nearest dataset index ----------
def compute_nearest_dataset_idx(query_feats, dataset_feats, metric="cosine"):
    if metric == "euclidean" and query_feats.shape[1] != dataset_feats.shape[1]:
        print("Warning: feature dims mismatch, using cosine instead")
        metric = "cosine"

    if metric == "euclidean":
        dists = torch.cdist(query_feats, dataset_feats)
        idx = torch.argmin(dists, dim=1)
        return idx, dists
    elif metric == "cosine":
        qn = torch.nn.functional.normalize(query_feats, dim=1)
        dn = torch.nn.functional.normalize(dataset_feats, dim=1)
        sims = qn @ dn.t()   # similarity matrix
        idx = torch.argmax(sims, dim=1)
        return idx, sims
    else:
        raise ValueError("Unsupported metric")

# ---------- Helper: extract / cache dataset features ----------
def extract_and_cache_dataset_features(encoder, loader, cache_path, device, dataset_for_features):
    if os.path.exists(cache_path):
        print("Loading cached dataset features:", cache_path)
        cached = torch.load(cache_path, map_location='cpu')
        return cached['features'], cached['paths']

    encoder.eval()
    all_feats, all_paths = [], []

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)   # x already normalized for CIFAR in loader
            feats = encoder(x)                # (B, C, Hf, Wf)
            feats_gap = feats.mean(dim=[2,3]) # (B, C)
            all_feats.append(feats_gap.cpu())

            # store dataset indices as "paths"
            start_idx = batch_idx * loader.batch_size
            end_idx   = start_idx + feats_gap.shape[0]
            for i in range(start_idx, min(end_idx, len(dataset_for_features))):
                all_paths.append(i)

    features = torch.cat(all_feats, dim=0)
    torch.save({'features': features, 'paths': all_paths}, cache_path)
    print("Cached dataset features to:", cache_path)
    return features, all_paths

# ---------- Transforms ----------
preprocess_for_encoder = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

display_transform = transforms.Compose([transforms.ToTensor()])

# ---------- CIFAR-10 datasets & loaders ----------
dataset_for_features = datasets.CIFAR10(root=cifar_root, train=False, transform=preprocess_for_encoder, download=True)
dataset_for_display  = datasets.CIFAR10(root=cifar_root, train=False, transform=display_transform, download=False)

dataset_loader = DataLoader(dataset_for_features, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)
display_loader = DataLoader(dataset_for_display, batch_size=dataset_batch, shuffle=False, num_workers=2, pin_memory=True)

# ---------- Step 1: Extract / cache dataset features ----------
dataset_features, dataset_paths = extract_and_cache_dataset_features(encoder_for_bn, dataset_loader, feature_cache_path, device, dataset_for_features)
dataset_features = dataset_features.to(device)
dataset_feat_dim = dataset_features.shape[1]

# ---------- Step 2: Main reconstruction -> nearest neighbor comparison ----------
n_images_to_generate = n_images
global_saved = 0

for img_idx in range(n_images_to_generate):
    print(f"\n--- Producing nearest-match display for reconstruction {img_idx+1}/{n_images_to_generate} ---")

    # Recreate same reconstruction process (to ensure recon is available)
    z = torch.randn(z_shape, device=device, requires_grad=True)
    if optimize_skips:
        z_skip1 = torch.randn(skip_shapes[0], device=device, requires_grad=True)
        z_skip2 = torch.randn(skip_shapes[1], device=device, requires_grad=True)
        z_skip3 = torch.randn(skip_shapes[2], device=device, requires_grad=True)
        opt_params = [z, z_skip1, z_skip2, z_skip3]
    else:
        z_skip1 = torch.zeros(skip_shapes[0], device=device)
        z_skip2 = torch.zeros(skip_shapes[1], device=device)
        z_skip3 = torch.zeros(skip_shapes[2], device=device)
        opt_params = [z]

    optimizer_emb = torch.optim.Adam(opt_params, lr=lr)
    for step in range(1, num_steps+1):
        optimizer_emb.zero_grad()
        out = decoder(z, [z_skip1, z_skip2, z_skip3])
        loss = (l2_weight * torch.mean(out**2) +
                tv_weight * tv_loss(out) +
                z_l2_weight * torch.mean(z**2))
        if bn_weight > 0:
            loss += bn_weight * bn_stat_loss_from_encoder(out, encoder_for_bn, device)
        loss.backward()
        optimizer_emb.step()

    recon = denorm_if_needed(out.detach().cpu())    # (B,3,H,W) cpu

    # Compute features of reconstruction (using CIFAR-sized preprocessing)
    recon_for_enc = recon.to(device)
    recon_feats = get_feature_for_img_tensor_cifar(recon_for_enc, encoder_for_bn, device)  # (B, C)

    # Find nearest CIFAR-10 image (cosine)
    idxs, sims = compute_nearest_dataset_idx(recon_feats, dataset_features, metric="cosine")
    nearest_idx = idxs[0].item()

    # Get nearest CIFAR-10 image for display
    nearest_img, _ = dataset_for_display[nearest_idx]   # (3,32,32) tensor in [0,1]
    # Resize both to 32x32 -> for display we keep 32 but can upscale for plotting clarity
    recon_img_display = transforms.ToPILImage()(recon[0])
    nearest_img_display = transforms.ToPILImage()(nearest_img)

    # ---------- Display side by side ----------
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(recon_img_display); axs[0].set_title("Reconstruction"); axs[0].axis('off')
    axs[1].imshow(nearest_img_display); axs[1].set_title(f"Nearest CIFAR-10 (idx={nearest_idx})"); axs[1].axis('off')
    sim_val = float(sims[0, nearest_idx].cpu()) if compare_distance == "cosine" else float(sims[0, nearest_idx].cpu())
    plt.suptitle(f"Recon {img_idx:03d}  -- cosine similarity: {sim_val:.4f}")
    pair_save = os.path.join(save_pairs_dir, f"recon_{img_idx:03d}_nearest.png")
    plt.savefig(pair_save, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)

    # Save individual images (as in your pipeline)
    vutils.save_image(transforms.ToTensor()(recon_img_display), os.path.join(save_pairs_dir, f"recon_{img_idx:03d}.png"))
    vutils.save_image(transforms.ToTensor()(nearest_img_display), os.path.join(save_pairs_dir, f"nearest_{img_idx:03d}.png"))
    print(f"Saved pair: {pair_save}  -- dataset idx: {nearest_idx}")
    global_saved += 1

print(f"\nDone. Saved {global_saved} recon vs nearest CIFAR-10 pairs to: {save_pairs_dir}")

# %% Cell 28
extract_and_cache_dataset_features = None

# %% Cell 29
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# %% Cell 30
# Universal Pretrained Model Reconstruction Dashboard (robust path handling)
# NOTE: notebook shell command commented during conversion: !pip install -q gradio nbformat pillow

import gradio as gr
import nbformat
import os
import sys
import glob
import tempfile
import shutil
import traceback
from io import StringIO
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import urllib.request
import zipfile
import tarfile
from PIL import Image

# ----------------- 🔧 EDIT: Set your notebook path -----------------
NOTEBOOK_PATH = "/content/drive/MyDrive/Colab Notebooks/ Experiments and Testing-Decoder_Final.ipynb"
# ------------------------------------------------------------------

# -------------------- Parse the notebook --------------------------
nb = nbformat.read(open(NOTEBOOK_PATH, "r", encoding="utf-8"), as_version=4)
model_cells, current_model = {}, None
for cell in nb.cells:
    # treat a top-level markdown heading (starting with "#") as model name
    if cell.cell_type == "markdown" and cell.source.strip().startswith("#"):
        current_model = cell.source.strip().splitlines()[0].replace("#", "").strip()
    elif cell.cell_type == "code" and current_model:
        model_cells[current_model] = cell.source
        current_model = None

model_names = list(model_cells.keys())

# -------------------- Utility: download & extraction --------------------
def _download_url(url, dst_dir):
    parsed = urlparse(url)
    fname = os.path.basename(parsed.path) or "downloaded_file"
    dst = os.path.join(dst_dir, fname)
    # google-drive special-case handled by caller if needed
    urllib.request.urlretrieve(url, dst)
    return dst

def _extract_if_archive(filepath, dst_dir):
    try:
        if zipfile.is_zipfile(filepath):
            with zipfile.ZipFile(filepath, "r") as z:
                z.extractall(dst_dir)
            return dst_dir
        if tarfile.is_tarfile(filepath):
            with tarfile.open(filepath, "r:*") as t:
                t.extractall(dst_dir)
            return dst_dir
    except Exception:
        pass
    return filepath

def _is_google_drive(url_or_id):
    return ("drive.google.com" in url_or_id) or (len(url_or_id) >= 25 and all(c.isalnum() or c in "-_" for c in url_or_id))

def _gdrive_download(drive_url_or_id, dst_dir):
    # Try to extract file id from common drive url formats
    if "drive.google.com" in drive_url_or_id:
        parsed = urlparse(drive_url_or_id)
        # handle /file/d/FILEID/ and ?id=FILEID
        if "/file/d/" in parsed.path:
            fid = parsed.path.split("/file/d/")[1].split("/")[0]
        else:
            qs = parse_qs(parsed.query)
            fid = qs.get("id", [None])[0]
    else:
        fid = drive_url_or_id  # assume raw id
    if not fid:
        raise ValueError("Could not parse Google Drive file id.")
    # Public file direct-download URL
    dl_url = f"https://drive.google.com/uc?export=download&id={fid}"
    return _download_url(dl_url, dst_dir)

# -------------------- Resolver: return a local filesystem path --------------------
def resolve_input(uploaded_file, path_or_url_txt, work_dir=None):
    """
    Resolve input priority:
      1) uploaded_file (gr.File) -> returns local temp path
      2) path_or_url_txt (string):
           - if exists on local FS -> return it
           - if http/https -> download and return local filepath (or extract folder)
           - if google drive url or id -> try to download (public file)
           - else raise FileNotFoundError
    Returns: local path (file or directory)
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="resolve_")
    os.makedirs(work_dir, exist_ok=True)

    # 1) uploaded file object from Gradio (take precedence)
    if uploaded_file:
        # gr.File may be an object with .name or .file, or a dict with 'name'/'tmp_path'
        try_paths = []
        # Newer gradio returns a TemporaryFile-like with .name
        if hasattr(uploaded_file, "name") and isinstance(uploaded_file.name, str):
            try_paths.append(uploaded_file.name)
        # older returns dict-like
        if isinstance(uploaded_file, dict):
            if uploaded_file.get("tmp_path"):
                try_paths.append(uploaded_file.get("tmp_path"))
            if uploaded_file.get("name"):
                try_paths.append(uploaded_file.get("name"))
        # tempfile-like .file attribute
        if hasattr(uploaded_file, "file") and hasattr(uploaded_file.file, "name"):
            try_paths.append(uploaded_file.file.name)

        for p in try_paths:
            if p and os.path.exists(p):
                # move to our work_dir to avoid ephemeral file deletion surprises
                dst = os.path.join(work_dir, os.path.basename(p))
                shutil.copy(p, dst)
                return dst

    # 2) text input or URL
    if not path_or_url_txt or not str(path_or_url_txt).strip():
        raise FileNotFoundError("No input provided (neither upload nor path/URL).")

    txt = str(path_or_url_txt).strip()

    # If it's already an existing local path
    if os.path.exists(txt):
        # If file and archive -> extract; if dir -> return dir
        if os.path.isdir(txt):
            return txt
        out = _extract_if_archive(txt, work_dir)
        return out

    parsed = urlparse(txt)
    scheme = parsed.scheme.lower()

    try:
        if scheme in ("http", "https"):
            # special-case google drive links
            if "drive.google.com" in txt:
                downloaded = _gdrive_download(txt, work_dir)
            else:
                downloaded = _download_url(txt, work_dir)
            return _extract_if_archive(downloaded, work_dir)

        # try google drive raw id or short id
        if _is_google_drive(txt):
            downloaded = _gdrive_download(txt, work_dir)
            return _extract_if_archive(downloaded, work_dir)

    except Exception as e:
        raise FileNotFoundError(f"Failed to download or resolve '{txt}': {e}")

    raise FileNotFoundError(f"Path/URL not found or not resolvable: {txt}")

# -------------------- Helper: get reconstruction images --------------------
def get_recon_pairs(folder="deepinv_outputs/recon_vs_nearest", max_pairs=5):
    pairs = sorted(glob.glob(os.path.join(folder, "*.png")))
    imgs = []
    for p in pairs[-max_pairs:]:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(img)
        except Exception:
            pass
    return imgs

# -------------------- Executor: run selected model cell --------------------
def run_selected_model(model_choice,
                       decoder_file, decoder_path_txt,
                       encoder_file, encoder_path_txt):
    logs_buf = StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = logs_buf

    # Resolve decoder path (uploaded file takes precedence)
    work_dir = tempfile.mkdtemp(prefix="run_")
    try:
        try:
            resolved_decoder = resolve_input(decoder_file, decoder_path_txt, work_dir=work_dir)
        except Exception as e:
            raise FileNotFoundError(f"Decoder resolution error: {e}")

        # Determine if this model requires encoder (CIFAR)
        lower = (model_choice or "").lower()
        cifar_mode = "cifar" in lower

        resolved_encoder = None
        if cifar_mode:
            try:
                resolved_encoder = resolve_input(encoder_file, encoder_path_txt, work_dir=work_dir)
            except Exception as e:
                raise FileNotFoundError(f"Encoder resolution error (required for CIFAR models): {e}")

        # inject resolved paths into globals used by notebook code
        globals()["DECODER_PATH"] = resolved_decoder
        globals()["ENCODER_PATH"] = resolved_encoder if cifar_mode else None

        code = model_cells.get(model_choice, None)
        if code is None:
            raise ValueError(f"No code cell found for model '{model_choice}'")

        # execute the notebook cell code in globals()
        exec(code, globals())
        print("\n✅ Reconstruction completed successfully.")
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        logs = logs_buf.getvalue()

    imgs = get_recon_pairs()
    if not imgs:
        logs += "\n⚠️ No images found in deepinv_outputs/recon_vs_nearest/. Ensure the notebook cell saves recon images to that folder."
    # Ensure exactly 5 outputs (some may be None)
    imgs += [None] * (5 - len(imgs))
    return logs, imgs[:5]

# -------------------- Gradio UI --------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Pretrained Model Reconstruction Dashboard (robust uploads & paths)")
    gr.Markdown(
        "- Upload a decoder `.pth` **or** paste a local path / HTTP / Google Drive link.\n"
        "- For CIFAR models (name contains 'cifar') provide an encoder `.pth` similarly.\n"
        "- Uploaded files take precedence over pasted paths/URLs."
    )

    with gr.Row():
        model_dd = gr.Dropdown(choices=model_names, label="Select Model")
    with gr.Row():
        decoder_file = gr.File(label="Upload Decoder .pth (preferred)", file_count="single")
        decoder_tb = gr.Textbox(label="Or paste Decoder path / URL", placeholder="/content/drive/MyDrive/decoder_final.pth or https://...")
    with gr.Row():
        encoder_file = gr.File(label="Upload Encoder .pth (only for CIFAR models)", file_count="single")
        encoder_tb = gr.Textbox(label="Or paste Encoder path / URL (only for CIFAR models)", placeholder="/content/drive/MyDrive/resnet50_cifar.pth or https://...")

    run_btn = gr.Button("▶️ Run Selected Model")
    output_log = gr.Textbox(label="Execution Log", lines=20)

    with gr.Row():
        img_outputs = [gr.Image(label=f"Recon Pair {i+1}", type="pil", show_label=True) for i in range(5)]

    def wrapped_run(model_name, dec_file, dec_txt, enc_file, enc_txt):
        logs, imgs = run_selected_model(model_name, dec_file, dec_txt, enc_file, enc_txt)
        # return [logs] + imgs
        return [logs] + imgs

    run_btn.click(wrapped_run, inputs=[model_dd, decoder_file, decoder_tb, encoder_file, encoder_tb],
                  outputs=[output_log] + img_outputs)
    model_dd.change(wrapped_run, inputs=[model_dd, decoder_file, decoder_tb, encoder_file, encoder_tb],
                    outputs=[output_log] + img_outputs)

# Launch
demo.launch(share=True, server_name="0.0.0.0", server_port=7089)

# %% Cell 31
from google.colab import drive
drive.flush_and_unmount()

