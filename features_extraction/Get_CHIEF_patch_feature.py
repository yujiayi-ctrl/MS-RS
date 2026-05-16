#!/usr/bin/env python3
"""
Usage:
    CUDA_VISIBLE_DEVICES=1 nohup python /home/yujy/PC-TMD/features_extraction/Get_CHIEF_patch_feature.py \
        --base_data_dir /252_node_user_storage/yujy/CAMS_data/patches \
        --log_dir /252_node_user_storage/yujy/CAMS_data/features_768 \
        --model_weight /data/home/scxj642/run/yujy/MS-RS/features_extraction/CHIEF_CTransPath.pth \
        --magnifications 5x 10x 20x \
        --gpu_ids 1 > 1_18_features.log 2>&1 &
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys
import pandas as pd
import numpy as np
import pickle
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import time
import io
import math
from functools import partial

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

Image.MAX_IMAGE_PIXELS = None


# ==================== Tool fuction ====================

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ==================== Swin Transformer Components ====================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class ConvStem(nn.Module):
    """CTransPath的ConvStem - 与CHIEF权重结构完全匹配"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # 构建与权重文件匹配的结构
        # proj.0: Conv2d, proj.1: BatchNorm2d, proj.2: ReLU
        # proj.3: Conv2d, proj.4: BatchNorm2d, proj.5: ReLU
        # proj.6: Conv2d (1x1)
        stem = []
        input_dim, output_dim = 3, embed_dim // 8  # 768//8 = 96
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class SwinTransformerForCTransPath(nn.Module):
    """CTransPath使用的Swin Transformer"""
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=21841,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # CTransPath uses ConvStem
        self.patch_embed = ConvStem(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def ctranspath():

    model = SwinTransformerForCTransPath(
        img_size=224,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        num_classes=21841,
    )
    return model


def load_ctranspath(weight_path, device='cuda'):
    """Load the pre-trained CTransPath model"""
    print(f"📥 Loading CTransPath model: {weight_path}")
    
    model = ctranspath()
    model.head = nn.Identity()
    
    td = torch.load(weight_path, map_location='cpu', weights_only=False)
    state_dict = td['model'] if 'model' in td else td
    
    filtered_dict = {}
    skipped_keys = []
    for k, v in state_dict.items():
        if 'relative_position_index' in k or 'attn_mask' in k:
            skipped_keys.append(k)
        else:
            filtered_dict[k] = v
    
    if skipped_keys:
        print(f"   Skipping {len(skipped_keys)} buffer keys")
    
    missing, unexpected = model.load_state_dict(filtered_dict, strict=False)
    
    if missing:

        real_missing = [k for k in missing if 'relative_position_index' not in k and 'attn_mask' not in k]
        if real_missing:
            print(f"   ⚠️ Missing parameters: {real_missing}")
    
    if unexpected:
        print(f"   ⚠️ Unexpected parameters: {unexpected}")
    
    print("   ✓ Weights loaded successfully")
    
    model = model.to(device)
    model.eval()
    
    # 测试
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224).to(device)
        test_output = model(test_input)
        print(f"   ✓ Feature dimensions: {test_output.shape[-1]}")
    
    return model



class PatchDataset(Dataset):
    def __init__(self, patch_dir, slide_id, magnification, transform=None):
        self.patch_dir = patch_dir
        self.slide_id = slide_id
        self.magnification = magnification
        self.transform = transform
        
        self.patch_paths = sorted(glob.glob(os.path.join(patch_dir, '*.jpg')))
        if not self.patch_paths:
            self.patch_paths = sorted(glob.glob(os.path.join(patch_dir, '*.png')))
        
        self.patch_coords = [self._parse_coords(os.path.basename(p)) for p in self.patch_paths]
    
    def _parse_coords(self, filename):
        try:
            parts = filename.replace('.jpg', '').replace('.png', '').split('_')
            return (int(parts[1]), int(parts[2]))
        except:
            return (0, 0)
    
    def __len__(self):
        return len(self.patch_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.patch_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'file_name': os.path.basename(self.patch_paths[idx]),
            'coords': torch.tensor(self.patch_coords[idx], dtype=torch.float32)
        }


def extract_features(model, loader, device, use_amp=True):
    model.eval()
    all_features, all_file_names, all_coords = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Extracting", leave=False):
            images = batch['image'].to(device, non_blocking=True)
            
            if use_amp and device.type == 'cuda':
                with autocast():
                    features = model(images)
            else:
                features = model(images)
            
            all_features.append(features.float().cpu())
            all_file_names.extend(batch['file_name'])
            all_coords.append(batch['coords'])
    
    return torch.cat(all_features, dim=0), all_file_names, torch.cat(all_coords, dim=0)


def main():
    parser = argparse.ArgumentParser(description='CHIEF CTransPath Feature Extraction')
    
    parser.add_argument('--base_data_dir', type=str, default='/252_node_user_storage/yujy/CAMS_data/patches')
    parser.add_argument('--log_dir', type=str, default='/252_node_user_storage/yujy/CAMS_data/features_768')
    parser.add_argument('--model_weight', type=str, default='/data/home/scxj642/run/yujy/MS-RS/features_extraction/CHIEF_CTransPath.pth')
    parser.add_argument('--dataset_csv', type=str, default='/home/yujy/CAMS_data/CAMS_clinical_processed585.csv')
    parser.add_argument('--magnifications', nargs='+', type=str, default=['5x', '10x', '20x'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--save_format', type=str, default='pt', choices=['pt', 'pkl'])
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("🔬 CHIEF CTransPath Feature Extraction (768-dim) - Standalone")
    print("="*70)
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Patch目录: {args.base_data_dir}")
    print(f"输出目录: {args.log_dir}")
    print(f"放大倍数: {args.magnifications}")
    print(f"批大小: {args.batch_size}")
    print("="*70)
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model = load_ctranspath(args.model_weight, device)
    
    for mag in args.magnifications:
        os.makedirs(os.path.join(args.log_dir, mag), exist_ok=True)
    
    df = pd.read_csv(args.dataset_csv)
    print(f"\n📊 样本数量: {len(df)}")
    
    processed, skipped, failed = 0, 0, 0
    start_time = time.time()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Overall"):
        slide_id = row['slide_id']
        
        for mag in args.magnifications:
            output_path = os.path.join(args.log_dir, mag, f'{slide_id}.{args.save_format}')
            
            if args.skip_existing and os.path.exists(output_path):
                skipped += 1
                continue
            
            patch_dir = os.path.join(args.base_data_dir, slide_id, mag)
            
            if not os.path.exists(patch_dir):
                failed += 1
                continue
            
            dataset = PatchDataset(patch_dir, slide_id, mag, transform)
            
            if len(dataset) == 0:
                failed += 1
                continue
            
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
            
            features, file_names, coords = extract_features(model, loader, device, args.use_amp)
            
            if args.save_format == 'pt':
                torch.save(features, output_path)
            else:
                with open(output_path, 'wb') as f:
                    pickle.dump({
                        'slide_id': slide_id, 'magnification': mag,
                        'features': [{'feature': features[i].numpy(), 'file_name': file_names[i], 
                                    'coords': coords[i].numpy()} for i in range(len(features))],
                        'num_patches': len(features), 'feature_dim': 768
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            processed += 1
    
    total_time = (time.time() - start_time) / 60
    
    print("\n" + "="*70)
    print("✅ Feature extraction completed!")
    print("="*70)
    print(f"Total time: {total_time:.2f} minutes")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Feature dimension: 768")
    print(f"Save location: {args.log_dir}")
    print("="*70)


if __name__ == '__main__':
    main()