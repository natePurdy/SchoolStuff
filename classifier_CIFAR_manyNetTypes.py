import pickle
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from PIL import Image, ImageOps, ImageEnhance
import seaborn as sns
import random

"""
Input Data is CFAR-10 data set. ~87 percent accuracy on final test set using deeperCNN

INPUT DATA: BINARY files conatining batch splits of the data (should be roughly randomized across classes per batch)
- 32x32 colour image
-  The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. 
- The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

"""


# some important high level parameters...
numEpochs = 300
learningRate = 0.0005 # lower learning rate if using spiked NN
batchSize = 128
image_size = (32, 32)   # 32x32 colour image
augmentTrainData = True
dataSet = "CIFAR10"



def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True     # important for conv layers etc.
    torch.backends.cudnn.benchmark = False

# Call this early
set_global_seed(12345)   # pick any fixed number

# ############ DIFFERENT NN MODELS LAID OUT HERE ################################

# function to cut holes out or images randomly (augmentation helper)
def custom_cutout(img, rng, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
    if rng.random() > p:
        return img
    
    h, w = img.shape[1], img.shape[2]
    area = h * w
    
    for _ in range(100):  # num retries to find valid rect
        erase_area = area * rng.uniform(*scale)
        aspect_ratio = rng.uniform(*ratio)
        
        erase_h = int(round(np.sqrt(erase_area * aspect_ratio)))
        erase_w = int(round(np.sqrt(erase_area / aspect_ratio)))
        
        if erase_h >= h or erase_w >= w:
            continue
        
        x = rng.randint(0, w - erase_w)
        y = rng.randint(0, h - erase_h)
        
        # random value per channel (like value="random")
        v = rng.uniform(0, 1, size=(3, 1, 1))
        
        img[:, y:y+erase_h, x:x+erase_w] = v
        break  # success
    
    return img

# model is defined here
class CNN(nn.Module):
    def __init__(self, input_shape=(3, image_size[0], image_size[1]), num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding='same'),  # padding=1 keeps size for 3x3 kernel
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves spatial dimensions

            nn.Conv2d(8, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves again

            nn.Conv2d(16, 32, kernel_size=2, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # halves again
        )

        # Compute flattened size properly
        C, H, W = input_shape
        numPoolingLayers = 3 # count them above
        H = int(H /(2**numPoolingLayers)) # 3 MaxPool layers, each halves H
        W = int(W / (2**numPoolingLayers)) # 3 MaxPool layers, each halves W

        flattened_size = 32 * H * W

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# more deep version of basic CNN here
class DeeperCNN(nn.Module):
    def __init__(self, input_shape=(3,32,32), num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 8x8 -> 4x4
        )

        flattened_size = 128 * 4 * 4  # last block channels * spatial dims

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



class DeeperViT(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), num_classes=10,
                 patch_size=4,       # 32/4 = 8 → 64 patches
                 embed_dim=256,      # token dimension
                 depth=6,            # number of transformer layers
                 num_heads=8,        # attention heads
                 mlp_ratio=4,        # FFN hidden dim = embed_dim * mlp_ratio
                 dropout=0.1):
        super().__init__()
        
        channels, height, width = input_shape
        assert height == width and height % patch_size == 0, "Image size must be divisible by patch_size"
        
        self.patch_size = patch_size
        self.num_patches = (height // patch_size) ** 2   # e.g. 64 for 32×32 & patch=4
        self.embed_dim = embed_dim
        
        # 1. Patch embedding: Conv2d acts as linear projection per patch
        self.patch_embed = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 2. CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
        # 3. Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',           # or 'relu'
            batch_first=True,
            norm_first=True              # pre-norm → more stable on small data
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim)  # final norm
        )
        
        # 4. Classification head (from CLS token)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Optional: simple init (helps small models)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        # You can also init patch_embed weights, but PyTorch defaults are ok

    def forward(self, x):
        B = x.shape[0]  # batch size
        
        # Patch embed → flatten spatial dims → (B, num_patches, embed_dim)
        x = self.patch_embed(x)           # (B, embed_dim, h/p, w/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, 1 + num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoder (expects (B, seq_len, embed_dim))
        x = self.transformer_encoder(x)
        
        # Take CLS token output
        cls_output = x[:, 0]   # or x.mean(dim=1) for mean pooling
        
        # Classification
        logits = self.head(cls_output)
        return logits
# learnable torch friendly hyperbolic tan function to try out replacing the normalization layer in standard transformer architecture,
# based on https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Transformers_without_Normalization_CVPR_2025_paper.pdf 
class DyT(nn.Module):
    def __init__(self, dim: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1,), alpha_init))   # scalar or per-dim if you want
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x shape: (B, seq, dim)
        return self.gamma * torch.tanh(self.alpha * x) + self.beta

class DeeperViT_normalizedUsingTan(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), num_classes=10,
                 patch_size=4,
                 embed_dim=256,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4,
                 dropout=0.1):
        super().__init__()
        
        # ... (patch_embed, cls_token, pos_embed stay the same)

        # Transformer encoder layers — replace internal LayerNorm with DyT
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True              # keep pre-norm style
        )
        
        # Important: nn.TransformerEncoderLayer still has internal LayerNorms!
        # → We need to subclass or monkey-patch, or re-implement the block.
        # Simplest practical way → define custom block (recommended)

        # Option A: Custom transformer block with DyT (cleanest)
        class DyTTransformerLayer(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(
                    d_model, nhead, dropout=dropout, batch_first=True
                )
                self.linear1 = nn.Linear(d_model, dim_feedforward)
                self.dropout = nn.Dropout(dropout)
                self.linear2 = nn.Linear(dim_feedforward, d_model)
                self.dropout1 = nn.Dropout(dropout)
                self.dropout2 = nn.Dropout(dropout)
                
                # ← Here we replace LayerNorm with DyT
                self.norm1 = DyT(d_model)
                self.norm2 = DyT(d_model)
                
                self.activation = F.gelu

            def forward(self, src):
                # Pre-norm style
                src2 = self.norm1(src)
                attn_output, _ = self.self_attn(src2, src2, src2)
                src = src + self.dropout1(attn_output)
                
                src2 = self.norm2(src)
                ff_output = self.linear2(self.dropout(self.activation(self.linear1(src2))))
                src = src + self.dropout2(ff_output)
                return src

        # Now use stack of custom layers
        self.transformer_encoder = nn.Sequential(*[
            DyTTransformerLayer(
                embed_dim, num_heads, embed_dim * mlp_ratio, dropout
            ) for _ in range(depth)
        ])

        # Final norm → also replace
        # self.final_norm = nn.LayerNorm(embed_dim)   # ← remove / comment
        self.final_norm = DyT(embed_dim)             # ← use DyT

        # Head
        self.head = nn.Sequential(
            # nn.LayerNorm(embed_dim),          # ← remove
            DyT(embed_dim),                     # ← replace
            nn.Linear(embed_dim, num_classes)
        )

        self._init_weights()

    def forward(self, x):
        # ... same patch embedding + cls + pos ...

        x = self.transformer_encoder(x)
        
        cls_output = x[:, 0]
        
        # Optional: apply final DyT before head
        cls_output = self.final_norm(cls_output)
        
        return self.head(cls_output)


# use this residul block class to add resnet to an existing CNN model
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = torch.relu(out)
        return out

# trying to add on resnet to existing deep CNN (needs ResidualBlock class...)
class DeeperCNN_withResnet(nn.Module):
    def __init__(self, input_shape=(3,32,32), num_classes=10):
        super().__init__()

        self.features = nn.Sequential(

            # Stem
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 1 (32x32)
            ResidualBlock(32),
            ResidualBlock(32),
            nn.MaxPool2d(2),  # -> 16x16

            # Block 2 (16x16)
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.MaxPool2d(2),  # -> 8x8

            # Block 3 (8x8)
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.MaxPool2d(2)   # -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class deepRecurrantNN(nn.Module):
    def __init__(self, input_shape=(3,32,32), num_classes=10, hidden_size=256):
        super().__init__()
        
        # Optional: lightweight feature extractor (like a CNN but lighter duty)
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → 16x16
            # ... add more if desired
        )
        
        # RNN processes sequence (e.g., rows or flattened patches)
        # Here: treat as sequence of row vectors after flatten/reshape
        channels_after_pre = 32  # adjust based on preprocess
        row_dim = channels_after_pre * 16  # e.g., 32 channels × 16 width after pool
        
        self.rnn = nn.LSTM(
            input_size=row_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True  # often helps
        )
        
        # Classifier on final hidden state (or last output)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),  # *2 if bidirectional
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Optional conv features first
        x = self.preprocess(x)          # (B, C', H/2, W/2)
        
        # Reshape to sequence: e.g., treat height as time, width*channels as feature
        B, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3)       # (B, H, C, W)
        x = x.reshape(B, H, -1)         # (B, seq_len=H, feature_dim=C*W)
        
        # RNN
        out, (h_n, c_n) = self.rnn(x)
        # Use last hidden (or mean pool, or concat directions)
        x = h_n[-2:].transpose(0,1).reshape(B, -1)  # bidirectional last
        
        return self.classifier(x)


# spiking neural network based on CNN as starting point
import snntorch as snn
from snntorch import surrogate
class SpikingDeeperCNN(nn.Module):
    def __init__(self, input_shape=(3,32,32), num_classes=10, num_steps=25, beta=0.9):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta
        slope = 10  # Lower for wider surrogate (helps gradients flow)
        self.spike_grad = surrogate.fast_sigmoid(slope=slope)
        # raise spike threshold to improve validation accuracy, makes random neuron firing less impactful
        self.spikeThreshold = 0.6  # Lower initial threshold if having training issues at statrt
        
        # Features: Conv blocks with spiking neurons
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.lif1_1 = snn.Leaky(beta=beta, threshold=self.spikeThreshold, learn_threshold=True, spike_grad=self.spike_grad)
        
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.lif1_2 = snn.Leaky(beta=beta, threshold=self.spikeThreshold, learn_threshold=True, spike_grad=self.spike_grad)
        
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.lif2_1 = snn.Leaky(beta=beta, threshold=self.spikeThreshold, learn_threshold=True, spike_grad=self.spike_grad)
        
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.lif2_2 = snn.Leaky(beta=beta, threshold=self.spikeThreshold, learn_threshold=True, spike_grad=self.spike_grad)
        
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.lif3_1 = snn.Leaky(beta=beta, threshold=self.spikeThreshold, learn_threshold=True, spike_grad=self.spike_grad)
        
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.lif3_2 = snn.Leaky(beta=beta, threshold=self.spikeThreshold, learn_threshold=True, spike_grad=self.spike_grad)
        
        self.pool3 = nn.MaxPool2d(2)

        flattened_size = 128 * 4 * 4
        
        # Classifier with spiking layers
        self.fc1 = nn.Linear(flattened_size, 256)
        self.lif_fc1 = snn.Leaky(beta=beta, threshold=self.spikeThreshold, learn_threshold=True, spike_grad=self.spike_grad)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.lif_fc2 = snn.Leaky(beta=beta, threshold=self.spikeThreshold, learn_threshold=True, spike_grad=self.spike_grad)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, num_classes)
        self.lif_fc3 = snn.Leaky(beta=beta, threshold=self.spikeThreshold, learn_threshold=True, spike_grad=self.spike_grad)  # Output layer

    def forward(self, x):
        # Initialize membranes
        mem1_1 = self.lif1_1.init_leaky()
        mem1_2 = self.lif1_2.init_leaky()
        mem2_1 = self.lif2_1.init_leaky()
        mem2_2 = self.lif2_2.init_leaky()
        mem3_1 = self.lif3_1.init_leaky()
        mem3_2 = self.lif3_2.init_leaky()
        mem_fc1 = self.lif_fc1.init_leaky()
        mem_fc2 = self.lif_fc2.init_leaky()
        mem_fc3 = self.lif_fc3.init_leaky()
        
        # Record spikes and membranes over time
        spk_out_rec = []
        mem_out_rec = []  # NEW: Record output membranes for loss
        
        for step in range(self.num_steps):
            cur = torch.poisson(x * 100)  # amplify signal sicne spikes are weak
            
            # Block 1 (unchanged)
            cur = self.conv1_1(cur)
            cur = self.bn1_1(cur)
            spk1_1, mem1_1 = self.lif1_1(cur, mem1_1)
            
            cur = self.conv1_2(spk1_1)
            cur = self.bn1_2(cur)
            spk1_2, mem1_2 = self.lif1_2(cur, mem1_2)
            
            cur = self.pool1(spk1_2)
            
            # Block 2 (unchanged)
            cur = self.conv2_1(cur)
            cur = self.bn2_1(cur)
            spk2_1, mem2_1 = self.lif2_1(cur, mem2_1)
            
            cur = self.conv2_2(spk2_1)
            cur = self.bn2_2(cur)
            spk2_2, mem2_2 = self.lif2_2(cur, mem2_2)
            
            cur = self.pool2(spk2_2)
            
            # Block 3 (unchanged)
            cur = self.conv3_1(cur)
            cur = self.bn3_1(cur)
            spk3_1, mem3_1 = self.lif3_1(cur, mem3_1)
            
            cur = self.conv3_2(spk3_1)
            cur = self.bn3_2(cur)
            spk3_2, mem3_2 = self.lif3_2(cur, mem3_2)
            
            cur = self.pool3(spk3_2)
            
            # Classifier (unchanged)
            cur = cur.view(cur.size(0), -1)
            cur = self.fc1(cur)
            spk_fc1, mem_fc1 = self.lif_fc1(cur, mem_fc1)
            cur = self.dropout1(spk_fc1)
            
            cur = self.fc2(cur)
            spk_fc2, mem_fc2 = self.lif_fc2(cur, mem_fc2)
            cur = self.dropout2(spk_fc2)
            
            cur = self.fc3(cur)
            spk_out, mem_fc3 = self.lif_fc3(cur, mem_fc3)
            
            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_fc3)  # NEW: Record membrane
        
        # Return stacked spikes and membranes
        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)

############################# end of nn models class definitions ########################################


# data set loading by batch routine, even though the data is already all loaded in as a binary dict, this will need to be used for training on images with higher resolution
class CIFARDataset(Dataset):
    def __init__(self, data_dict, augment=False, randSeed=42):
        """
        data_dict must contain:
          - b'data': numpy array (N, 3072) or (N, 3, 32, 32)
          - b'labels': list or array (N,)
        """
        self.X = data_dict[b'data']
        self.y = np.array(data_dict[b'labels'], dtype=np.int64)
        self.augment = augment
        self.rngForAugs = np.random.RandomState(randSeed)
         #more fancy augmentation#
        self.randaug = T.RandAugment(num_ops=2, magnitude=9)   # common for CIFAR ViTs; magnitude 9–10 aggressive but good

        # reshape once if needed
        if self.X.ndim == 2:
            self.X = self.X.reshape(-1, 3, 32, 32)

    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        img = self.X[idx]  # (C,H,W)
        label = self.y[idx]

        # Convert to PIL Image for augmentation
        img = np.transpose(img, (1, 2, 0))  # (H,W,C)
        img = (img).astype(np.uint8)
        img = Image.fromarray(img)

        if self.augment:
            # --- Horizontal flip ---
            if self.rngForAugs.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # --- Random crop with padding ---
            img = ImageOps.expand(img, border=4, fill=0)  # pad 4 pixels
            left = self.rngForAugs.randint(0, 9)
            top  = self.rngForAugs.randint(0, 9)
            img = img.crop((left, top, left+32, top+32))

            # --- Color jitter ---
            factor = 0.8 + self.rngForAugs.random() * 0.4  # brightness
            img = ImageEnhance.Brightness(img).enhance(factor)
            factor = 0.8 + self.rngForAugs.random() * 0.4  # contrast
            img = ImageEnhance.Contrast(img).enhance(factor)
            factor = 0.8 + self.rngForAugs.random() * 0.4  # saturation
            img = ImageEnhance.Color(img).enhance(factor)
            # more random augmentations...
            img = ImageOps.expand(img, border=4, fill=0)
            left = self.rngForAugs.randint(0, 9)
            top  = self.rngForAugs.randint(0, 9)
            img = img.crop((left, top, left+32, top+32))
            # Add RandAugment here (applies to PIL Image)
            img = self.randaug(img)

            # --- Convert to NumPy for Cutout ---
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # (C,H,W)
            # cut out some of the image randomly. like punching holes
            img = custom_cutout(img, self.rngForAugs, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))


        else:
            # If not augmenting, just normalize to [0,1] and convert to (C,H,W)
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


# function kto open CIFAR files (python version of data download) - this was form the website
def unpickle(file):
    """# Loaded in this way, each of the batch files contains a dictionary with the following elements:
    # data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    # labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
    """

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def parseInputData(dataFolder):

    classDefinitions = dataFolder + 'batches.meta'
    dataBatch1 = dataFolder + "data_batch_1"  # can load them in individually to have partitioned data (train on batches 1 through 5, test on test batch)
    dataBatch2 = dataFolder + "data_batch_2"
    dataBatch3 = dataFolder + "data_batch_3"
    dataBatch4 = dataFolder + "data_batch_4"
    dataBatch5 = dataFolder + "data_batch_5" # use for validation during training
    testBatch = dataFolder + "test_batch"

    # generate the data containing dictionaries 
    batch1_dict = unpickle(dataBatch1)
    batch2_dict = unpickle(dataBatch2)
    batch3_dict = unpickle(dataBatch3)
    batch4_dict = unpickle(dataBatch4)
    batch5_dict = unpickle(dataBatch5)
    testBatch_dict = unpickle(testBatch)

    #keys of the dictionary thats loaded in
    DATA_KEY = b'data'
    LABEL_KEY = b'labels'
    # combine the training data into one massive dict
    combinedForTraining = {}
    forValidation = {}
    forTesting = {}

    # concatenate image data
    combinedForTraining[b'data'] = np.concatenate(
        (
            batch1_dict[b'data'],
            batch2_dict[b'data'],
            batch3_dict[b'data'],
            batch4_dict[b'data'],
        ),
        axis=0
    )

    # concatenate labels (lists)
    combinedForTraining[b'labels'] = (
        batch1_dict[b'labels'] +
        batch2_dict[b'labels'] +
        batch3_dict[b'labels'] +
        batch4_dict[b'labels'] 
    )

    # for validation 
    forValidation[b'labels'] = batch5_dict[b'labels']
    forValidation[b'data'] = batch5_dict[b'data']

    # also do the same thing for clarity with the test data
    forTesting[b'labels'] = testBatch_dict[b'labels']
    forTesting[b'data'] = testBatch_dict[b'data']

    # print(combinedForTraining)
    # print(forTesting)


    # define the classes of objects that are in the images (can also be loaded in from)
    meta = unpickle(classDefinitions)
    meta_decoded = {
        key.decode("utf-8"): value
        for key, value in meta.items()
    }

    # Decode class names too
    meta_decoded["label_names"] = [
        name.decode("utf-8") for name in meta_decoded["label_names"]
    ]
    label_names = meta_decoded["label_names"]
    # convert to dict for easier lookup later on
    class_id_to_name = {
        i: name for i, name in enumerate(label_names)
    }

    return combinedForTraining, forValidation, forTesting, class_id_to_name

def trainTheArtificialNN(model, train_loader, val_loader, test_loader, optimizer):

    """
    For training CNN's, ViT's, Transformers, a general training pipeline using pytorch...
    
    """

    # for tracking training info
    history = {
    "loss": [],
    "accuracy": [],
    "val_loss": [],
    "val_accuracy": []
    }
    ########### DO TRAINING HERE ##############
    for epoch in range(numEpochs):
        #  Training 
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # loading bar returns data (isnt that nice)
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{numEpochs}", leave=False)

        # go through the batches of data
        for images, labels in train_bar:
            # load in batch of images and labels, save to computer mem
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # zero out gradient for each pass
            outputs = model(images) # pass a batch of images through model
            loss = criterion(outputs, labels) #compute loss given loss functiion
            loss.backward() # backward pass
            optimizer.step() # adjust model parameters using computed gradients
            running_loss += loss.item() * labels.size(0) # compute contribution to epoch loss from this single batch
            preds = outputs.argmax(dim=1) # determine predicted classes
            correct += (preds == labels).sum().item() # how many of them are correct
            total += labels.size(0) # running total of samples passed

            # update training bar after batch is sent through
            train_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct/total:.4f}"
            })

        train_loss = running_loss / total
        train_acc = correct / total

        # Execute the validation phase of the training process
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad(): # not updating model using validation data...
            for images, labels in val_loader:
                # laod in batch of validation images for validation
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # update training progress like keras does
        print(
            f"Epoch {epoch+1}/{numEpochs} "
            f"- loss: {train_loss:.4f} "
            f"- acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_acc: {val_acc:.4f}"
        )

    # Evaluate on the test set once done training...
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
            labels_all.extend(y.numpy())

    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)
    test_accuracy = (preds_all == labels_all).mean()
    print(f"Test accuracy: {test_accuracy:.4f}")


    return preds_all, labels_all, history, test_accuracy

def trainTheNaturalNN(model, train_loader, val_loader, test_loader, optimizer):
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }
    
    for epoch in range(numEpochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{numEpochs}", leave=False)

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            spk_rec, mem_rec = model(images)  # NEW: Get both spikes and membranes
            
            # NEW: Loss on membranes at each step, summed over time
            loss_train = torch.zeros((1), dtype=torch.float, device=device)
            for t in range(model.num_steps):
                loss_train += criterion(mem_rec[t], labels)
            
            #loss for spiking occurs over num_steps... divide if comparing loss to CNN type network
            loss_train = loss_train/model.num_steps
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            optimizer.step()
            
            running_loss += loss_train.item() * labels.size(0)  # Adjust for summed loss
            
            # Use spikes for predictions (rate decoding)
            spk_sum = spk_rec.sum(dim=0)
            preds = spk_sum.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Your debug print (updated)
            mean_spikes = spk_rec.mean()
            mean_mem = mem_rec.mean()
            # print(f"Mean spikes: {mean_spikes:.4f} | Mean mem magnitude: {mem_rec.abs().mean():.4f}")
            
            train_bar.set_postfix({
                "loss": f"{loss_train.item():.4f}",
                "acc": f"{correct/total:.4f}"
            })

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation (similar changes)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                spk_rec, mem_rec = model(images)
                
                loss_val = torch.zeros((1), dtype=torch.float, device=device)
                for t in range(model.num_steps):
                    loss_val += criterion(mem_rec[t], labels)
                
                loss_val = loss_val/model.num_steps
                val_loss += loss_val.item() * labels.size(0)
                spk_sum = spk_rec.sum(dim=0)
                preds = spk_sum.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{numEpochs} "
            f"- loss: {train_loss:.4f} "
            f"- acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_acc: {val_acc:.4f}"
        )

    # Test evaluation (similar changes)
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            spk_rec, mem_rec = model(x)
            spk_sum = spk_rec.sum(dim=0)
            preds_all.extend(spk_sum.argmax(dim=1).cpu().numpy())
            labels_all.extend(y.numpy())

    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)
    test_accuracy = (preds_all == labels_all).mean()
    print(f"Test accuracy: {test_accuracy:.4f}")

    return preds_all, labels_all, history, test_accuracy

def plotTrainingHistory(history, outputFolder, saveFig=False):



    # plot the training history
    history_df = pd.DataFrame(history)
    history_df.plot(figsize=(8,5))
    plt.grid(True)
    plt.ylim(None)
    if saveFig == True:
        plt.savefig(f"{outputFolder}/training.png")
        history_df.to_csv(f"{outputFolder}/_{numEpochs}epochs_{np.round(test_accuracy,3)}acc_batchsize{batchSize}.txt")
    plt.show()


def plotConfusionMatrix(labels_all, preds_all, class_id_to_name, outputFolder, saveFig=False):

    conf_matrix = np.zeros((10,10), dtype=int)
    for t, p in zip(labels_all, preds_all):
        conf_matrix[t, p] += 1
    class_names = [class_id_to_name[i] for i in range(len(class_id_to_name))] # extract class names from class dict
    sns.heatmap(conf_matrix, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if saveFig ==True:
        plt.savefig(f"{outputFolder}/confusionMat.png")
    plt.show()



################################################# main routine here... ###############################################

# cpu? decide what hardware the mnodel will live on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# where is the data located (its binary format like)
dataFolder = "/mnt/c/Users/npurd/Documents/SCHOOL_crap/ece_523/sandbox/dataSets/CIFAR/cifar-10-batches-py/"

# parse the data into training, validation, and testing data
combinedForTraining, forValidation, forTesting, class_id_to_name = parseInputData(dataFolder)

# MODEL SETUP ---- choose chich model and replace the following line...
model = SpikingDeeperCNN().to(device)                                                 # <------------------- set model here
criterion = nn.CrossEntropyLoss() # for binary classification (ex: dog or cat)
optimizer = optim.Adam(model.parameters(), lr=learningRate) # seems okay
# if using transformer, need adaptive learning rate or else it wont perform at all really (long warm up period)
if model.__class__.__name__ == "deeperVIT":
    optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay=0.05) # seems okay

testName = f"model{model.__class__.__name__}_augsAndPunches"
outputFolder = f"testResults_{dataSet}{testName}"
os.makedirs(outputFolder, exist_ok=True)
print(f"\nUsing model: {model.__class__.__name__} right now for training BTW...\n")


# intitalize classes for training and testing (use same random seed for recreatable results... always.)
train_dataset = CIFARDataset(combinedForTraining, augment=True,randSeed=12345)
val_dataset   = CIFARDataset(forValidation, augment=False, randSeed=12345)
test_dataset = CIFARDataset(forTesting, augment=False, randSeed=12345)
# training data
train_loader = DataLoader(train_dataset,batch_size=batchSize, shuffle=True, num_workers=2)
# validation data
val_loader = DataLoader(val_dataset,batch_size=batchSize,shuffle=False,num_workers=2)
# testing data
test_loader = DataLoader(test_dataset,batch_size=batchSize,shuffle=False,num_workers=2)

print(model.__class__.__name__)
########### DO TRAINING HERE ##############
if model.__class__.__name__ == "SpikingDeeperCNN":
    # slightly different training paradigm than regular CNN or transformer training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)  # optional small weight decay
    preds_all, labels_all, history, test_accuracy = trainTheNaturalNN(model, train_loader, val_loader, test_loader, optimizer)
else:
    preds_all, labels_all, history, test_accuracy = trainTheArtificialNN(model, train_loader, val_loader, test_loader, optimizer)

# plot the training results...
plotTrainingHistory(history, outputFolder, saveFig=True)

#make confusion matrix to show class "TF/TN" - ie what did model confuse classes with...
plotConfusionMatrix(labels_all, preds_all, class_id_to_name, outputFolder, saveFig=True)


