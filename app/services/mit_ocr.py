"""
Manga-Image-Translator OCR 模型支持

支持模型:
- model_48px: ocr_ar_48px.ckpt (基于 ConvNeXt + Transformer，文字高度 48px)
- model_48px_ctc: ocr-ctc.ckpt (基于 ResNet + Transformer + CTC，文字高度 48px)

直接从 manga-translator-ui 复制的模型定义
"""

import os
import math
import zipfile
from typing import List, Tuple, Optional
from threading import Lock

import cv2
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.core.logger import logger
from app.core.paths import MODELS_DIR

# 导入 XPOS 位置编码
from .xpos_relative_position import XPOS

# 模型下载 URL 配置
MIT_OCR_MODEL_URLS = {
    "model_48px": {
        "model": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr_ar_48px.ckpt",
        "dict": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/alphabet-all-v7.txt",
    },
    "model_48px_ctc": {
        "archive": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr-ctc.zip",
        "model": "ocr-ctc.ckpt",
        "dict": "alphabet-all-v5.txt",
    },
}

# 文字高度
TEXT_HEIGHTS = {
    "model_48px": 48,
    "model_48px_ctc": 48,
}

_MODEL_LOCK = Lock()
_MIT_OCR_MODELS = {}


def _download_file(url: str, path: str) -> None:
    """下载文件"""
    import requests
    from tqdm import tqdm
    
    logger.info(f"下载 {url} -> {path}")
    
    response = requests.get(url, stream=True, allow_redirects=True)
    total = int(response.headers.get("content-length", 0))
    
    with open(path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(path)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def _ensure_model_files(model_type: str) -> Tuple[str, str]:
    """确保模型文件存在，如果不存在则下载"""
    model_dir = MODELS_DIR / "mit_ocr"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if model_type == "model_48px":
        model_path = model_dir / "ocr_ar_48px.ckpt"
        dict_path = model_dir / "alphabet-all-v7.txt"
        
        if not model_path.exists():
            _download_file(MIT_OCR_MODEL_URLS["model_48px"]["model"], str(model_path))
        if not dict_path.exists():
            _download_file(MIT_OCR_MODEL_URLS["model_48px"]["dict"], str(dict_path))
            
    elif model_type == "model_48px_ctc":
        model_path = model_dir / "ocr-ctc.ckpt"
        dict_path = model_dir / "alphabet-all-v5.txt"
        
        if not model_path.exists() or not dict_path.exists():
            # 下载 zip 文件并解压
            archive_path = model_dir / "ocr-ctc.zip"
            if not archive_path.exists():
                _download_file(MIT_OCR_MODEL_URLS["model_48px_ctc"]["archive"], str(archive_path))
            
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(model_dir)
    
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    return str(model_path), str(dict_path)


# ============ 工具函数 ============

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """生成因果掩码"""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


# ============ Model 48px 组件 ============

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, ks=7, padding=3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=ks, padding=padding, groups=dim)
        self.norm = nn.BatchNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1, 1, 0)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1, 1, 0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(1, dim, 1, 1), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return input + x


class ConvNext_FeatureExtractor(nn.Module):
    def __init__(self, img_height=48, in_dim=3, dim=512, n_layers=12):
        super().__init__()
        base = dim // 8
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, base, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(base),
            nn.ReLU(),
            nn.Conv2d(base, base * 2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(),
        )
        self.block1 = self._make_layers(base * 2, 4)
        self.down1 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(),
        )
        self.block2 = self._make_layers(base * 4, 12)
        self.down2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(),
        )
        self.block3 = self._make_layers(base * 8, 10, ks=5, padding=2)
        self.down3 = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(),
        )
        self.block4 = self._make_layers(base * 8, 8, ks=3, padding=1)
        self.down4 = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, kernel_size=(3, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(),
        )

    def _make_layers(self, dim, n, ks=7, padding=3):
        layers = []
        for i in range(n):
            layers.append(ConvNeXtBlock(dim, ks=ks, padding=padding))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.down3(x)
        x = self.block4(x)
        x = self.down4(x)
        return x


def transformer_encoder_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
    x = src
    if self.norm_first:
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
    else:
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
    return x


class XposMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, self_attention=False, encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.xpos = XPOS(self.head_dim, embed_dim)
        self.batch_first = True
        self._qkv_same_embed_dim = True

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None,
                need_weights=False, is_causal=False, k_offset=0, q_offset=0):
        assert not is_causal
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)

        if self.xpos is not None:
            k = self.xpos(k, offset=k_offset, downscale=True)
            q = self.xpos(q, offset=q_offset, downscale=False)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, embed_dim).transpose(0, 1)

        attn = self.out_proj(attn)
        return attn, attn_weights if need_weights else (attn, None)


# ============ Model 48px OCR ============

class OCR48px(nn.Module):
    """48px OCR 模型"""
    def __init__(self, dictionary, max_len):
        super().__init__()
        self.max_len = max_len
        self.dictionary = dictionary
        self.dict_size = len(dictionary)
        embd_dim = 320
        nhead = 4
        
        self.backbone = ConvNext_FeatureExtractor(48, 3, embd_dim)
        
        # Encoder layers with XposMultiheadAttention
        self.encoders = nn.ModuleList()
        for i in range(4):
            encoder = nn.TransformerEncoderLayer(embd_dim, nhead, dropout=0, batch_first=True, norm_first=True)
            encoder.self_attn = XposMultiheadAttention(embd_dim, nhead, self_attention=True)
            self.encoders.append(encoder)
        
        # Decoder layers with XposMultiheadAttention
        self.decoders = nn.ModuleList()
        for i in range(5):
            decoder = nn.TransformerDecoderLayer(embd_dim, nhead, dropout=0, batch_first=True, norm_first=True)
            decoder.self_attn = XposMultiheadAttention(embd_dim, nhead, self_attention=True)
            decoder.multihead_attn = XposMultiheadAttention(embd_dim, nhead, encoder_decoder_attention=True)
            self.decoders.append(decoder)
        
        self.embd = nn.Embedding(self.dict_size, embd_dim)
        self.pred1 = nn.Sequential(nn.Linear(embd_dim, embd_dim), nn.GELU(), nn.Dropout(0.15))
        self.pred = nn.Linear(embd_dim, self.dict_size)
        self.pred.weight = self.embd.weight
        
        self.color_pred1 = nn.Sequential(nn.Linear(embd_dim, 64), nn.ReLU())
        self.color_pred_fg = nn.Linear(64, 3)
        self.color_pred_bg = nn.Linear(64, 3)
        self.color_pred_fg_ind = nn.Linear(64, 2)
        self.color_pred_bg_ind = nn.Linear(64, 2)

    def _encoder_forward(self, memory, encoder_mask):
        """使用 XposMultiheadAttention 的编码器前向传播"""
        for layer in self.encoders:
            # norm_first 模式
            x = memory
            x = layer.norm1(x)
            # self-attention with xpos
            x = layer.self_attn(x, x, x, key_padding_mask=encoder_mask)[0]
            x = memory + layer.dropout1(x)
            # FFN
            memory = x + layer._ff_block(layer.norm2(x))
        return memory

    def _decoder_forward(self, embd, memory, tgt_mask, memory_mask):
        """使用 XposMultiheadAttention 的解码器前向传播"""
        tgt = embd
        for layer in self.decoders:
            # norm_first 模式
            x = tgt
            x = layer.norm1(x)
            # self-attention with xpos
            x = layer.self_attn(x, x, x, attn_mask=tgt_mask)[0]
            x = tgt + layer.dropout1(x)
            # cross-attention
            x2 = layer.norm2(x)
            x2 = layer.multihead_attn(x2, memory, memory, key_padding_mask=memory_mask, q_offset=0)[0]
            x = x + layer.dropout2(x2)
            # FFN
            tgt = x + layer._ff_block(layer.norm3(x))
        return tgt

    def forward(self, img, char_idx, decoder_mask, encoder_mask):
        memory = self.backbone(img)
        memory = einops.rearrange(memory, 'N C 1 W -> N W C')
        memory = self._encoder_forward(memory, encoder_mask)
        
        N, L = char_idx.shape
        char_embd = self.embd(char_idx)
        casual_mask = generate_square_subsequent_mask(L).to(img.device)
        decoded = self._decoder_forward(char_embd, memory, casual_mask, encoder_mask)
        
        pred_char_logits = self.pred(self.pred1(decoded))
        color_feats = self.color_pred1(decoded)
        return (pred_char_logits,
                self.color_pred_fg(color_feats),
                self.color_pred_bg(color_feats),
                self.color_pred_fg_ind(color_feats),
                self.color_pred_bg_ind(color_feats))

    def infer_simple(self, img: torch.Tensor, beam_size: int = 5, max_length: int = 255) -> List[Tuple[str, float]]:
        """简化的贪婪解码"""
        device = img.device
        N, C, H, W = img.shape
        
        # 编码
        memory = self.backbone(img)
        memory = einops.rearrange(memory, 'N C 1 W -> N W C')
        memory = self._encoder_forward(memory, None)
        
        results = []
        start_tok = 1  # <S>
        end_tok = 2    # </S>
        
        for n in range(N):
            tokens = [start_tok]
            total_logprob = 0.0
            
            for _ in range(max_length):
                tgt = torch.tensor([tokens], dtype=torch.long, device=device)
                tgt_embd = self.embd(tgt)
                casual_mask = generate_square_subsequent_mask(len(tokens)).to(device)
                decoded = self._decoder_forward(tgt_embd, memory[n:n+1], casual_mask, None)
                
                logits = self.pred(self.pred1(decoded))[-1]
                probs = F.log_softmax(logits, dim=-1)
                top_prob, top_idx = probs.max(dim=-1)
                
                if top_idx.item() == end_tok:
                    break
                
                tokens.append(top_idx.item())
                total_logprob += top_prob.item()
            
            # 转换为文字
            text = ""
            for tok_id in tokens[1:]:
                ch = self.dictionary[tok_id]
                if ch == "<SP>":
                    ch = " "
                if ch not in ("<S>", "</S>"):
                    text += ch
            
            prob = math.exp(total_logprob / max(len(tokens) - 1, 1))
            results.append((text, prob))
        
        return results


# ============ Model 48px CTC 组件 ============

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, offset=0):
        x = x + self.pe[:, offset: offset + x.size(1), :]
        return x


class BasicBlockCTC(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return out + residual


class ResNetCTC(nn.Module):
    def __init__(self, input_channel, output_channel, block, layers):
        super().__init__()
        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]
        self.inplanes = int(output_channel / 8)
        
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 8), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 8))
        self.conv0_2 = nn.Conv2d(int(output_channel / 8), self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.maxpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.maxpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[1], kernel_size=3, stride=1, padding=1, bias=False)

        self.maxpool3 = nn.AvgPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[2], kernel_size=3, stride=1, padding=1, bias=False)

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], kernel_size=3, stride=(2, 1), padding=(1, 1), bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3], kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4_3 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.BatchNorm2d(self.inplanes),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = F.relu(x)
        x = self.conv0_2(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.layer4(x)
        x = self.bn4_1(x)
        x = F.relu(x)
        x = self.conv4_1(x)
        x = self.bn4_2(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_3(x)
        return x


class ResNet_FeatureExtractorCTC(nn.Module):
    def __init__(self, input_channel, output_channel=128):
        super().__init__()
        self.ConvNet = ResNetCTC(input_channel, output_channel, BasicBlockCTC, [4, 6, 8, 6, 3])

    def forward(self, input):
        return self.ConvNet(input)


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu",
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pe = PositionalEncoding(d_model, max_len=2048)
        self.activation = F.gelu

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(self.pe(x), self.pe(x), x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x


class OCR48pxCTC(nn.Module):
    """48px CTC OCR 模型"""
    def __init__(self, dictionary, max_len):
        super().__init__()
        self.max_len = max_len
        self.dictionary = dictionary
        self.dict_size = len(dictionary)
        
        self.backbone = ResNet_FeatureExtractorCTC(3, 320)
        enc = CustomTransformerEncoderLayer(320, 8, 320 * 4, dropout=0.05, batch_first=True, norm_first=True)
        self.encoders = nn.TransformerEncoder(enc, 3)
        self.char_pred_norm = nn.Sequential(nn.LayerNorm(320), nn.Dropout(0.1), nn.GELU())
        self.char_pred = nn.Linear(320, self.dict_size)
        self.color_pred1 = nn.Sequential(nn.Linear(320, 6))

    def forward(self, img):
        feats = self.backbone(img).squeeze(2)
        feats = self.encoders(feats.permute(0, 2, 1))
        pred_char_logits = self.char_pred(self.char_pred_norm(feats))
        pred_color_values = self.color_pred1(feats)
        return pred_char_logits, pred_color_values

    def decode_ctc(self, img: torch.Tensor, blank: int = 0) -> List[Tuple[str, float]]:
        """CTC 解码"""
        device = img.device
        N, C, H, W = img.shape
        
        feats = self.backbone(img).squeeze(2)
        feats = self.encoders(feats.permute(0, 2, 1))
        pred_char_logits = self.char_pred(self.char_pred_norm(feats))
        pred_color_values = self.color_pred1(feats)
        
        results = []
        logprobs = pred_char_logits.log_softmax(2)
        _, preds_index = logprobs.max(2)
        preds_index = preds_index.cpu()
        pred_color_values = pred_color_values.cpu().clamp_(0, 1)
        
        for b in range(N):
            text = ""
            last_ch = blank
            total_logprob = 0.0
            count = 0
            
            for t in range(pred_char_logits.size(1)):
                pred_ch = preds_index[b, t]
                if pred_ch != last_ch and pred_ch != blank:
                    ch = self.dictionary[pred_ch]
                    if ch == "<SP>":
                        ch = " "
                    if ch not in ("<S>", "</S>"):
                        text += ch
                    total_logprob += logprobs[b, t, pred_ch].item()
                    count += 1
                last_ch = pred_ch
            
            prob = math.exp(total_logprob / max(count, 1))
            results.append((text, prob))
        
        return results


# ============ 模型加载接口 ============

def get_mit_ocr_model(model_type: str, device: str = "cpu") -> Tuple[nn.Module, int]:
    """获取 MIT OCR 模型"""
    global _MIT_OCR_MODELS
    
    with _MODEL_LOCK:
        cache_key = f"{model_type}:{device}"
        if cache_key in _MIT_OCR_MODELS:
            return _MIT_OCR_MODELS[cache_key], TEXT_HEIGHTS[model_type]
        
        # 确保模型文件存在
        model_path, dict_path = _ensure_model_files(model_type)
        
        # 加载字典
        with open(dict_path, "r", encoding="utf-8") as f:
            dictionary = [s[:-1] for s in f.readlines()]
        
        # 创建模型
        if model_type == "model_48px":
            # 使用原生 MIT 实现
            from .mit_ocr_48px_native import OCR as OCR48pxNative
            model = OCR48pxNative(dictionary, 768)
            sd = torch.load(model_path, map_location="cpu", weights_only=False)
            # 处理 PyTorch Lightning 格式
            if 'state_dict' in sd:
                sd = sd['state_dict']
            # 移除 'model.' 前缀
            cleaned_sd = {}
            for k, v in sd.items():
                if k.startswith('model.'):
                    cleaned_sd[k[6:]] = v
                else:
                    cleaned_sd[k] = v
            model.load_state_dict(cleaned_sd, strict=True)
            
        elif model_type == "model_48px_ctc":
            # 使用原生 MIT 实现
            from .mit_ocr_ctc_native import OCR
            model = OCR(dictionary, 768)
            sd = torch.load(model_path, map_location="cpu", weights_only=False)
            sd = sd["model"] if "model" in sd else sd
            # 必须删除 PE 缓冲区键（这是原始实现的关键步骤）
            pe_keys_to_remove = [
                'encoders.layers.0.pe.pe',
                'encoders.layers.1.pe.pe', 
                'encoders.layers.2.pe.pe'
            ]
            for key in pe_keys_to_remove:
                if key in sd:
                    del sd[key]
            model.load_state_dict(sd, strict=False)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        model.eval()
        if device == "cuda" and torch.cuda.is_available():
            model = model.to(device)
        
        _MIT_OCR_MODELS[cache_key] = model
        logger.info(f"MIT OCR 模型 {model_type} 加载成功，设备: {device}")
        
        return model, TEXT_HEIGHTS[model_type]


def mit_ocr_recognize(image_pil: Image.Image, model_type: str = "model_48px") -> str:
    """使用 MIT OCR 模型识别图片中的文字"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, text_height = get_mit_ocr_model(model_type, device)
    
    # 转换图片
    img_np = np.array(image_pil.convert("RGB"))
    
    # 检查最小尺寸
    h, w = img_np.shape[:2]
    if h < 4 or w < 4:
        logger.debug(f"MIT OCR ({model_type}): 图像太小 ({w}x{h})，跳过")
        return ""
    
    # 检测竖排文字（高度明显大于宽度），需要旋转
    is_vertical = h > w * 1.2
    if is_vertical:
        # 逆时针旋转90度，将竖排转为横排
        img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = img_np.shape[:2]
        logger.debug(f"MIT OCR ({model_type}): 检测到竖排文字，旋转后尺寸 {w}x{h}")
    
    # 调整高度到 text_height，保持宽高比
    new_w = int(round(w * text_height / h))
    if new_w < 4:
        new_w = 4
    
    img_resized = cv2.resize(img_np, (new_w, text_height), interpolation=cv2.INTER_AREA)
    
    # 准备输入（归一化到 [-1, 1]，与原始实现一致）
    img_tensor = torch.from_numpy(img_resized).float() - 127.5
    img_tensor = img_tensor / 127.5
    img_tensor = einops.rearrange(img_tensor, "H W C -> 1 C H W")
    if device == "cuda":
        img_tensor = img_tensor.cuda()
    
    # 推理
    with torch.no_grad():
        if model_type == "model_48px_ctc":
            # CTC 模型使用 decode 方法
            texts = model.decode(img_tensor, [new_w], blank=0)
            if texts and texts[0]:
                chars = texts[0]
                text = ""
                total_logprob = 0.0
                for (chid, lp, *_) in chars:
                    ch = model.dictionary[chid]
                    if ch == "<SP>":
                        ch = " "
                    if ch not in ("<S>", "</S>"):
                        text += ch
                    total_logprob += lp
                prob = math.exp(total_logprob / max(len(chars), 1))
                logger.debug(f"MIT OCR ({model_type}): {text} (prob={prob:.4f})")
                return text
            return ""
        else:
            # model_48px 使用 beam search
            results = model.infer_beam_batch(img_tensor, [new_w], beams_k=5, max_seq_length=255)
            if results:
                pred_chars_index, prob, *_ = results[0]
                text = ""
                for chid in pred_chars_index:
                    ch = model.dictionary[chid]
                    if ch == "<SP>":
                        ch = " "
                    if ch not in ("<S>", "</S>"):
                        text += ch
                logger.debug(f"MIT OCR ({model_type}): {text} (prob={prob:.4f})")
                return text
    return ""


def mit_ocr_recognize_with_bbox(
    image_pil: Image.Image, 
    bbox: np.ndarray,
    model_type: str = "model_48px_ctc"
) -> Tuple[str, float]:
    """使用 MIT OCR 模型识别图片中指定区域的文字"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, text_height = get_mit_ocr_model(model_type, device)
    
    # 转换图片
    img_np = np.array(image_pil.convert("RGB"))
    
    # 裁剪区域
    bbox = np.array(bbox)
    
    if bbox.ndim == 1 and len(bbox) == 4:
        x1, y1, x2, y2 = map(int, bbox)
        region = img_np[y1:y2, x1:x2]
    elif bbox.ndim == 2 and bbox.shape == (4, 2):
        region = _get_transformed_region(img_np, bbox, text_height)
    else:
        logger.warning(f"不支持的 bbox 格式: {bbox.shape}")
        return "", 0.0
    
    if region.size == 0:
        return "", 0.0
    
    # 检查最小尺寸
    h, w = region.shape[:2]
    if h < 4 or w < 4:
        logger.debug(f"MIT OCR ({model_type}): 区域太小 ({w}x{h})，跳过")
        return "", 0.0
    
    # 准备输入（归一化到 [-1, 1]）
    img_tensor = (torch.from_numpy(region).float() / 127.5 - 1.0)
    img_tensor = einops.rearrange(img_tensor, "H W C -> 1 C H W")
    if device == "cuda":
        img_tensor = img_tensor.cuda()
    
    # 推理
    with torch.no_grad():
        results = model.infer_beam_batch(img_tensor, [w], beams_k=5, max_seq_length=255)
    
    if results:
        pred_chars_index, prob, *_ = results[0]
        text = ""
        for chid in pred_chars_index:
            ch = model.dictionary[chid]
            if ch == "<SP>":
                ch = " "
            if ch not in ("<S>", "</S>"):
                text += ch
        logger.debug(f"MIT OCR ({model_type}): {text} (prob={prob:.4f})")
        return text, prob
    return "", 0.0


def _get_transformed_region(image: np.ndarray, pts: np.ndarray, text_height: int) -> np.ndarray:
    """透视变换"""
    pts = pts.astype(np.float32)
    
    def _dist(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    width_top = _dist(pts[0], pts[1])
    width_bottom = _dist(pts[3], pts[2])
    
    target_width = int(max(width_top, width_bottom))
    target_height = text_height
    
    if target_width < 4:
        target_width = 4
    
    dst_pts = np.array([
        [0, 0],
        [target_width, 0],
        [target_width, target_height],
        [0, target_height]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    region = cv2.warpPerspective(image, M, (target_width, target_height))
    
    return region