"""
MIT model_48px_ctc OCR 模型（原生实现）

直接从 manga-translator-ui 复制的核心模型类
"""

import math
from typing import List, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BasicBlock(nn.Module):
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


class ResNet(nn.Module):
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


class ResNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel=128):
        super().__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [4, 6, 8, 6, 3])

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


class OCR(nn.Module):
    """MIT 48px CTC OCR 模型"""
    def __init__(self, dictionary, max_len):
        super().__init__()
        self.max_len = max_len
        self.dictionary = dictionary
        self.dict_size = len(dictionary)
        self.backbone = ResNet_FeatureExtractor(3, 320)
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

    def decode(self, img: torch.Tensor, img_widths: List[int], blank: int = 0, verbose: bool = False) -> List[List[Tuple]]:
        """解码 OCR 结果"""
        N, C, H, W = img.shape
        assert H == 48 and C == 3
        feats = self.backbone(img).squeeze(2)
        feats = self.encoders(feats.permute(0, 2, 1))
        pred_char_logits = self.char_pred(self.char_pred_norm(feats))
        pred_color_values = self.color_pred1(feats)
        return self.decode_ctc_top1(pred_char_logits, pred_color_values, blank, verbose=verbose)

    def decode_ctc_top1(self, pred_char_logits, pred_color_values, blank, verbose=False) -> List[List[Tuple]]:
        """CTC Top-1 解码"""
        pred_chars = [[] for _ in range(pred_char_logits.size(0))]
        logprobs = pred_char_logits.log_softmax(2)
        _, preds_index = logprobs.max(2)
        preds_index = preds_index.cpu()
        pred_color_values = pred_color_values.cpu().clamp_(0, 1)
        for b in range(pred_char_logits.size(0)):
            last_ch = blank
            for t in range(pred_char_logits.size(1)):
                pred_ch = preds_index[b, t]
                if pred_ch != last_ch and pred_ch != blank:
                    lp = logprobs[b, t, pred_ch].item()
                    pred_chars[b].append((
                        pred_ch,
                        lp,
                        pred_color_values[b, t][0].item(),
                        pred_color_values[b, t][1].item(),
                        pred_color_values[b, t][2].item(),
                        pred_color_values[b, t][3].item(),
                        pred_color_values[b, t][4].item(),
                        pred_color_values[b, t][5].item()
                    ))
                last_ch = pred_ch
        return pred_chars

    def infer_beam_batch(self, img: torch.FloatTensor, img_widths: List[int], beams_k: int = 5,
                         start_tok: int = 1, end_tok: int = 2, pad_tok: int = 0,
                         max_finished_hypos: int = 2, max_seq_length: int = 384):
        """兼容接口：使用 CTC 解码"""
        N, C, H, W = img.shape
        results = self.decode(img, img_widths, blank=0)
        output = []
        for b, chars in enumerate(results):
            if not chars:
                output.append((torch.LongTensor([end_tok]), 0.0, 
                              torch.zeros(1, 3), torch.ones(1, 3),
                              torch.zeros(1, 2), torch.zeros(1, 2)))
                continue
            idx = torch.LongTensor([chid for (chid, *_) in chars])
            total_logprob = sum(lp for (_, lp, *_) in chars)
            prob = math.exp(total_logprob / max(len(chars), 1))
            
            fg_pred = torch.tensor([[fr, fg, fb] for (_, _, fr, fg, fb, *_) in chars])
            bg_pred = torch.tensor([[br, bg, bb] for (_, _, *_, br, bg, bb) in chars])
            fg_ind_pred = torch.zeros(len(chars), 2)
            bg_ind_pred = torch.zeros(len(chars), 2)
            
            output.append((idx, prob, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred))
        return output

    def decode_simple(self, img: torch.Tensor, blank: int = 0) -> List[Tuple[str, float]]:
        """简化解码，只返回文本和概率"""
        N, C, H, W = img.shape
        results = self.decode(img, [W], blank)
        output = []
        for chars in results:
            text = ""
            total_logprob = 0.0
            count = 0
            for (chid, lp, *_) in chars:
                ch = self.dictionary[chid]
                if ch == "<SP>":
                    ch = " "
                if ch not in ("<S>", "</S>"):
                    text += ch
                total_logprob += lp
                count += 1
            prob = math.exp(total_logprob / max(count, 1))
            output.append((text, prob))
        return output