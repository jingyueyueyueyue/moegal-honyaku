"""
MIT model_48px OCR 模型（原生实现）

直接从 manga-translator-ui 复制的核心模型类
"""

import math
from typing import List, Optional, Tuple
from collections import defaultdict

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入 XPOS 位置编码
from .xpos_relative_position import XPOS


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


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


def transformer_encoder_forward(
    self,
    src: torch.Tensor,
    src_mask: Optional[torch.Tensor] = None,
    src_key_padding_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False
) -> torch.Tensor:
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

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=False,
        is_causal=False,
        k_offset=0,
        q_offset=0
    ):
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


class Hypothesis:
    """解码假设，用于 beam search"""
    def __init__(self, device, start_tok: int, end_tok: int, padding_tok: int, memory_idx: int, num_layers: int, embd_dim: int):
        self.device = device
        self.start_tok = start_tok
        self.end_tok = end_tok
        self.padding_tok = padding_tok
        self.memory_idx = memory_idx
        self.embd_size = embd_dim
        self.num_layers = num_layers
        # cached_activations: list of [1, L, E] tensors
        self.cached_activations = [torch.zeros(1, 0, self.embd_size).to(self.device) for _ in range(num_layers + 1)]
        self.out_idx = torch.LongTensor([start_tok]).to(self.device)
        self.out_logprobs = torch.FloatTensor([0]).to(self.device)
        self.length = 0

    def seq_end(self):
        return self.out_idx.view(-1)[-1] == self.end_tok

    def logprob(self):
        return self.out_logprobs.mean().item()

    def sort_key(self):
        return -self.logprob()

    def prob(self):
        return self.out_logprobs.mean().exp().item()

    def __len__(self):
        return self.length

    def extend(self, idx, logprob):
        ret = Hypothesis(self.device, self.start_tok, self.end_tok, self.padding_tok, self.memory_idx, self.num_layers, self.embd_size)
        ret.cached_activations = [item.clone() for item in self.cached_activations]
        ret.length = self.length + 1
        ret.out_idx = torch.cat([self.out_idx, torch.LongTensor([idx]).to(self.device)], dim=0)
        ret.out_logprobs = torch.cat([self.out_logprobs, torch.FloatTensor([logprob]).to(self.device)], dim=0)
        return ret


def next_token_batch(
    hyps: List[Hypothesis],
    memory: torch.Tensor,
    memory_mask: torch.BoolTensor,
    decoders: nn.ModuleList,
    embd: nn.Embedding
):
    """批量解码下一个 token"""
    N = len(hyps)
    offset = len(hyps[0])

    # 获取每个假设的最后一个 token
    last_toks = torch.stack([item.out_idx[-1] for item in hyps])
    # N, 1, E
    tgt = embd(last_toks).unsqueeze(1)

    # 获取对应的 memory
    memory = torch.stack([memory[idx, :, :] for idx in [item.memory_idx for item in hyps]], dim=0)

    for l, layer in enumerate(decoders):
        # 合并缓存的激活
        combined_activations = torch.cat([item.cached_activations[l] for item in hyps], dim=0)
        combined_activations = torch.cat([combined_activations, tgt], dim=1)
        for i in range(N):
            hyps[i].cached_activations[l] = combined_activations[i:i+1, :, :]
        # Self-attention
        tgt = tgt + layer.self_attn(layer.norm1(tgt), layer.norm1(combined_activations), layer.norm1(combined_activations), q_offset=offset)[0]
        # Cross-attention
        tgt = tgt + layer.multihead_attn(layer.norm2(tgt), memory, memory, key_padding_mask=memory_mask, q_offset=offset)[0]
        # FFN
        tgt = tgt + layer._ff_block(layer.norm3(tgt))

    for i in range(N):
        hyps[i].cached_activations[len(decoders)] = torch.cat([hyps[i].cached_activations[len(decoders)], tgt[i:i+1, :, :]], dim=1)

    return tgt.squeeze(1)


class OCR(nn.Module):
    """MIT 48px OCR 模型（原生实现）"""
    def __init__(self, dictionary, max_len):
        super().__init__()
        self.max_len = max_len
        self.dictionary = dictionary
        self.dict_size = len(dictionary)
        embd_dim = 320
        nhead = 4

        self.backbone = ConvNext_FeatureExtractor(48, 3, embd_dim)

        # Encoder layers
        self.encoders = nn.ModuleList()
        for i in range(4):
            encoder = nn.TransformerEncoderLayer(embd_dim, nhead, dropout=0, batch_first=True, norm_first=True)
            encoder.self_attn = XposMultiheadAttention(embd_dim, nhead, self_attention=True)
            self.encoders.append(encoder)

        # Decoder layers
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

    def forward(self, img, char_idx, decoder_mask, encoder_mask):
        memory = self.backbone(img)
        memory = einops.rearrange(memory, 'N C 1 W -> N W C')
        for layer in self.encoders:
            memory = transformer_encoder_forward(layer, memory, src_key_padding_mask=encoder_mask)

        N, L = char_idx.shape
        char_embd = self.embd(char_idx)
        casual_mask = generate_square_subsequent_mask(L).to(img.device)
        decoded = char_embd
        for layer in self.decoders:
            decoded = layer(decoded, memory, tgt_mask=casual_mask, tgt_key_padding_mask=decoder_mask, memory_key_padding_mask=encoder_mask)

        pred_char_logits = self.pred(self.pred1(decoded))
        color_feats = self.color_pred1(decoded)
        return (pred_char_logits,
                self.color_pred_fg(color_feats),
                self.color_pred_bg(color_feats),
                self.color_pred_fg_ind(color_feats),
                self.color_pred_bg_ind(color_feats))

    def infer_beam_batch(self, img: torch.FloatTensor, img_widths: List[int], beams_k: int = 5,
                         start_tok: int = 1, end_tok: int = 2, pad_tok: int = 0,
                         max_finished_hypos: int = 2, max_seq_length: int = 384):
        """Beam search decoding"""
        N, C, H, W = img.shape
        assert H == 48 and C == 3

        # 编码
        memory = self.backbone(img)
        memory = einops.rearrange(memory, 'N C 1 W -> N W C')

        # 计算有效特征长度
        valid_feats_length = [(x + 3) // 4 + 2 for x in img_widths]
        input_mask = torch.zeros(N, memory.size(1), dtype=torch.bool, device=img.device)
        for i, l in enumerate(valid_feats_length):
            input_mask[i, l:] = True

        # 编码器前向
        for layer in self.encoders:
            memory = transformer_encoder_forward(layer, memory, src_key_padding_mask=input_mask)

        # 初始化假设
        hypos = [Hypothesis(img.device, start_tok, end_tok, pad_tok, i, len(self.decoders), 320) for i in range(N)]

        # 第一步解码
        decoded = next_token_batch(hypos, memory, input_mask, self.decoders, self.embd)
        pred_char_logprob = self.pred(self.pred1(decoded)).log_softmax(-1)
        pred_chars_values, pred_chars_index = torch.topk(pred_char_logprob, beams_k, dim=1)

        new_hypos = []
        finished_hypos = defaultdict(list)
        for i in range(N):
            for k in range(beams_k):
                new_hypos.append(hypos[i].extend(pred_chars_index[i, k].item(), pred_chars_values[i, k].item()))
        hypos = new_hypos

        # 迭代解码
        for _ in range(max_seq_length):
            decoded = next_token_batch(hypos, memory, torch.stack([input_mask[hyp.memory_idx] for hyp in hypos]), self.decoders, self.embd)
            pred_char_logprob = self.pred(self.pred1(decoded)).log_softmax(-1)
            pred_chars_values, pred_chars_index = torch.topk(pred_char_logprob, beams_k, dim=1)

            hypos_per_sample = defaultdict(list)
            for i, h in enumerate(hypos):
                for k in range(beams_k):
                    hypos_per_sample[h.memory_idx].append(h.extend(pred_chars_index[i, k].item(), pred_chars_values[i, k].item()))

            hypos = []
            for i in hypos_per_sample.keys():
                cur_hypos = sorted(hypos_per_sample[i], key=lambda a: a.sort_key())[:beams_k + 1]
                to_added_hypos = []
                sample_done = False
                for h in cur_hypos:
                    if h.seq_end():
                        finished_hypos[i].append(h)
                        if len(finished_hypos[i]) >= max_finished_hypos:
                            sample_done = True
                            break
                    else:
                        if len(to_added_hypos) < beams_k:
                            to_added_hypos.append(h)
                if not sample_done:
                    hypos.extend(to_added_hypos)

            if len(hypos) == 0:
                break

        # 收集结果
        result = []
        for i in range(N):
            if i in finished_hypos and len(finished_hypos[i]) > 0:
                best_hyp = min(finished_hypos[i], key=lambda h: h.sort_key())
                idx = best_hyp.out_idx[1:].cpu().numpy()  # 跳过 start_tok
                prob = best_hyp.prob()
            else:
                idx = torch.LongTensor([end_tok]).numpy()
                prob = 0.0

            # 颜色预测（简化版）
            fg_pred = torch.zeros(len(idx), 3)
            bg_pred = torch.ones(len(idx), 3)
            fg_ind_pred = torch.zeros(len(idx), 2)
            bg_ind_pred = torch.zeros(len(idx), 2)

            result.append((idx, prob, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred))

        return result

    def infer_simple(self, img: torch.Tensor, beam_size: int = 5, max_length: int = 255) -> List[Tuple[str, float]]:
        """简化解码接口"""
        N, C, H, W = img.shape
        results = self.infer_beam_batch(img, [W], beams_k=beam_size, max_seq_length=max_length)
        output = []
        for idx, prob, *_ in results:
            text = ""
            for chid in idx:
                ch = self.dictionary[chid]
                if ch == "<SP>":
                    ch = " "
                if ch not in ("<S>", "</S>"):
                    text += ch
            output.append((text, prob))
        return output
