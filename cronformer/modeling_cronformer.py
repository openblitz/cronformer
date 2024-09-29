from contextlib import nullcontext
from os import path, environ
from typing import Optional

import torch.nn.functional as F

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedModel, DistilBertConfig, DistilBertModel
from torch import nn

from cronformer.configuration_cronformer import CronformerConfig


# Copied from modeling_llama.py
class CronformerRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CronformerSelfAttention(nn.Module):
    def __init__(self, config: CronformerConfig):
        super(CronformerSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_dropout

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.rotary_embed = CronformerRotaryEmbedding(self.head_dim)

    def forward(self, hidden_states, position_ids):
        batch_size, seq_len, embed_dim = hidden_states.size()

        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_embed(query, position_ids)
        query_rot, key_rot = apply_rotary_pos_emb(query, key, cos, sin)

        context = (torch.backends.cuda.sdp_kernel(enable_math=False)
                   if not environ.get("ONNX_EXPORT_MODE", None)
                   else nullcontext())

        with context:
            attn_output = F.scaled_dot_product_attention(query_rot, key_rot, value, dropout_p=self.attention_dropout, is_causal=True)

        # Concatenate attention heads and project back to original dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class CronformerDecoderLayer(nn.Module):
    def __init__(self, config: CronformerConfig):
        super(CronformerDecoderLayer, self).__init__()
        self.self_attn = CronformerSelfAttention(config)
        self.cross_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, position_ids, encoder_outputs, attention_mask):
        self_attn_output = self.self_attn(self.layer_norm1(hidden_states), position_ids)
        hidden_states = hidden_states + self.dropout(self_attn_output)

        cross_attn_output, _ = self.cross_attn(self.layer_norm2(hidden_states), encoder_outputs, encoder_outputs, attn_mask=attention_mask, need_weights=False)
        hidden_states = hidden_states + self.dropout(cross_attn_output)

        feed_forward_output = self.feed_forward(self.layer_norm3(hidden_states))
        hidden_states = hidden_states + self.dropout(feed_forward_output)

        return hidden_states


CRON_DIMS = 5  # minute, hour, day, month, day_of_week


class CronformerDecoder(nn.Module):
    def __init__(self, config: CronformerConfig):
        super(CronformerDecoder, self).__init__()

        num_decoder_layers = 4

        self.config = config
        self.token_embedding = nn.Embedding(config.cron_vocab_size, config.hidden_size)
        with torch.no_grad():
            self.token_embedding.weight *= config.initializer_range
        self.decoder_layers = nn.ModuleList([
            CronformerDecoderLayer(config)
            for _ in range(num_decoder_layers)
        ])
        self.cron_head = nn.Linear(config.hidden_size, config.cron_vocab_size)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, output_ids, encoder_outputs, output_position_ids, attention_mask):
        embeddings = self.token_embedding(output_ids)

        if attention_mask is not None and len(attention_mask.shape) == 2:
            attention_mask = (attention_mask.unsqueeze(1)
                              .repeat_interleave(self.config.num_attention_heads, dim=0)
                              .repeat_interleave(output_ids.size(1), dim=1)
                              .bool())

        hidden_states = embeddings
        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(hidden_states, output_position_ids, encoder_outputs, attention_mask=attention_mask)
        hidden_states = self.norm(hidden_states)

        return self.cron_head(hidden_states)


class CronformerModel(PreTrainedModel):
    config_class = CronformerConfig

    def __init__(self, config: CronformerConfig):
        super(CronformerModel, self).__init__(config)
        self.encoder = DistilBertModel(config.to_distilbert())
        self.decoder = CronformerDecoder(config)
        self.config = config

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, output_ids, cron_dims=None, attention_mask=None):
        if cron_dims is None:
            cron_dims = list(range(CRON_DIMS))

        output_ids = output_ids[cron_dims]

        # Note: Huggingface & PyTorch have opposite conventions for the attention mask.
        encoder_outputs = self.encoder(input_ids, ~attention_mask if attention_mask is not None else None).last_hidden_state
        output_position_ids = torch.arange(0, output_ids.size(2), dtype=torch.long, device=output_ids.device).expand((output_ids.shape[1:])).contiguous()

        return self.decoder(
            output_ids.view(-1, output_ids.size(-1)),
            encoder_outputs.repeat_interleave(output_ids.size(0), dim=0).contiguous(),
            output_position_ids.repeat_interleave(output_ids.size(0), dim=0).contiguous(),
            attention_mask.repeat_interleave(output_ids.size(0), dim=0).contiguous() if attention_mask is not None else None,
        ).reshape(output_ids.shape[0], output_ids.shape[1], output_ids.shape[2], -1)

    @classmethod
    def from_distilbert(cls, config: Optional[CronformerConfig]=None):
        cronformer = cls(config)

        distilbert_config = config.to_distilbert()
        cronformer.encoder = DistilBertModel.from_pretrained('distilbert/distilbert-base-uncased', config=distilbert_config)

        return cronformer
