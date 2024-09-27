from os import path
from typing import Optional

import torch.nn.functional as F

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedModel, DistilBertConfig, DistilBertModel
from torch import nn

config = DistilBertConfig.from_pretrained('distilbert/distilbert-base-uncased')
encoder = DistilBertModel.from_pretrained('distilbert/distilbert-base-uncased', config=config)
output_tokenizer = Tokenizer.from_file(path.join(path.dirname(__file__), 'tokenizer.json'))
output_vocab_size = output_tokenizer.get_vocab_size()


# Copied from modeling_llama.py
class CronRotaryEmbedding(nn.Module):
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


class CronSelfAttention(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super(CronSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.dim // config.num_attention_heads
        self.attention_dropout = config.attention_dropout

        self.query = nn.Linear(config.dim, config.dim)
        self.key = nn.Linear(config.dim, config.dim)
        self.value = nn.Linear(config.dim, config.dim)
        self.out_proj = nn.Linear(config.dim, config.dim)

        self.rotary_embed = CronRotaryEmbedding(self.head_dim)

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

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            attn_output = F.scaled_dot_product_attention(query_rot, key_rot, value, dropout_p=self.attention_dropout, is_causal=True)

        # Concatenate attention heads and project back to original dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class DecoderLayer(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super(DecoderLayer, self).__init__()
        self.self_attn = CronSelfAttention(config)
        self.cross_attn = nn.MultiheadAttention(config.dim, config.num_attention_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.dim),
        )
        self.layer_norm1 = nn.LayerNorm(config.dim)
        self.layer_norm2 = nn.LayerNorm(config.dim)
        self.layer_norm3 = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, position_ids, encoder_outputs, attention_mask):
        self_attn_output = self.self_attn(self.layer_norm1(hidden_states), position_ids)
        hidden_states = hidden_states + self.dropout(self_attn_output)

        cross_attn_output, _ = self.cross_attn(self.layer_norm2(hidden_states), encoder_outputs, encoder_outputs, attn_mask=attention_mask, needs_weights=False)
        hidden_states = hidden_states + self.dropout(cross_attn_output)

        feed_forward_output = self.feed_forward(self.layer_norm3(hidden_states))
        hidden_states = hidden_states + self.dropout(feed_forward_output)

        return hidden_states


CRON_DIMS = 5  # minute, hour, day, month, day_of_week


class CronDecoder(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super(CronDecoder, self).__init__()

        num_decoder_layers = 4

        self.config = config
        self.token_embedding = nn.Embedding(output_vocab_size, config.dim)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config)
            for _ in range(num_decoder_layers)
        ])
        self.cron_heads = nn.ModuleList([
            nn.Linear(config.dim, output_vocab_size)
            for _ in range(CRON_DIMS)
        ])

    def forward(self, output_ids, encoder_outputs, output_position_ids, cron_dim, attention_mask):
        embeddings = self.token_embedding(output_ids)

        if attention_mask is not None and len(attention_mask.shape) == 2:
            attention_mask = (attention_mask.unsqueeze(1)
                              .repeat_interleave(self.config.num_attention_heads, dim=0)
                              .repeat_interleave(output_ids.size(1), dim=1)
                              .bool())

        hidden_states = embeddings
        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(hidden_states, output_position_ids, encoder_outputs, attention_mask=attention_mask)

        return self.cron_heads[cron_dim](hidden_states)


class CronformerModel(PreTrainedModel):
    config_class = DistilBertConfig

    def __init__(self, config: DistilBertConfig):
        super(CronformerModel, self).__init__(config)
        self.encoder = DistilBertModel(config)
        self.decoder = CronDecoder(config)

    def forward(self, input_ids, output_ids, cron_dims=None, attention_mask=None):
        if cron_dims is None:
            cron_dims = list(range(CRON_DIMS))

        encoder_outputs = self.encoder(input_ids, attention_mask).last_hidden_state
        output_position_ids = torch.arange(0, output_ids.size(2), dtype=torch.long, device=output_ids.device).expand((output_ids.shape[1:])).contiguous()
        logits = torch.stack(
            [self.decoder(output_ids[i], encoder_outputs, output_position_ids, i, attention_mask) for i in cron_dims],
            dim=0,
        )

        return logits

    @classmethod
    def from_distilbert(cls, config: Optional[DistilBertConfig] = None, torch_dtype: Optional[torch.dtype] = None):
        if config is None:
            config = DistilBertConfig.from_pretrained('distilbert/distilbert-base-uncased')

        cronformer = cls(config)
        cronformer.encoder = DistilBertModel.from_pretrained('distilbert/distilbert-base-uncased', config=config, torch_dtype=torch_dtype)

        return cronformer
