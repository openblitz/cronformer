from os import path
from typing import Optional

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedModel, DistilBertConfig, DistilBertModel
from torch import nn

config = DistilBertConfig.from_pretrained('distilbert/distilbert-base-uncased')
encoder = DistilBertModel.from_pretrained('distilbert/distilbert-base-uncased', config=config)
output_tokenizer = Tokenizer.from_file(path.join(path.dirname(__file__), 'tokenizer.json'))
output_vocab_size = output_tokenizer.get_vocab_size()


class DecoderLayer(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(config.dim, config.num_attention_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(config.dim, config.num_attention_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.dim),
        )
        self.layer_norm1 = nn.LayerNorm(config.dim)
        self.layer_norm2 = nn.LayerNorm(config.dim)
        self.layer_norm3 = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, encoder_outputs, attention_mask):
        hidden_states = self.layer_norm1(hidden_states)
        self_attn_causal_mask = torch.triu(torch.ones(hidden_states.size(1), hidden_states.size(1)), diagonal=1).to(hidden_states.device).bool()
        self_attn_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states, attn_mask=self_attn_causal_mask, is_causal=True)
        hidden_states = hidden_states + self.dropout(self_attn_output)

        hidden_states = self.layer_norm2(hidden_states)

        cross_attn_output, _ = self.cross_attn(hidden_states, encoder_outputs, encoder_outputs, attn_mask=attention_mask)
        hidden_states = hidden_states + self.dropout(cross_attn_output)

        hidden_states = self.layer_norm3(hidden_states)
        feed_forward_output = self.feed_forward(hidden_states)
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

    def forward(self, output_ids, encoder_outputs, cron_dim, attention_mask):
        embeddings = self.token_embedding(output_ids)

        if attention_mask is not None and len(attention_mask.shape) == 2:
            attention_mask = (attention_mask.unsqueeze(1)
                              .repeat_interleave(self.config.num_attention_heads, dim=0)
                              .repeat_interleave(output_ids.size(1), dim=1)
                              .bool())

        hidden_states = embeddings
        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(hidden_states, encoder_outputs, attention_mask=attention_mask)

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
        logits = torch.stack(
            [self.decoder(output_ids[i], encoder_outputs, i, attention_mask) for i in cron_dims],
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
