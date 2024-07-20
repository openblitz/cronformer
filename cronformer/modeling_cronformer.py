from os import path
from typing import Optional

import torch
from tokenizers import Tokenizer
from transformers import BertConfig, BertModel, PreTrainedModel
from torch import nn

config = BertConfig.from_pretrained('bert-base-uncased')
encoder = BertModel.from_pretrained('bert-base-uncased', config=config)
output_tokenizer = Tokenizer.from_file(path.join(path.dirname(__file__), 'tokenizer.json'))
output_vocab_size = output_tokenizer.get_vocab_size()


class DecoderLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, encoder_outputs, attention_mask):
        hidden_states = self.layer_norm1(hidden_states)
        self_attn_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
        hidden_states = hidden_states + self.dropout(self_attn_output)

        hidden_states = self.layer_norm2(hidden_states)

        cross_attn_output, _ = self.cross_attn(hidden_states, encoder_outputs, encoder_outputs)
        hidden_states = hidden_states + self.dropout(cross_attn_output)

        hidden_states = self.layer_norm3(hidden_states)
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = hidden_states + self.dropout(feed_forward_output)

        return hidden_states


CRON_DIMS = 5  # minute, hour, day, month, day_of_week


class CronDecoder(nn.Module):
    def __init__(self, config: BertConfig):
        super(CronDecoder, self).__init__()

        self.token_embedding = nn.Embedding(output_vocab_size, config.hidden_size)
        self.decoder_layer = DecoderLayer(config)
        self.cron_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, output_vocab_size)
            for _ in range(CRON_DIMS)
        ])

    def forward(self, output_ids, encoder_outputs, cron_dim):
        embeddings = self.token_embedding(output_ids)
        attention_mask = torch.triu(torch.ones(output_ids.size(1), output_ids.size(1)), diagonal=1).to(output_ids.device)
        hidden_states = self.decoder_layer(embeddings, encoder_outputs, attention_mask=attention_mask)
        return self.cron_heads[cron_dim](hidden_states)


class CronformerModel(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config: BertConfig):
        super(CronformerModel, self).__init__(config)
        self.encoder = BertModel(config)
        self.decoder = CronDecoder(config)

    def forward(self, input_ids, output_ids, cron_dims=None):
        if cron_dims is None:
            cron_dims = list(range(CRON_DIMS))

        encoder_outputs = self.encoder(input_ids).last_hidden_state
        logits = torch.stack(
            [self.decoder(output_ids[i], encoder_outputs, i) for i in cron_dims],
            dim=0,
        )

        return logits

    @classmethod
    def from_bert(cls, config: Optional[BertConfig] = None):
        if config is None:
            config = BertConfig.from_pretrained('bert-base-uncased')

        cronformer = cls(config)
        cronformer.encoder = BertModel.from_pretrained('bert-base-uncased', config=config)

        return cronformer
