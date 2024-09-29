from transformers import PretrainedConfig, DistilBertConfig


class CronformerConfig(PretrainedConfig):
    model_type = "cronformer"
    attribute_map = {
        "attention_dropout": "attention_probs_dropout_prob",
        "dim": "hidden_size",
        "n_heads": "num_attention_heads",
        "n_layers": "num_hidden_layers",
        "vocab_size": "lang_vocab_size",
    }

    def __init__(
        self,
        lang_tokenizer="distilbert/distilbert-base-uncased",
        lang_vocab_size=30522,
        cron_vocab_size=78,
        hidden_size=768,
        num_hidden_enc_layers=6,
        num_hidden_dec_layers=4,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_dropout=0.1,
        max_seq_length=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=0,
        is_encoder_decoder=True,
        tie_word_embeddings=False,
        **kwargs
    ):
        self.lang_tokenizer = lang_tokenizer
        self.lang_vocab_size = lang_vocab_size
        self.cron_vocab_size = cron_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_enc_layers = num_hidden_enc_layers
        self.num_hidden_dec_layers = num_hidden_dec_layers
        self.num_hidden_layers = num_hidden_enc_layers + num_hidden_dec_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_seq_length
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(**kwargs, pad_token_id=pad_token_id)

    def to_distilbert(self):
        return DistilBertConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            dim=self.hidden_size,
            num_hidden_layers=self.num_hidden_enc_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_dim=self.intermediate_size,
            hidden_act=self.hidden_act,
            attention_dropout=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            dropout=self.hidden_dropout_prob,
        )