# cronformer

Get a cron schedule from natural text

Cronformer is a encoder-decoder model fine-tuned from Bert Encoder to generate cron schedules from natural text. To simplify the attention modeling, five different unembedding heads are used for each component of a Cron expression. This reduces the effective sequence length of each forward pass.
