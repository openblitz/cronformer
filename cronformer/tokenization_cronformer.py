from os import path
from typing import Optional

from tokenizers import Tokenizer
from transformers.tokenization_utils_base import TextInput


CRON_TOKENS = [
    ["<minute>", "</minute>"],
    ["<hour>", "</hour>"],
    ["<date>", "</date>"],
    ["<month>", "</month>"],
    ["<day_of_week>", "</day_of_week>"],
]


class CronformerTokenizer:
    cron_tokens = CRON_TOKENS

    def __init__(self, vocab_file=None, **kwargs):
        self.output_tokenizer = Tokenizer.from_file(vocab_file or path.join(path.dirname(__file__), 'tokenizer.json'))
        self.output_vocab_size = self.output_tokenizer.get_vocab_size()

        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return self.output_vocab_size

    def tokenize(self, text: list["TextInput"], sequence_length: Optional[int] = None):
        minutes, hours, dates, months, days_of_week = text.split(" ")
        pad_token_id = self.output_tokenizer.token_to_id("<pad>")

        tokens = [
            self.output_tokenizer.encode(start_token + value + end_token).ids
            for (start_token, end_token), value in zip(CRON_TOKENS, [minutes, hours, dates, months, days_of_week])
        ]
        max_len = max(len(token) for token in tokens)

        if sequence_length and max_len > sequence_length:
            raise ValueError(f"Sequence length {sequence_length} is too short for the tokens")
        if sequence_length:
            max_len = sequence_length

        tokens = [token + [pad_token_id] * (max_len - len(token)) for token in tokens]

        return tokens

    def decode(self, tokens: list[int], skip_special_tokens=True):
        return self.output_tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
