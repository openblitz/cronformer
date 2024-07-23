from dataclasses import dataclass

import torch
from transformers import BertTokenizer

from cronformer.modeling_cronformer import CronformerModel
from cronformer.tokenization_cronformer import CronformerTokenizer


@dataclass
class CronformerGeneration:
    completion: str
    token_ids: list[list[int]]


def generate(model: CronformerModel, prompt: str, max_tokens=10):
    input_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    output_tokenizer = CronformerTokenizer()

    input_ids = torch.tensor(input_tokenizer.encode(prompt))[None].to(model.device)
    output_ids = torch.tensor([
        [output_tokenizer.output_tokenizer.token_to_id(cron_tokens[0])]
        for cron_tokens in output_tokenizer.cron_tokens
    ]).to(model.device).unsqueeze(dim=1)
    output_mask = [1] * len(output_ids)
    stop_ids = [output_tokenizer.output_tokenizer.token_to_id(token_pair[1]) for token_pair in output_tokenizer.cron_tokens]
    pad_token_id = output_tokenizer.output_tokenizer.token_to_id("<pad>")

    token_count = 0

    while sum(output_mask) > 0 and token_count < max_tokens:
        cron_dims = [i for i, mask in enumerate(output_mask) if mask == 1]
        logits = model(input_ids, output_ids, cron_dims=cron_dims)
        predicted_ids = torch.argmax(logits, dim=-1).squeeze(dim=1)

        new_column = torch.tensor([[[pad_token_id]] for _ in range(len(output_ids))], device=model.device)
        output_ids = torch.cat((output_ids, new_column), dim=2)

        for i, prediction_component_ids in enumerate(predicted_ids):
            cron_dim = cron_dims[i]
            next_token_id = prediction_component_ids[-1]

            output_ids[cron_dim][0][-1] = next_token_id

            if next_token_id in stop_ids:
                output_mask[cron_dim] = 0

        token_count += 1

    return CronformerGeneration(
        completion=" ".join([output_tokenizer.decode([
            token_id
            for token_id in token_ids
            if token_id != pad_token_id
        ], skip_special_tokens=True) for token_ids in output_ids.squeeze().tolist()]),
        token_ids=[
            token_id
            for token_ids in output_ids.squeeze().tolist()
            for token_id in token_ids if token_id != pad_token_id
        ],
    )


