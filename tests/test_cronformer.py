import json
import unittest
from os import path

import torch
from cronformer.modeling_cronformer import CronformerModel
from cronformer.tokenization_cronformer import CronformerTokenizer
from transformers import BertTokenizer


class TestCronformer(unittest.TestCase):
    def test_cronformer(self):
        with open(path.join(path.dirname(__file__), "test_cronformer.jsonl")) as eval_file:
            evals = map(json.loads, eval_file.readlines())

        model = CronformerModel.from_bert()
        input_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        output_tokenizer = CronformerTokenizer()
        device = torch.device("cpu")

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")

        model.to(device)

        for input, output in map(lambda entry: (entry["input"], entry["output"]), evals):
            input_ids = torch.tensor(input_tokenizer.encode(input))[None].to(device)
            output_ids = torch.tensor(output_tokenizer.tokenize(output)).to(device).unsqueeze(dim=1)

            logits = model(input_ids, output_ids)
            predicted_ids = torch.argmax(logits, dim=-1).squeeze(dim=1)

            prediction = " ".join([output_tokenizer.decode(prediction_component_ids.tolist()) for prediction_component_ids in predicted_ids])

            print(f"[INPUT] {input}")
            print(f"[LABEL] {output}")
            print(f"[PRED.] {prediction}")
            print()
