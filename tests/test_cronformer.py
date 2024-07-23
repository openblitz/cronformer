import json
import unittest
from os import path, environ

import torch

from cronformer.generation import generate
from cronformer.modeling_cronformer import CronformerModel
from cronformer.tokenization_cronformer import CronformerTokenizer
from transformers import BertTokenizer


class TestCronformer(unittest.TestCase):
    def test_cronformer(self):
        with open(path.join(path.dirname(__file__), "test_cronformer.jsonl")) as eval_file:
            evals = map(json.loads, eval_file.readlines())

        model = CronformerModel.from_pretrained(environ["CRONFORMER_MODEL_DIR"]) if environ.get("CRONFORMER_MODEL_DIR") else CronformerModel.from_bert()

        if not environ.get("CRONFORMER_MODEL_DIR"):
            print("WARNING: CRONFORMER_MODEL_DIR not set, using untrained model")

        input_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        output_tokenizer = CronformerTokenizer()
        device = torch.device("cpu")

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")

        model.to(device)

        for input, output in map(lambda entry: (entry["input"], entry["output"]), evals):
            output_ids = torch.tensor(output_tokenizer.tokenize(output))
            generation = generate(model, input)
            prediction_ids = generation.token_ids
            prediction = generation.completion

            with self.subTest(input):
                self.assertEqual(output_ids.view(-1).tolist(), prediction_ids, f"Expected {output} but got {prediction}")
