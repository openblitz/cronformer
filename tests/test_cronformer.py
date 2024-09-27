import json
import unittest
from os import path, environ

import torch

from cronformer.generation import generate
from cronformer.modeling_cronformer import CronformerModel
from cronformer.tokenization_cronformer import CronformerTokenizer
from transformers import DistilBertTokenizer


class TestCronformer(unittest.TestCase):
    def test_cronformer(self):
        with open(path.join(path.dirname(__file__), "test_cronformer.jsonl")) as eval_file:
            evals = map(json.loads, eval_file.readlines())

        model = CronformerModel.from_pretrained(environ["CRONFORMER_MODEL_DIR"]) if environ.get("CRONFORMER_MODEL_DIR") else CronformerModel.from_distilbert()

        if not environ.get("CRONFORMER_MODEL_DIR"):
            print("WARNING: CRONFORMER_MODEL_DIR not set, using untrained model")

        input_tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        output_tokenizer = CronformerTokenizer()
        device = torch.device("cpu")

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")

        model.to(device)

        def without_pad_ids(ids):
            return [id for id in ids if id != output_tokenizer.pad_token_id]

        for input, output in map(lambda entry: (entry["input"], entry["output"]), evals):
            output_alt = output.replace("*/", "0/")  # Allow non-standard cron syntax
            output_ids = without_pad_ids(torch.tensor(output_tokenizer.tokenize(output)).view(-1).tolist())
            output_alt_ids = without_pad_ids(torch.tensor(output_tokenizer.tokenize(output_alt)).view(-1).tolist())
            generation = generate(model, input)
            prediction_ids = generation.token_ids
            prediction = generation.completion

            with self.subTest(input):
                self.assertTrue(output_ids == prediction_ids or output_alt_ids == prediction_ids,
                                f"Expected {output} but got {prediction}.\nPredicted {prediction_ids}\n\nvs.\n\n{output_ids}")
