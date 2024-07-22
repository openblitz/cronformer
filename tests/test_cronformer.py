import json
import unittest
from os import path, environ

import torch
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
            input_ids = torch.tensor(input_tokenizer.encode(input))[None].to(device)
            output_ids = torch.tensor(output_tokenizer.tokenize(output)).to(device).unsqueeze(dim=1)

            logits = model(input_ids, output_ids)
            predicted_ids = torch.argmax(logits, dim=-1).squeeze(dim=1)

            prediction_ids = []
            prediction_tokens = []

            for i, prediction_component_ids in enumerate(predicted_ids):
                output_component_ids = output_ids[i].view(-1)
                try:
                    pad_token_id_index = output_component_ids.tolist().index(output_tokenizer.output_tokenizer.token_to_id("<pad>"))
                except ValueError:
                    pad_token_id_index = len(output_component_ids)
                prediction_component_ids[pad_token_id_index:] = output_tokenizer.output_tokenizer.token_to_id("<pad>")
                prediction_ids += [output_component_ids.tolist()[0]] + prediction_component_ids.tolist()[:-1]
                prediction_tokens += [output_tokenizer.decode(prediction_component_ids.tolist()[0:pad_token_id_index - 1], skip_special_tokens=True)]

            prediction = " ".join(prediction_tokens)

            if environ.get("CRONFORMER_DEBUG") == "1":
                print(f"[INPUT] {input}")
                print(f"[LABEL] {output_tokenizer.decode(output_ids.view(-1).tolist(), skip_special_tokens=False).replace('><', '> <').replace('<pad>', '')}")
                print(f"[PRED.] {prediction}")
                print()

            with self.subTest(input):
                self.assertEqual(output_ids.view(-1).tolist(), prediction_ids, f"Expected {output} but got {prediction}")
