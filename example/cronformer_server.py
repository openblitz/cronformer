import json

from fastapi import Response
import numpy as np
import modal
import onnxruntime
from tokenizers.tokenizers import Tokenizer

cronformer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi~=0.115.0",
        "numpy~=1.26.4",
        "onnxruntime~=1.19.2",
        "transformers~=4.42.4",
        "tokenizers~=0.19.1",
    )
)

app = modal.App("cronformer")


CRON_TOKENS = [
    ("<minute>", "</minute>"),
    ("<hour>", "</hour>"),
    ("<date>", "</date>"),
    ("<month>", "</month>"),
    ("<day_of_week>", "</day_of_week>"),
]


@app.cls(container_idle_timeout=240, image=cronformer_image)
class Model:
    @modal.build(force=False)  # Set to True when new version of Cronformer is published.
    def build(self):
        from huggingface_hub import snapshot_download

        ignore = [
            "*.safetensors",
            "*.fp16.onnx",
        ]
        snapshot_download(
            "openblitz/cronformer", ignore_patterns=ignore,
        )

    @modal.enter()
    def enter(self):
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()

        with fs.open("openblitz/cronformer/cronformer.onnx", "rb") as f:
            self.session = onnxruntime.InferenceSession(f.read())

        with fs.open("openblitz/cronformer/distilbert.json", "r") as f:
            self.input_tokenizer = Tokenizer.from_str(f.read())

        with fs.open("openblitz/cronformer/tokenizer.json", "r") as f:
            self.output_tokenizer = Tokenizer.from_str(f.read())

    def inference(self, prompt: str) -> str:
        input_ids = np.array([self.input_tokenizer.encode(prompt).ids])
        output_ids = np.array([
            [[self.output_tokenizer.token_to_id(open_token)]]
            for open_token, _ in CRON_TOKENS
        ])
        output_mask = [1] * len(output_ids)
        stop_ids = [self.output_tokenizer.token_to_id(close_token) for _, close_token in CRON_TOKENS]
        pad_token_id = self.output_tokenizer.token_to_id("<pad>")

        while sum(output_mask) > 0 and output_ids.shape[2] < 12:
            logits = np.array(self.session.run(["logits"], {
                "input_ids": input_ids,
                "output_ids": output_ids,
            }))
            logits = logits[:, :5, :, -1, :].reshape(-1, logits.shape[-1])
            predicted_ids = np.argmax(logits, axis=-1)

            new_column = np.array([[[pad_token_id]]] * len(output_ids))
            output_ids = np.concatenate((output_ids, new_column), axis=2)

            for i, prediction_component_id in enumerate(predicted_ids):
                next_token_id = prediction_component_id

                if output_mask[i] == 1:
                    output_ids[i][0][-1] = next_token_id

                if next_token_id in stop_ids:
                    output_mask[i] = 0

        return " ".join([
            self.output_tokenizer.decode([
                token_id
                for token_id in token_ids
                if token_id != pad_token_id
            ], skip_special_tokens=True) for token_ids in output_ids.squeeze().tolist()
        ])

    @modal.web_endpoint(docs=True, label="cronformer", method="POST")
    def web_inference(self, body: dict):
        prompt = body["prompt"]

        return Response(
            content=json.dumps({"cron": self.inference(prompt)}),
            media_type="application/json",
        )

