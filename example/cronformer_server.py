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

    def inference(self, prompt: str):
        input_ids = np.array([self.input_tokenizer.encode(prompt).ids])
        output_ids = np.array([
            [[self.output_tokenizer.token_to_id(open_token)]]
            for open_token, _ in CRON_TOKENS
        ])

        logits = np.array(self.session.run(["logits"], {
            "input_ids": input_ids,
            "output_ids": output_ids,
        }))

        logits = logits[:, :, -1, :].reshape(-1, logits.shape[-1])
        tokens = logits.argmax(axis=1)

        value = []
        for t in tokens:
            value.append(self.output_tokenizer.id_to_token(t))

        return value

    @modal.web_endpoint(docs=True, label="cronformer", method="POST")
    def web_inference(self, body: dict):
        prompt = body["prompt"]

        return Response(
            content=json.dumps(self.inference(prompt)),
            media_type="application/json",
        )

