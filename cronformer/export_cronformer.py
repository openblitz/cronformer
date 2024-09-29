import argparse
from datetime import datetime

import onnx
import onnxruntime
import torch
from os import environ

from onnxconverter_common import float16

from transformers import AutoTokenizer

from .modeling_cronformer import CronformerModel
from .tokenization_cronformer import CronformerTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    args = parser.parse_args()

    model_dir = args.model_dir
    model = CronformerModel.from_pretrained(model_dir)
    input_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    output_tokenizer = CronformerTokenizer()

    example_input_ids = torch.tensor([[input_tokenizer.cls_token_id]])
    example_output_ids = torch.tensor([[[output_tokenizer.output_tokenizer.token_to_id(cron_tokens[0])]] for cron_tokens in output_tokenizer.cron_tokens])
    model.eval()

    environ["ONNX_EXPORT_MODE"] = "1"

    torch.onnx.export(
        model,
        (example_input_ids,
        example_output_ids),
        f"{model_dir}/cronformer.onnx",
        input_names=["input_ids", "output_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "input_sequence_length"},
            "output_ids": {1: "batch_size", 2: "output_sequence_length"},
            "logits": {1: "batch_size", 2: "output_sequence_length"},
        },
        export_params=True,
        opset_version=17,
    )
    onnx_model = onnx.load(f"{model_dir}/cronformer.onnx")
    onnx.checker.check_model(onnx_model, full_check=True)

    example_dynamic_input_ids = torch.tensor([[input_tokenizer.cls_token_id, input_tokenizer.sep_token_id]])

    onnxruntime_session = onnxruntime.InferenceSession(f"{model_dir}/cronformer.onnx", providers=["CPUExecutionProvider"])
    start_time = datetime.now()
    onnxruntime_session.run(None, {"input_ids": example_dynamic_input_ids.tolist(), "output_ids": example_output_ids.tolist()})
    end_time = datetime.now()
    print(f"Ran fp32 onnx model in {(end_time - start_time).total_seconds()}s")

    model_fp16 = float16.convert_float_to_float16(onnx_model)
    onnxruntime_session = onnxruntime.InferenceSession(f"{model_dir}/cronformer_fp16.onnx", providers=["CPUExecutionProvider"])
    start_time = datetime.now()
    onnxruntime_session.run(None, {"input_ids": example_dynamic_input_ids.tolist(), "output_ids": example_output_ids.tolist()})
    end_time = datetime.now()
    print(f"Ran fp16 onnx model in {(end_time - start_time).total_seconds()}s")

    onnx.save(model_fp16, f"{model_dir}/cronformer_fp16.onnx")

