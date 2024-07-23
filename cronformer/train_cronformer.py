import argparse
import json
import os
from os import path
from typing import Optional

import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BertTokenizer

from cronformer.modeling_cronformer import CronformerModel
from cronformer.tokenization_cronformer import CronformerTokenizer


class CronExpressionDataset(Dataset):
    def __init__(self, data_file_path, input_tokenizer: PreTrainedTokenizer, output_tokenizer: CronformerTokenizer, input_sequence_length: int = 512, sequence_length: Optional[int] = None):
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.data = self.load_data(data_file_path)
        self.input_sequence_length = input_sequence_length
        self.sequence_length = sequence_length

    def load_data(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                instance = json.loads(line)
                data.append((instance['input'], instance['output']))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, output_cron = self.data[idx]
        tokenized = self.input_tokenizer(input_text, max_length=self.input_sequence_length, padding="max_length", return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0).bool()
        output_ids = torch.tensor(self.output_tokenizer.tokenize(output_cron, sequence_length=self.sequence_length))
        return input_ids, attention_mask, output_ids


def eval_batch(model: CronformerModel, valid_loader: DataLoader, device: torch.device, num_steps: int, pad_token_id: int, output_dir: str, disable_wandb: bool):
    model.eval()
    valid_loss = 0

    with torch.no_grad():
        for input_ids, attention_mask, output_ids in tqdm(valid_loader, leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output_ids = output_ids.to(device).permute(1, 0, 2).contiguous()  # [cron_dim, batch_size, sequence_length]
            logits = model(input_ids, output_ids, attention_mask=attention_mask)

            output_ids[output_ids == pad_token_id] = -100

            logits = logits[:, :, :-1].contiguous()
            output_ids = output_ids[:, :, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), output_ids.view(-1))
            valid_loss += loss.item() / len(valid_loader)

    if not disable_wandb:
        wandb.log({"valid/loss": valid_loss}, step=num_steps)
    print(f"Validation Loss {valid_loss}")

    model.save_pretrained(path.join(output_dir, f"num_steps-{num_steps}"))

    if num_steps >= 0:
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Cronformer with PyTorch")
    parser.add_argument("-p", "--project", type=str, default="cronformer", help="Wandb project name")
    parser.add_argument("-n", "--name", type=str, default="cronformer", help="Wandb run name")
    parser.add_argument("-b", "--batch-size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size for training")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to the training data file")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("-o", "--output-dir", type=str, default="./.cronformer", help="Output directory for the model")
    parser.add_argument("--input-sequence-length", type=int, default=512, help="Maximum sequence length for the input")
    parser.add_argument("-s", "--sequence-length", type=int, default=128, help="Maximum sequence length for the model")
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum number of steps to train")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Fraction of the dataset to use for validation")
    parser.add_argument("--validation-stride", type=int, default=1000, help="Number of steps between validation runs")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable logging to wandb")
    args = parser.parse_args()

    if not args.disable_wandb:
        wandb.init(
            project=args.project,
            name=args.name,
            config={
                "batch_size": args.batch_size,
                "dataset": args.dataset,
                "epoches": args.epochs,
                "learning_rate": args.learning_rate,
            },
        )

    input_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    output_tokenizer = CronformerTokenizer()
    pad_token_id = output_tokenizer.output_tokenizer.token_to_id("<pad>")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    dataset = CronExpressionDataset(
        args.dataset,
        input_tokenizer,
        output_tokenizer,
        input_sequence_length=args.input_sequence_length,
        sequence_length=args.sequence_length,
    )
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [
        len(dataset) - int(len(dataset) * args.validation_split),
        int(len(dataset) * args.validation_split),
    ])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.micro_batch_size, shuffle=False)
    model = CronformerModel.from_bert().to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    num_steps = 0
    num_grad_accumulation_steps = args.batch_size // args.micro_batch_size

    if not args.batch_size % args.micro_batch_size == 0:
        raise ValueError("Batch size should be divisible by micro batch size")

    output_dir = path.join(os.getcwd(), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in tqdm(range(args.epochs)):
        if not args.disable_wandb:
            wandb.log({"epoch": epoch}, step=num_steps)

        for input_ids, attention_mask, output_ids in tqdm(train_loader, leave=False):
            if num_steps % args.validation_stride == 0 and num_steps > 0:
                eval_batch(model, valid_loader, device, num_steps, pad_token_id, output_dir, args.disable_wandb)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output_ids = output_ids.to(device).permute(1, 0, 2).contiguous()  # [cron_dim, batch_size, sequence_length]
            batch_loss = 0

            optimizer.zero_grad()

            num_actual_grad_accumulation_steps = 0
            for i in range(0, input_ids.shape[0], args.micro_batch_size):
                num_actual_grad_accumulation_steps += 1
                logits = model(input_ids[i:i + args.micro_batch_size], output_ids[:, i:i + args.micro_batch_size], attention_mask=attention_mask[i:i + args.micro_batch_size])

                labels = output_ids[:, i:i + args.micro_batch_size].clone()
                labels[labels == pad_token_id] = -100

                logits = logits[:, :, :-1].contiguous()
                labels = labels[:, :, 1:].contiguous()
                loss = (torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),
                                                         labels.view(-1)))
                loss.backward()

                batch_loss += loss.item()
            batch_loss /= num_actual_grad_accumulation_steps

            optimizer.step()

            if not args.disable_wandb:
                wandb.log({"train/loss": batch_loss}, step=num_steps)
            print(f"Epoch {epoch}, Step {num_steps}, Loss {batch_loss}")

            num_steps += 1

            if args.max_steps > 0 and num_steps >= args.max_steps:
                print("WARNING: Maximum number of steps reached")
                break

    model.save_pretrained(path.join(output_dir, "final"))

