# Performs self disclosure inference on sample input

import argparse
import csv

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from data_loading.data_loader import CSVDataLoader
from models.linear import Linear


def embed_roberta(input, mask, roberta, pool):
    hidden_states = roberta(input, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
    if pool == "mean":
        return hidden_states.mean(dim=1)
    elif pool == "cls":
        return hidden_states[:, 0, :]
    else:
        raise ValueError("Incorrect pool strategy: ", pool)


def main(csv_path, model_path, embed, max_embed_length, pool, header):
    tokenizer = None
    roberta = None
    if embed:
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        roberta = AutoModel.from_pretrained("roberta-base")

    loader = CSVDataLoader.from_data_path(csv_path, test_size=1.0, header=header)
    model = torch.load(model_path)
    model.eval()

    testloader = loader.generate_loaders(tokenizer, max_length=max_embed_length)[0]["eval"]

    results = []
    for batch in testloader:
        if embed:
            X, mask, _ = batch
            X = embed_roberta(X, mask, roberta, pool)
        else:
            X, _ = batch

        self_dis = model.get_self_dis(X)
        results += self_dis.tolist()

    results = [[result] for result in results]
    print(results)

    with open("inference_results.csv", "w") as f:
        write = csv.writer(f)
        write.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform self disclosure inference")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default=".results/measuring_language_of_self_disclosure/full_model_roberta/full_model.p",
    )
    parser.add_argument(
        "-c",
        "--csv_path",
        type=str,
        required=True,
        help="The path to the csv for input. See README.md for the required format.",
    )
    parser.add_argument(
        "-e",
        "--embed",
        action="store_true",
        default=False,
        help="Whether or not to convert input data to roberta embeddings",
    )
    parser.add_argument(
        "-l",
        "--max_embed_length",
        type=int,
        default=None,
        help="Max number of tokens per input text",
    )
    parser.add_argument(
        "-p",
        "--pool",
        type=str,
        choices=["mean", "cls"],
        help="The strategy to pool each token's final hidden state into the embedding. 'mean' will take the average whereas 'cls' will use the hidden state for the cls token",
        default="cls",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="Whether or not the CSV has a header",
        default=False,
    )
    args = parser.parse_args()
    main(args.csv_path, args.model_path, args.embed, args.max_embed_length, args.pool, args.header)
