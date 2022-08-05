from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch import distributed as dist
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from data_loading.utils import DistributedEvalSampler


class CSVDataLoader:
    """Object to generate pytorch DataLoaders for cross val and train test split"""

    def __init__(
        self,
        data,
        test_size=0.3,
        perform_split=True,
        crossfold=0,
        train_size=None,
        labels=None,
        feats=None,
        seed=0,
    ):
        print(data.columns)

        if labels is None:
            to_drop = ["index"]
            if "task_idx" in data.columns:
                to_drop += ["task_idx"]
            cols = data.drop(to_drop, axis=1).columns
            if len(cols) < 768:
                feats = cols[0]
                labels = cols[1:]
            else:
                feats = cols[:768]
                labels = cols[768:]
            if len(labels) == 0:
                labels = ["index"]
        else:
            if feats is None:
                feats = data.drop(labels + ["index"], axis=1).columns
        # print('\n\n\n\n labels: ', labels)
        if train_size is not None:
            test_size = 1 - (train_size / len(data))
            train_text, test_text, train_labels, test_labels = train_test_split(
                data[feats], data[labels], test_size=test_size, random_state=seed
            )
            folds = [{"train": (train_text, train_labels), "eval": (test_text, test_labels)}]
        elif test_size == 0:
            folds = [
                {
                    "train": (data[feats], data[labels]),
                    "eval": (pd.Series([]), pd.Series([])),
                }
            ]
        elif test_size == 1:
            folds = [
                {
                    "eval": (data[feats], data[labels]),
                    "train": (pd.Series([]), pd.Series([])),
                }
            ]
        elif crossfold > 0:
            kf = KFold(n_splits=crossfold, shuffle=True, random_state=seed)
            folds = []
            for train_idxs, test_idxs in kf.split(data):
                # print(data[feats].head())
                # print(feats)
                train_text = data[feats].iloc[train_idxs]
                test_text = data[feats].iloc[test_idxs]
                train_labels = data[labels].iloc[train_idxs, :]
                test_labels = data[labels].iloc[test_idxs, :]
                folds.append(
                    {
                        "train": (train_text, train_labels),
                        "eval": (test_text, test_labels),
                    }
                )
        elif perform_split:
            train_text, test_text, train_labels, test_labels = train_test_split(
                data[feats], data[labels], test_size=test_size, random_state=seed
            )

            folds = [{"train": (train_text, train_labels), "eval": (test_text, test_labels)}]
        else:
            folds = []

        # self.folds: list[dict[str, tuple[Any, Any]]] = folds
        self.folds: Any = folds
        self.seed = seed
        print(self.folds[0]["eval"][0])

    @classmethod
    def from_data_path(
        cls, path, test_size=0.3, crossfold=0, train_size=None, labels=None, header=True
    ):
        if header:
            data: Any = pd.read_csv(path)
        else:
            data: Any = pd.read_csv(path, header=None)
        data.dropna(inplace=True)
        data.reset_index(inplace=True)
        return cls(data, test_size, crossfold=crossfold, train_size=train_size, labels=labels)

    @classmethod
    def from_multitask_paths(
        cls,
        paths,
        test_size=0.3,
        crossfold=0,
        labels=None,
        feats=None,
        train_size=None,
        seed=0,
        norm=False,
    ):

        dfs = []
        for i, path in enumerate(paths):
            data: Any = pd.read_csv(path)
            data.dropna(inplace=True)
            data.reset_index(inplace=True)
            if norm:
                data["label"] = (data["label"] - data["label"].mean()) / data["label"].std()
            data["task_idx"] = i
            dfs.append(data)

        return cls(
            pd.concat(dfs).sample(frac=1, random_state=seed).reset_index(drop=True),
            test_size,
            crossfold=crossfold,
            labels=labels,
            feats=feats,
            train_size=train_size,
            seed=seed,
        )

    @classmethod
    def from_train_test_path(cls, train_path, test_path, **kwargs):
        train: Any = pd.read_csv(train_path)
        test: Any = pd.read_csv(test_path)
        data = train.copy().append(test.copy(), ignore_index=True)
        loader = cls(data, perform_split=False, **kwargs)
        loader.folds = [
            {
                "train": (train["text"], train["label"]),
                "eval": (test["text"], test["label"]),
            }
        ]
        # input_lengths = []
        # for fold in loader.folds:
        #     trainX, _ = fold["train"]
        #     for x in trainX.tolist():
        #         input_lengths.append(len(x))

        return loader

    def generate_loaders(self, tokenizer, max_length=None, batch_size=32):
        """Returns list of tuple of trainloader, testloader. One for each fold

        Args:
            tokenizer (transformers.Tokenizer): Needed to encode input data
            max_length (int, optional): Maximum length of encoded input data. Encoded data will be truncated or padded. Defaults to None.
            batch_size (int, optional): Defaults to 32.

        Returns:
            list(tuple(trainloader, testloader))
        """
        loaders = []
        # if max_length is None and tokenizer is not None:
        #     input_lengths = []
        #     for fold in self.folds:
        #         trainX, _ = fold["train"]
        #         testX, _ = fold["eval"]
        #         for x in trainX.tolist() + testX.tolist():
        #             encoded = tokenizer(x, add_special_tokens=True, return_tensors="pt")
        #             input_lengths.append(encoded["input_ids"].shape[1])

        #     max_length = min(120, int(np.percentile(input_lengths, 95)))
        #     print("\n\n max length: ", max_length, " actual max: ", max(input_lengths))

        for fold in self.folds:
            trainX, trainY = fold["train"]
            train_task_idx = None
            if len(trainX) > 0:
                if tokenizer:
                    train_X = tokenizer(
                        trainX.values.squeeze().tolist(),
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt",
                    )
                else:
                    if "task_idx" in trainX.columns:
                        columns = trainX.columns.drop(["task_idx"])
                        # print(
                        #     trainX["task_idx"].head(), "len: ", len(trainX["task_idx"])
                        # )
                        train_task_idx = torch.tensor(trainX["task_idx"].tolist())
                    else:
                        columns = trainX.columns
                    train_X = torch.tensor([trainX[label].tolist() for label in columns]).transpose(
                        0, 1
                    )

                train_y = torch.tensor(
                    [trainY[label].tolist() for label in trainY.columns]
                ).transpose(0, 1)
                if train_task_idx is not None:
                    # print(
                    #     "\n\n\n  sizes: ",
                    #     train_X.size(),
                    #     train_y.size(),
                    #     train_task_idx.size(),
                    # )
                    train_data = TensorDataset(train_X, train_y, train_task_idx)
                elif tokenizer is not None:
                    train_data = TensorDataset(train_X, train_y)
                else:
                    train_data = TensorDataset(train_X, train_y)

                if dist.is_initialized():
                    sampler = DistributedSampler(train_data, seed=self.seed)
                else:
                    sampler = RandomSampler(train_data)
                train_dataloader = DataLoader(
                    train_data,
                    batch_size=batch_size,
                    sampler=sampler,
                    pin_memory=True,
                )
            else:
                train_dataloader = []

            testX, testY = fold["eval"]
            test_task_idx = None
            if len(testX) > 0:
                if tokenizer:
                    # print(testX.values.squeeze().tolist())
                    test_X = tokenizer(
                        testX.values.squeeze().tolist(),
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt",
                    )
                else:
                    if "task_idx" in testX.columns:
                        columns = testX.columns.drop(["task_idx"])
                        test_task_idx = torch.tensor(testX["task_idx"].tolist())
                    else:
                        columns = testX.columns
                    test_X = torch.tensor([testX[label].tolist() for label in columns]).transpose(
                        0, 1
                    )

                if len(testY.columns) > 0:
                    test_y = torch.tensor(
                        [testY[label].tolist() for label in testY.columns]
                    ).transpose(0, 1)
                elif tokenizer is not None:
                    test_y = torch.zeros(len(test_X["input_ids"]))

                if test_task_idx is not None:

                    test_data = TensorDataset(test_X, test_y, test_task_idx)
                elif tokenizer is not None:
                    test_data = TensorDataset(test_X["input_ids"], test_X["attention_mask"], test_y)
                else:
                    test_data = TensorDataset(test_X, test_y)

                if dist.is_initialized():
                    sampler = DistributedEvalSampler(test_data, seed=self.seed)
                else:
                    sampler = RandomSampler(test_data)

                test_dataloader = DataLoader(
                    test_data,
                    batch_size=192,
                    shuffle=False,
                    pin_memory=True,
                )
            else:
                test_dataloader = []

            loaders.append({"train": train_dataloader, "eval": test_dataloader})

        return loaders
