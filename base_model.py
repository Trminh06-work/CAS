from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

@dataclass
class ModelConfig:
    use_optim: bool = False
    metric: str = "rmse"
    n_splits: int = 5
    seed: int = 42

# ================================== DL Auxiliary Classes and Functions ==================================

def pick_device(prefer_mps: bool = True, prefer_cuda: bool = True) -> str:
    """Pick the best available accelerator with a slight preference ordering."""
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if prefer_mps and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, "cudnn", None) is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class CreateDataLoader:
    def __init__(
        self,
        X_train, X_test,
        y_train, y_test,
        batch_size: int = 512,
        val_size: float = 0.1,
        is_shuffle = True,
        seed: int = 42,
    ):
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size = val_size,
                shuffle = is_shuffle,
                random_state = seed,
            )
        except ValueError:
            X_val = self._empty_like(X_train)
            y_val = self._empty_like(y_train)

        # Save
        self.X_train = self._to_tensor(X_train)
        self.X_val = self._to_tensor(X_val)

        # Do not convert X_test and y_test, keep it in pd.DataFrame to plot it later
        self.X_test_tensor = self._to_tensor(X_test)

        self.y_train = self._to_tensor(y_train)
        self.y_val   = self._to_tensor(y_val)

        self.batch_size = batch_size
        self.is_shuffle = is_shuffle


    def _empty_like(self, x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x.iloc[0:0]
        x = np.asarray(x)
        return x[:0]


    def _to_tensor(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype = np.float32, copy=False)
        else:
            X = np.asarray(X, dtype = np.float32)

        if X.ndim == 1:
            X = X.reshape(-1)

        return torch.from_numpy(X)


    def create(self):
        train_dataset = CustomDataset(self.X_train, self.y_train)
        val_dataset = CustomDataset(self.X_val, self.y_val)

        # Avoid a last mini-batch of size 1 (BatchNorm crashes in train mode)
        drop_last = (len(train_dataset) > self.batch_size) and (len(train_dataset) % self.batch_size == 1)

        train_loader = DataLoader(
            train_dataset,
            batch_size = self.batch_size,
            shuffle = self.is_shuffle,
            drop_last = drop_last,
            num_workers = 4,
        )
        val_loader = None
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size = self.batch_size,
                shuffle = not self.is_shuffle,
                num_workers = 4,
            )

        return train_loader, val_loader

# ================================== BaseModel Abstract Class ==================================

class BaseModel(ABC):
    def __init__(
            self,
            df_train: pd.DataFrame = None, df_test: pd.DataFrame = None,
            config: ModelConfig = None,
    ):
        if df_train is None or df_test is None:
            raise ValueError("train or test files are None")

        self.df_train = df_train
        self.df_test = df_test

        # Split X/y (assumes last column is target)
        self.X_train = df_train.iloc[:, :-1]
        self.y_train = df_train.iloc[:, -1]
        self.X_test = df_test.iloc[:, :-1]
        self.y_test = df_test.iloc[:, -1]

        self.config = config if config is not None else ModelConfig()


    @abstractmethod
    def fit(self):
        raise NotImplementedError


    @abstractmethod
    def predict(self):
        raise NotImplementedError


