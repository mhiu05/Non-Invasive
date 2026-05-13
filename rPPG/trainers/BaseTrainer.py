"""Base trainer class for all rPPG model trainers."""

import argparse
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Abstract base class defining the interface for all model trainers."""

    @staticmethod
    def add_trainer_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add common trainer CLI arguments."""
        parser.add_argument("--device", type=str, default="cuda:0")
        return parser

    def __init__(self, config, data_loader_dict: dict):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.model = None
        self.optimizer = None

    def train(self, data_loader_dict: dict):
        raise NotImplementedError

    def test(self, data_loader_dict: dict):
        raise NotImplementedError

    def _save_model(self, epoch: int):
        os.makedirs(self.config.MODEL.MODEL_DIR, exist_ok=True)
        path = os.path.join(
            self.config.MODEL.MODEL_DIR,
            f"{self.config.TRAIN.MODEL_FILE_NAME}_Epoch{epoch}.pth",
        )
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model: {path}")

    def _load_model(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded model: {path}")
