import torch
import logging
import copy
import numpy as np
from typing import Optional


class EarlyStopping:
    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 2,
        verbose: bool = False,
        mode: str = "min",
        restore_best_weights: bool = True,
        save_path: Optional[str] = None,
        baseline: Optional[float] = None,
        start_from_epoch: int = 0,
    ):
        if mode not in ["min", "max", "auto"]:
            raise ValueError("mode must be 'min', 'max', or 'auto'")

        self.monitor = monitor
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.save_path = save_path
        self.baseline = baseline
        self.start_from_epoch = start_from_epoch

        if mode == "auto":
            if monitor.endswith(("acc", "accuracy", "auc")):
                self.mode = "max"
            else:
                self.mode = "min"
        else:
            self.mode = mode

        if self.mode == "min":
            self.monitor_op = lambda current, best: current < best - self.min_delta
            self.best_score = np.inf
        else:
            self.monitor_op = lambda current, best: current > best + self.min_delta
            self.best_score = -np.inf

        self.reset()

    def reset(self):
        self.wait = 0
        self.best_weights = None
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.should_stop = False

    def on_epoch_end(self, current_value: float, model: torch.nn.Module, epoch: int = 0) -> bool:
        if not np.isfinite(current_value):
            logging.warning(f"{self.monitor} is not finite (value = {current_value}). Skipping early stop.")
            return False

        if epoch < self.start_from_epoch:
            if self.verbose:
                logging.info(f"Warm-up epoch {epoch + 1}: Skipping early stop check.")
            return False

        if self.monitor_op(current_value, self.best_score):
            self.best_score = current_value
            self.best_epoch = epoch
            self.wait = 0

            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())

            if self.save_path:
                torch.save(self.best_weights, self.save_path)
                if self.verbose:
                    logging.info(f"Best model saved to {self.save_path}")

            if self.verbose:
                logging.info(f"Epoch {epoch + 1}: {self.monitor} improved to {current_value:.4f}")
        else:
            self.wait += 1
            if self.verbose:
                logging.info(f"Epoch {epoch + 1}: No improvement in {self.monitor} (wait={self.wait}/{self.patience})")

            if self.wait >= self.patience:
                if self.baseline is not None and not self.monitor_op(current_value, self.baseline):
                    logging.info(f"Baseline ({self.baseline}) not reached. Stopping early.")

                self.should_stop = True
                self.stopped_epoch = epoch

                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        logging.info(f"Restored best model from epoch {self.best_epoch + 1}")

                return True

        return False

    def get_best_score(self) -> Optional[float]:
        return self.best_score

    def get_best_epoch(self) -> Optional[int]:
        return self.best_epoch
