import copy
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """

    def __init__(self, patience=5, min_delta=0, mode="min", should_restore=True):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
                            Default: 5
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            mode (str): One of ["min", "max"]. In "min" mode, training will stop when the quantity
                        monitored has stopped decreasing; in "max" mode it will stop when the
                        quantity monitored has stopped increasing.
                            Default: "min"
            should_restore (bool): Whether to restore model weights from the epoch with the best value
                        of the monitored quantity.
                            Default: True
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.should_restore = should_restore
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_weights = None

        if self.mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', got {self.mode}")

    def __call__(self, metric, model):
        """
        Evaluates the metric and updates the early stopping state.
        Returns True if early stopping criteria is met.
        """
        if self.best_score is None:
            self.best_score = metric
            self.best_weights = copy.deepcopy(model.state_dict())
            logger.info(f"Initial best score set to {self.best_score:.4f}")
        elif self._is_improvement(metric):
            self.best_score = metric
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
            logger.info(f"New best score: {self.best_score:.4f}. Resetting patience.")
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered.")

        return self.early_stop

    def _is_improvement(self, metric):
        if self.mode == "min":
            return metric < (self.best_score - self.min_delta)
        else:
            return metric > (self.best_score + self.min_delta)

    def restore_best_weights(self, model):
        """
        Restores the model weights from the epoch with the best value of the monitored quantity.
        """
        if self.should_restore and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"Restored best weights with score: {self.best_score:.4f}")
        elif not self.should_restore:
            logger.info("Restore best weights is disabled.")
        else:
            logger.warning("No best weights to restore.")
