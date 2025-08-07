import torch

class EarlyStopping:
    """
    Args:
        patience (int): #epochs with no improvement before stopping
        min_delta (float): minimum change to qualify as an improvement
        mode ("min"|"max"): minimise val_loss or maximise metric
    """
    def __init__(self, patience=3, min_delta=0.0, mode="min"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_score = None
        self.count      = 0
        self.should_stop= False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return

        improve = (score < self.best_score - self.min_delta) if self.mode == "min" \
                  else (score > self.best_score + self.min_delta)

        if improve:
            self.best_score = score
            self.count      = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True
