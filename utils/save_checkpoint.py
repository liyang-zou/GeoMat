import shutil
import torch
import numpy as np

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        """Stop after ``patience`` checks without an improvement over ``delta``.

        Parameters
        ----------
        patience : int
            Number of non-improving checks allowed.
        verbose : bool
            Print the counter after each non-improving check.
        delta : float
            Minimum loss reduction treated as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > (self.best_loss - self.delta):
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop
