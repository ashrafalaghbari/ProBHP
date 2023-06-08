import torch
from copy import deepcopy
class EarlyStopping:
    """
    EarlyStopping is a class used to implement the early stopping strategy to prevent overfitting.
    It is used as a callback function to monitor the performance of a deep learning model during training.
    Monitor a specific metric and stop training when it stops improving.
    If the monitored metric doesn't improve for a certain number of epochs (patience),
    the training process is stopped early.

    Attributes
    ----------
    monitor : str
        Metric being monitored.
    patience : int
        Number of epochs to wait before stopping the training process if there is no improvement.
    verbose : int
        Verbosity mode.
    mode : str
        Whether to minimize or maximize the monitored metric.
    counter : int
        Number of epochs since the last improvement in the monitored metric.
    best_score : float or None
        Best score obtained so far in the monitored metric.
    early_stop : bool
        Whether the training process was stopped early due to lack of improvement in the monitored metric.
    restore_best_weights : bool
        Whether to restore the best model weights when the training process is stopped early.
    best_state_dict : dict or None
        State dictionary of the best model weights obtained so far.
    """
    def __init__(
        self,
        patience=10,
        mode='min',
        verbose=False,
        delta=0,
        path='checkpoint.pt',
        restore_best_weights=False
    ):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.mode = mode
        self.best_epoch = None
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.restore_best_weights = restore_best_weights
        self.best_state_dict = None

    def __call__(
        self,
        model,
        epoch,
        loss
    ):
        """
        Update the state of the EarlyStopping object based on the monitored metric and
        the current state of the model.

        Parameters
        ----------
        `model` : torch.nn.Module
            Current state of the model.
        `epoch` : int
            Current epoch of the training process.
        `loss` : float
            Value of the monitored metric on the validation set.


        Returns
        -------
        None
        """
        # Calculate the score based on the mode of the EarlyStopping object
        if self.mode == 'min':
            score = -loss
        else:
            score = loss

        # If the best score is None, set the best score to the current score and save the current state of the model if necessary -> 1st epoch
        if self.best_score is None:
            self.best_score = score

        # If the current score is less than the best score, increase the counter and check if early stopping is required
        # Also save the current state of the model if necessary
        elif score <  self.best_score + self.delta:
            self.counter += 1
            if self.counter == self.patience:
                self.early_stop = True
                self.best_epoch = epoch - self.patience
                self.best_score = abs(self.best_score)
                if self.verbose:
                    print(f'EarlyStopping-> best epoch: {self.best_epoch}, best score: {self.best_score}')
                if self.restore_best_weights:
                    model.load_state_dict(torch.load(self.path))

        # If the current score is better than the best score, update the best score, reset the counter and save the current state of the model if necessary
        else:
            self.best_score = score
            self.counter = 0
            self.best_state_dict = deepcopy(model.state_dict())
            if self.restore_best_weights:
                torch.save(self.best_state_dict, self.path)
