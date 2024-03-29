"""These are mostly linear models that are optimized by gradient descent.

This module contains contains two classes:
    1. GradNMFImputer - Standard non-negative matrix factorization solved by
       mini-batch gradient descent.
    2. LinearPerceptronImputer - Latent factors are combined via a single
       fully-connected layer. This would be strictly linear, except we apply
       a softplus activation to the output in order to ensure non-negative
       results.
"""
import torch

from .base import BaseImputer


class GradNMFImputer(BaseImputer):
    """A non-negative matrix factorization model for imputation.

    This model specifically uses mini-batch gradient descent for optimization
    rather than the multiplicative update rule.

    Parameters
    ----------
    n_rows : int
        The number of rows in the input matrix.
    n_cols : int
        The number of columns in the input matrix.
    n_factors : int
        The number of latent factors used to represent the rows and columns.
    train_batch_size : int, optional
        The number of training examples in each mini-batch. ``None``
        will use the full dataset.
    eval_batch_size : int, optional
        The number of examples in each mini-batch during evaluation. ``None``
        will use the full dataset.
    n_epochs : int, optional
        The number of epochs to train for.
    patience : int, optional
        The number of epochs to wait for early stopping.
    stopping_tol : float, optional
        The early stopping tolerance. The loss must decrease by at least this
        amount between epochs to avoid early stopping.
    loss_func : string, optional 
        The loss function to use for training. One of {MSE, s-loss}
    optimizer : torch.optim.Optimizer, optional
        The uninitialized optimizer to use. The default is
        ``torch.optim.Adam``.
    optimizer_kwargs : Dict, optional
        Keyword arguments for the optimizer.
    """
    def __init__(
        self,
        n_rows,
        n_cols,
        n_factors,
        train_batch_size=None,
        eval_batch_size=None,
        n_epochs=100,
        patience=10,
        stopping_tol=0.0001,
        loss_func="MSE",
        optimizer=None,
        optimizer_kwargs=None,
        non_negative=True
    ):
        """Initialize a GradNMFImputer"""
        super().__init__(
            n_rows=n_rows,
            n_cols=n_cols,
            n_row_factors=n_factors,
            n_col_factors=n_factors,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            n_epochs=n_epochs,
            patience=patience,
            stopping_tol=stopping_tol,
            loss_func=loss_func,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            non_negative=non_negative,
        )

    def forward(self, locs):
        """The forward pass.

        Parameters
        ----------
        locs : torch.tensor of shape (batch_size, 2)
            The indices for the rows and columns of the matrix.

        Returns
        -------
        torch.tensor of shape (batch_size,)
            The predicted values for the specified indices
        """
        row_factors = self.row_factors(locs[:, 0])
        col_factors = self.col_factors(locs[:, 1])
        pred = torch.bmm(row_factors[:, None, :], col_factors[:, :, None])
        return pred.squeeze()


    def full_reconstruct(self, X):
        """ Transform the input matrix with learned latent factors
        Multiplies latent factors to transforms all values, 
        training set and validation set

        Parameters
        ----------
        X: torch.tensor to be transformed

        Returns
        -------
        Y_recon: numpy array; the transformed matrix
        """
        # multiply row and column latent factors
        Y_recon = self.row_factors.weight @ self.col_factors.weight.T
        Y_recon = self._scaler.inverse_transform(Y_recon)

        return Y_recon.detach().cpu().numpy()
