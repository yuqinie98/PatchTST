from torch.nn.modules import loss
import torch
import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae,mse,rmse,mape,mspe

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

class TopkMSELoss(torch.nn.Module):
    def __init__(self, topk) -> None:
        super().__init__()
        self.topk = topk
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, output, label):
        losses = self.criterion(output, label).mean(2).mean(1)
        losses = torch.topk(losses, self.topk)[0]

        return losses

class SingleStepLoss(torch.nn.Module):
    """ Compute top-k log-likelihood and mse. """

    def __init__(self, ignore_zero):
        super().__init__()
        self.ignore_zero = ignore_zero

    def forward(self, mu, sigma, labels, topk=0):
        if self.ignore_zero:
            indexes = (labels != 0)
        else:
            indexes = (labels >= 0)

        distribution = torch.distributions.normal.Normal(mu[indexes], sigma[indexes])
        likelihood = -distribution.log_prob(labels[indexes])

        diff = labels[indexes] - mu[indexes]
        se = diff * diff

        if 0 < topk < len(likelihood):
            likelihood = torch.topk(likelihood, topk)[0]
            se = torch.topk(se, topk)[0]

        return likelihood, se

def AE_loss(mu, labels, ignore_zero):
    if ignore_zero:
        indexes = (labels != 0)
    else:
        indexes = (labels >= 0)

    ae = torch.abs(labels[indexes] - mu[indexes])
    return ae
