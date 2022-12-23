
import torch
import torch.nn as nn
from .core import Callback
from src.models.layers.revin import RevIN

class RevInCB(Callback):
    def __init__(self, num_features: int, eps=1e-5, 
                        affine:bool=False, denorm:bool=True):
        """        
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param denorm: if True, the output will be de-normalized

        This callback only works with affine=False.
        if affine=True, the learnable affine_weights and affine_bias are not learnt
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.denorm = denorm
        self.revin = RevIN(num_features, eps, affine)
    

    def before_forward(self): self.revin_norm()
    def after_forward(self): 
        if self.denorm: self.revin_denorm() 
        
    def revin_norm(self):
        xb_revin = self.revin(self.xb, 'norm')      # xb_revin: [bs x seq_len x nvars]
        self.learner.xb = xb_revin

    def revin_denorm(self):
        pred = self.revin(self.pred, 'denorm')      # pred: [bs x target_window x nvars]
        self.learner.pred = pred
    

