"""
Modified based on Informer.
@inproceedings{haoyietal-informer-2021,
  author    = {Haoyi Zhou and Shanghang Zhang and Jieqi Peng and Shuai Zhang and Jianxin Li and
               Hui Xiong and Wancai Zhang},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021, Virtual Conference},
  volume    = {35}, number    = {12}, pages     = {11106--11115}, publisher = {{AAAI} Press}, year      = {2021},
}
"""

import torch
import torch.nn as nn

import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeFeatureEmbedding, self).__init__()

        d_inp = 4
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

"""Embedding modules. The DataEmbedding is used by the ETT dataset for long range forecasting."""
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)

"""The CustomEmbedding is used by the electricity dataset and app flow dataset for long range forecasting."""
class CustomEmbedding(nn.Module):
    def __init__(self, c_in, d_model, temporal_size, seq_num, dropout=0.1):
        super(CustomEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = nn.Linear(temporal_size, d_model)
        self.seqid_embedding = nn.Embedding(seq_num, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark[:, :, :-1])\
            + self.seqid_embedding(x_mark[:, :, -1].long())

        return self.dropout(x)

"""The SingleStepEmbedding is used by all datasets for single step forecasting."""
class SingleStepEmbedding(nn.Module):
    def __init__(self, cov_size, num_seq, d_model, input_size, device):
        super().__init__()

        self.cov_size = cov_size
        self.num_class = num_seq
        self.cov_emb = nn.Linear(cov_size+1, d_model)
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.data_emb = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular')

        self.position = torch.arange(input_size, device=device).unsqueeze(0)
        self.position_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)], device=device)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def transformer_embedding(self, position, vector):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = position.unsqueeze(-1) / vector
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, x):
        covs = x[:, :, 1:(1+self.cov_size)]
        seq_ids = ((x[:, :, -1] / self.num_class) - 0.5).unsqueeze(2)
        covs = torch.cat([covs, seq_ids], dim=-1)
        cov_embedding = self.cov_emb(covs)
        data_embedding = self.data_emb(x[:, :, 0].unsqueeze(2).permute(0, 2, 1)).transpose(1,2)
        embedding = cov_embedding + data_embedding

        position = self.position.repeat(len(x), 1).to(x.device)
        position_emb = self.transformer_embedding(position, self.position_vec.to(x.device))

        embedding += position_emb

        return embedding
