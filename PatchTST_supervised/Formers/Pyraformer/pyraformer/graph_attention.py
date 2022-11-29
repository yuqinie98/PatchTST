"""
Test the time and CUDA memory consumption of different attention mechanisms.
"""

from typing import List
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from hierarchical_mm_tvm import graph_mm as graph_mm_tvm
import argparse
import time
import numpy as np
from math import sqrt

torch.cuda.set_device(0)
print('Using device: {}'.format(torch.cuda.get_device_name()))
import pynvml
pynvml.nvmlInit()


def get_q_k(input_size, window_size, stride, device):
    """Get the query-key index for PAM-TVM"""
    second_length = input_size // stride
    second_last = input_size - (second_length - 1) * stride
    third_start = input_size + second_length
    third_length = second_length // stride
    third_last = second_length - (third_length - 1) * stride
    max_attn = max(second_last, third_last)
    fourth_start = third_start + third_length
    fourth_length = third_length // stride
    full_length = fourth_start + fourth_length
    fourth_last = third_length - (fourth_length - 1) * stride
    max_attn = max(third_last, fourth_last)

    max_attn += window_size + 1
    mask = torch.zeros(full_length, max_attn, dtype=torch.int32, device=device) - 1

    # 按照层内、下层、上层的顺序为序列中每个q找对应的k
    # 第一层
    for i in range(input_size):
        mask[i, 0:window_size] = i + torch.arange(window_size) - window_size // 2
        # 当window在序列右端时，把它给注释掉
        mask[i, mask[i] > input_size - 1] = -1

        mask[i, -1] = i // stride + input_size
        mask[i][mask[i] > third_start - 1] = third_start - 1
    # 第二层
    for i in range(second_length):
        mask[input_size+i, 0:window_size] = input_size + i + torch.arange(window_size) - window_size // 2
        # 当window在序列左端时，置为-1
        mask[input_size+i, mask[input_size+i] < input_size] = -1
        # 当window在序列右端时，置为-1
        mask[input_size+i, mask[input_size+i] > third_start - 1] = -1

        if i < second_length - 1:
            mask[input_size+i, window_size:(window_size+stride)] = torch.arange(stride) + i * stride
        else:
            mask[input_size+i, window_size:(window_size+second_last)] = torch.arange(second_last) + i * stride

        mask[input_size+i, -1] = i // stride + third_start
        mask[input_size+i, mask[input_size+i] > fourth_start - 1] = fourth_start - 1
    # 第三层
    for i in range(third_length):
        mask[third_start+i, 0:window_size] = third_start + i + torch.arange(window_size) - window_size // 2
        # 当window在序列左端时，置为-1
        mask[third_start+i, mask[third_start+i] < third_start] = -1
        # 当window在序列右端时，置为-1
        mask[third_start+i, mask[third_start+i] > fourth_start - 1] = -1

        if i < third_length - 1:
            mask[third_start+i, window_size:(window_size+stride)] = input_size + torch.arange(stride) + i * stride
        else:
            mask[third_start+i, window_size:(window_size+third_last)] = input_size + torch.arange(third_last) + i * stride

        mask[third_start+i, -1] = i // stride + fourth_start
        mask[third_start+i, mask[third_start+i] > full_length - 1] = full_length - 1
    # 第四层
    for i in range(fourth_length):
        mask[fourth_start+i, 0:window_size] = fourth_start + i + torch.arange(window_size) - window_size // 2
        # 当window在序列左端时，置为-1
        mask[fourth_start+i, mask[fourth_start+i] < fourth_start] = -1
        # 当window在序列右端时，置为-1
        mask[fourth_start+i, mask[fourth_start+i] > full_length - 1] = -1

        if i < fourth_length - 1:
            mask[fourth_start+i, window_size:(window_size+stride)] = third_start + torch.arange(stride) + i * stride
        else:
            mask[fourth_start+i, window_size:(window_size+fourth_last)] = third_start + torch.arange(fourth_last) + i * stride

    return mask


def get_k_q(q_k_mask):
    """Get the key-query index from query-key index for PAM-TVM"""
    k_q_mask = q_k_mask.clone()
    for i in range(len(q_k_mask)):
        for j in range(len(q_k_mask[0])):
            if q_k_mask[i, j] >= 0:
                k_q_mask[i, j] = torch.where(q_k_mask[q_k_mask[i, j]] ==i )[0]
    
    return k_q_mask


def get_mask(input_size, window_size, inner_size, device):
    """Get the attention mask of PAM-Naive"""
    # Get the size of all layers
    all_size = []
    all_size.append(input_size)
    second_size = math.floor(input_size / window_size)
    all_size.append(second_size)
    third_size = math.floor(second_size / window_size)
    all_size.append(third_size)
    fourth_size = math.floor(third_size / window_size)
    all_size.append(fourth_size)

    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length, device=device)

    # Get the intra-scale mask of each scale
    inner_window = inner_size // 2
    # The first scale
    for i in range(input_size):
        left_side = max(i - inner_window, 0)
        right_side = min(i + inner_window + 1, input_size)
        mask[i, left_side:right_side] = 1
    # The second scale
    start = input_size
    for i in range(start, start + second_size):
        left_side = max(i - inner_window, start)
        right_side = min(i + inner_window + 1, start + second_size)
        mask[i, left_side:right_side] = 1
    # The third scale
    start = input_size + second_size
    for i in range(start, start + third_size):
        left_side = max(i - inner_window, start)
        right_side = min(i + inner_window + 1, start + third_size)
        mask[i, left_side:right_side] = 1
    # The fourth scale
    start = input_size + second_size + third_size
    for i in range(start, start + fourth_size):
        left_side = max(i - inner_window, start)
        right_side = min(i + inner_window + 1, start + fourth_size)
        mask[i, left_side:right_side] = 1

    # Get the inter-scale mask
    start = input_size
    for i in range(start, start + second_size):
        left_side = (i - input_size) * window_size
        if i == (start + second_size - 1):
            right_side = start
        else:
            right_side = (i - input_size + 1) * window_size
        mask[i, left_side:right_side] = 1
        mask[left_side:right_side, i] = 1
    # The third scale
    start = input_size + second_size
    for i in range(start, start + third_size):
        left_side = input_size + (i - start) * window_size
        if i == (start + third_size - 1):
            right_side = start
        else:
            right_side = input_size + (i - start + 1) * window_size
        mask[i, left_side:right_side] = 1
        mask[left_side:right_side, i] = 1
    # The fourth scale
    start = input_size + second_size + third_size
    for i in range(start, start + fourth_size):
        left_side = input_size + second_size + (i - start) * window_size
        if i == (start + fourth_size - 1):
            right_side = start
        else:
            right_side = input_size + second_size + (i - start + 1) * window_size
        mask[i, left_side:right_side] = 1
        mask[left_side:right_side, i] = 1

    mask = (1 - mask).bool()

    return mask, all_size


"""PAM"""
class GraphSelfAttention(nn.Module):
    def __init__(self, opt):
        super(GraphSelfAttention, self).__init__()
        self.normalize_before = opt.normalize_before
        self.n_head = opt.n_head
        self.d_k = opt.d_k

        self.w_qs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_ks = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_vs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(opt.d_k * opt.n_head, opt.d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.layer_norm = nn.LayerNorm(opt.d_model, eps=1e-6)
        self.dropout_attn = nn.Dropout(opt.dropout)
        self.dropout_fc = nn.Dropout(opt.dropout)
        self.seq_len = opt.seq_len
        self.window_size = opt.window_size
        self.stride_size = opt.stride_size
        self.q_k_mask = get_q_k(self.seq_len, self.window_size, self.stride_size, opt.device)
        self.k_q_mask = get_k_q(self.q_k_mask)


    def forward(self, hidden_states):
        residual = hidden_states

        hidden_states = hidden_states
        bsz, seq_len,  _ = hidden_states.size()

        q = hidden_states
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)
        q /= math.sqrt(self.d_k)

        q = q.view(bsz, seq_len, self.n_head, self.d_k)
        k = k.view(bsz, seq_len, self.n_head, self.d_k)
        q = q.float().contiguous()
        k = k.float().contiguous()
        # attn_weights.size(): (batch_size, L, num_heads, 11) 另外注意这里设置is_t1_diagonaled为False，用于q和k attention
        attn_weights = graph_mm_tvm(q, k, self.q_k_mask, self.k_q_mask, False, 0)
        attn_weights = self.dropout_attn(F.softmax(attn_weights, dim=-1))

        v = v.view(bsz, seq_len, self.n_head, self.d_k)
        v = v.float().contiguous()
        # 这里用于attention scores和v相乘，注意is_t1_diagonaled=True
        attn = graph_mm_tvm(attn_weights, v, self.q_k_mask, self.k_q_mask, True, 0)
        attn = attn.reshape(bsz, seq_len, self.n_head * self.d_k).contiguous()
        context = self.dropout_fc(self.fc(attn))
        context += residual

        if not self.normalize_before:
            context = self.layer_norm(context)

        return context


"""Multi-head self attention"""
class NormalSelfAttention(nn.Module):
    def __init__(self, opt):
        super(NormalSelfAttention, self).__init__()
        self.normalize_before = opt.normalize_before
        self.n_head = opt.n_head
        self.d_k = opt.d_k

        self.w_qs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_ks = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_vs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(opt.d_k * opt.n_head, opt.d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.layer_norm = nn.LayerNorm(opt.d_model, eps=1e-6)
        self.dropout_attn = nn.Dropout(opt.dropout)
        self.dropout_fc = nn.Dropout(opt.dropout)
        self.seq_len = opt.seq_len
        self.window_size = opt.window_size
        self.stride_size = opt.stride_size
        if opt.mask:
            self.mask, _ = get_mask(self.seq_len, self.stride_size, self.window_size, opt.device)
        else:
            self.mask = None


    def forward(self, hidden_states):
        residual = hidden_states

        hidden_states = hidden_states
        bsz, seq_len, _ = hidden_states.size()

        q = hidden_states
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)
        q /= math.sqrt(self.d_k)

        q = q.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        q = q.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()

        attn = torch.matmul(q, k.transpose(2, 3))

        if self.mask is not None:
            attn = attn.masked_fill(self.mask.unsqueeze(0).unsqueeze(1), -1e9)

        attn = self.dropout_attn(F.softmax(attn, dim=-1))
        attn = torch.matmul(attn, v).transpose(1, 2).contiguous()
        attn = attn.view(bsz, seq_len, self.n_head * self.d_k)

        context = self.dropout_fc(self.fc(attn))
        context += residual

        if not self.normalize_before:
            context = self.layer_norm(context)

        return context


"""Prob-sparse attention"""
class ProbSparseAttention(nn.Module):
    def __init__(self, opt):
        super(ProbSparseAttention, self).__init__()
        self.normalize_before = opt.normalize_before
        self.n_head = opt.n_head
        self.d_k = opt.d_k

        self.w_qs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_ks = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        self.w_vs = nn.Linear(opt.d_model, opt.n_head * opt.d_k, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(opt.d_k * opt.n_head, opt.d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.layer_norm = nn.LayerNorm(opt.d_model, eps=1e-6)
        self.dropout_attn = nn.Dropout(opt.dropout)
        self.dropout_fc = nn.Dropout(opt.dropout)
        self.seq_len = opt.seq_len
        self.factor = opt.factor

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()

        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        return context_in

    def forward(self, hidden_states):
        residual = hidden_states

        hidden_states = hidden_states
        bsz, seq_len, _ = hidden_states.size()

        q = hidden_states
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)
        q /= math.sqrt(self.d_k)

        q = q.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.d_k).transpose(1, 2)
        q = q.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()

        u = U_part = self.factor * np.ceil(np.log(seq_len)).astype('int').item() # c*ln(L_k)

        U_part = U_part if U_part<seq_len else seq_len
        u = u if u < seq_len else seq_len

        scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u) 

        # get the context
        context = self._get_initial_context(v, seq_len)
        # update the context with selected top_k queries
        context = self._update_context(context, v, scores_top, index, seq_len).transpose(1, 2).contiguous()

        context = context.view(bsz, seq_len, self.n_head * self.d_k)

        context = self.dropout_fc(self.fc(context))
        context += residual

        if not self.normalize_before:
            context = self.layer_norm(context)

        return context


def parsing():
    parser = argparse.ArgumentParser(description='Needed for graph self attention.')
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-normalize_before', type=bool, default=False)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)

    # arguments for Multiformer
    parser.add_argument('-window_size', type=int, default=3)
    parser.add_argument('-stride_size', type=int, default=25)

    # arguments for ProbSparse
    parser.add_argument('-factor', type=int, default=5)

    # arguments for full-attention
    parser.add_argument('-mask', type=int, default=0)

    parser.add_argument('-seq_len', type=int, default=1000)
    args = parser.parse_args()

    return args


def test_NSA(args, input_len):
    """Test the time and CUDA memory consumption of normal self attention."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(1)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    init_mem = meminfo.used / 1024**3

    NSA_Layer = NormalSelfAttention(args).to(args.device)
    optimizer = optim.Adam(NSA_Layer.parameters(), 1e-4)
    optimizer.zero_grad()
    hidden_state = torch.ones(4, input_len, args.d_model, dtype=torch.float32).to(args.device)
    fake_gt = torch.zeros(4, input_len, args.d_model).to(args.device)

    # Preload the layer
    result = NSA_Layer(hidden_state)
    loss = ((fake_gt  - result) ** 2).mean()
    loss.backward()
    optimizer.step()

    used_memory = 0
    start_time = time.time()
    for i in range(1000):
        result = NSA_Layer(hidden_state)
        handle = pynvml.nvmlDeviceGetHandleByIndex(1)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_memory += meminfo.used / 1024**3
        loss = ((fake_gt  - result) ** 2).mean()
        loss.backward()
        optimizer.step()

    print('NSA used average time: {} s'.format(round((time.time() - start_time) / 1000, 4)))
    used_memory = used_memory / 1000
    print('NSA used average memory: {} GB'.format(round(used_memory-init_mem, 4)))


def test_GSA(args, input_len):
    """Test the time and CUDA memory consumption of PAM."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(1)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    init_mem = meminfo.used / 1024**3

    GSA_Layer = GraphSelfAttention(args).to(args.device)
    optimizer = optim.Adam(GSA_Layer.parameters(), 1e-4)
    optimizer.zero_grad()
    hidden_state = torch.ones(4, input_len, args.d_model, dtype=torch.float32, device=args.device)
    fake_gt = torch.zeros(4, input_len, args.d_model, device=args.device)

    # Preload the layer
    result = GSA_Layer(hidden_state)
    loss = ((fake_gt  - result) ** 2).mean()
    loss.backward()
    optimizer.step()

    used_memory = 0
    repeat_times = 1000
    start_time = time.time()
    for i in range(repeat_times):
        result = GSA_Layer(hidden_state)
        handle = pynvml.nvmlDeviceGetHandleByIndex(1)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_memory += meminfo.used / 1024**3
        loss = ((fake_gt  - result) ** 2).mean()
        loss.backward()
        optimizer.step()

    print('GSA used time:{} s'.format(round((time.time() - start_time) / repeat_times, 4)))
    used_memory = used_memory / repeat_times
    print('GSA used average memory: {} GB'.format(round(used_memory-init_mem, 4)))


def test_PSA(args, input_len):
    """Test the time and CUDA memory consumption of Prob-sparse self attention."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(1)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    init_mem = meminfo.used / 1024**3

    LSA_Layer = ProbSparseAttention(args).to(args.device)
    optimizer = optim.Adam(LSA_Layer.parameters(), 1e-4)
    optimizer.zero_grad()
    hidden_state = torch.ones(4, input_len, args.d_model, dtype=torch.float32, device=args.device)
    fake_gt = torch.zeros(4, input_len, args.d_model, device=args.device)

    # Preload the layer
    result = LSA_Layer(hidden_state)
    loss = ((fake_gt  - result) ** 2).mean()
    loss.backward()
    optimizer.step()

    used_memory = 0
    repeat_times = 1000
    start_time = time.time()
    for i in range(repeat_times):
        result = LSA_Layer(hidden_state)
        handle = pynvml.nvmlDeviceGetHandleByIndex(1)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_memory += meminfo.used / 1024**3
        loss = ((fake_gt  - result) ** 2).mean()
        loss.backward()
        optimizer.step()

    print('LSA used time:{} s'.format(round((time.time() - start_time) / repeat_times, 4)))
    used_memory = used_memory / repeat_times
    print('LSA used average memory: {} GB'.format(round(used_memory-init_mem, 4)))


if __name__ == '__main__':
    args = parsing()
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    input_size = args.seq_len
    stride = args.stride_size
    second_length = input_size // stride
    third_length = second_length // stride
    fourth_length = third_length // stride
    input_len = input_size + second_length + third_length + fourth_length

    if args.mask:
        print('sequence length: {}'.format(input_len))
        test_NSA(args, input_len)
    else:
        print('sequence length: {}'.format(input_size))
        test_NSA(args, input_size)

    print('sequence length: {}'.format(input_len))
    test_GSA(args, input_len)
    print('sequence length: {}'.format(input_size))
    test_PSA(args, input_size)

