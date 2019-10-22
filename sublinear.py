import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SubLinear(nn.Module):
    #
    def __init__(self, in_features, out_features, bias=True):
        super(SubLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mapping = self.compute_mapping()
        self.weight = nn.ParameterList([
            nn.Parameter(torch.Tensor(1, r)) for r in self.mapping[1]
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.Tensor(1)) if bias else None for _ in self.mapping[1]
        ])
        self.reset_parameters()
    #
    def compute_mapping(self):
        coeffs = [1]*self.out_features + [-self.in_features]
        k = np.roots(coeffs).real[-1]
        in_ranges = np.power(k, np.arange(1,self.out_features+1))
        in_ranges = np.round(in_ranges).astype(int)
        in_ranges[-1] += self.in_features - sum(in_ranges)
        in_indices = np.zeros(self.out_features)
        in_indices[1:] = np.cumsum(in_ranges)[:-1]
        in_indices = np.round(in_indices).astype(int)
        out_indices = np.arange(self.out_features)
        return (list(in_indices), list(in_ranges), list(out_indices))
    #
    def reset_parameters(self):
        for i in range(self.out_features):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)
    #
    def forward(self, x):
        o = torch.Tensor((self.out_features))
        for (in_index, in_range, out_index) in zip(*self.mapping):
            i = x[..., in_index:in_index+in_range]
            o[out_index] = F.linear(i, self.weight[out_index], self.bias[out_index])
        return o

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

l = nn.Linear(512, 20)
s = SubLinear(512, 20)
print(count_parameters(l))
print(count_parameters(s))