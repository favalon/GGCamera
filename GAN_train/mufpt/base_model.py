import torch

T = torch
if torch.cuda.is_available():
    T = torch.cuda
