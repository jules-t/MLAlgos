import torch


def softmax(vector, index):
    exps = torch.exp(vector)
    numerator = exps[index]
    denominator = torch.sum(exps)
    
    return numerator / denominator