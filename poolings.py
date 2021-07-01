import torch

# LSE pooling
def pool(x, mode=None):
    if mode == "lse":
        out = torch.log(torch.mean(torch.exp(x), dim=0, keepdim=True))[0]
        return out
    elif mode == "mean":
        out = torch.mean(x, dim=0, keepdim=True)[0]
        return out
    else:
        out = torch.max(x, dim=0, keepdim=True)[0]
        return out
        