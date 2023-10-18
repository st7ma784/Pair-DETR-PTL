
import torch

def loss(x,one_hot):
    xi,indices=torch.nonzero(one_hot,as_tuple=True)
    index=indices.clone()
    counts=torch.zeros_like(indices)
    foundself=torch.zeros_like(indices)
    while not torch.all(foundself):
        index=indices[index]#### herea 
        counts[torch.logical_not(foundself)]= counts[torch.logical_not(foundself)]+1
        foundself=torch.logical_or(foundself,indices[index]==torch.arange(indices.shape[0],device=indices.device))
    return torch.sum(x[xi,indices]*counts)