
import torch

def loss(x,one_hot):
    #this only works for 2d SQUARE matrices
    #so we pad to the biggest square matrix
    #and then take the loss
    shape=x.shape
    assert x.shape==one_hot.shape
    maxdim=max(shape)
    padded=torch.zeros((maxdim,maxdim),device=x.device)
    padded2=torch.zeros((maxdim,maxdim),device=x.device)
    padded[:shape[0],:shape[1]]=x
    x=padded
    padded2[:shape[0],:shape[1]]=one_hot
    one_hot=padded2
    xi,indices=torch.nonzero(one_hot,as_tuple=True)
    index=indices.clone()
    counts=torch.zeros_like(indices)
    foundself=torch.zeros_like(indices)
    while not torch.all(foundself):
        index[:]=indices[index]#### herea 
        counts[torch.logical_not(foundself)]= counts[torch.logical_not(foundself)]+1
        foundself=torch.logical_or(foundself,indices[index]==torch.arange(indices.shape[0],device=indices.device))
    return torch.sum(x[xi,indices]*counts)