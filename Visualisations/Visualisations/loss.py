
import torch

def loss(one_hot):
    #this only works for 2d SQUARE matrices
    #so we pad to the biggest square matrix
    #and then take the loss
    shape=one_hot.shape
    maxdim=max(shape)
    padded=torch.zeros((maxdim,maxdim),device=x.device)
    padded[:shape[0],:shape[1]]=one_hot
    one_hot=padded
    xi,indices=torch.nonzero(one_hot,as_tuple=True) # not sure why this doesnt work, and seems to require LSA  for maths to worrk! 
    index=indices.clone()
    counts=torch.zeros_like(indices)
    foundself=torch.zeros_like(indices)
    while not torch.all(foundself):
        index[:]=indices[index]#### herea 
        counts[torch.logical_not(foundself)]= counts[torch.logical_not(foundself)]+1
        foundself=torch.logical_or(foundself,indices[index]==torch.arange(indices.shape[0],device=indices.device))
    return counts