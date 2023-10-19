
import torch

def loss(one_hot,x):
    #this only works for 2d SQUARE matrices
    #so we remove rows and columns that are all zeros
    one_hot=one_hot[torch.any(one_hot,dim=1,keepdim=True),torch.any(one_hot,dim=0,keepdim=True)]
    x= x[torch.any(one_hot,dim=1,keepdim=True),torch.any(one_hot,dim=0,keepdim=True)]

    xi,indices=torch.nonzero(one_hot,as_tuple=True) # not sure why this doesnt work, and seems to require LSA  for maths to worrk! 
    index=indices.clone()
    counts=torch.zeros_like(indices)
    foundself=torch.zeros_like(indices)
    while not torch.all(foundself):
        index[:]=indices[index]#### herea 
        counts[torch.logical_not(foundself)]= counts[torch.logical_not(foundself)]+1
        foundself=torch.logical_or(foundself,indices[index]==torch.arange(indices.shape[0],device=indices.device))
    values=x[one_hot]

    return torch.sum(counts*values).item()