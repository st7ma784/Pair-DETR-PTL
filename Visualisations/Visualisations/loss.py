
import torch
import logging
def loss(one_hot,x,app=None):
    #this only works for 2d SQUARE matrices
    #so we remove rows and columns that are all zeros

    dim0index=torch.sum(one_hot,dim=0,keepdim=False)==1
    dim1index=torch.sum(one_hot,dim=1,keepdim=False)==1
    if not torch.sum(dim0index)==torch.sum(dim1index):
        return float("nan")
    if app is not None:
        logging.warning("dim0index"+str(dim0index))
        logging.warning("dim1index"+str(dim1index))
    one_hot=one_hot[dim1index][:,dim0index]
    x= x[dim1index][:,dim0index]
    if app is not None:
        logging.warning("one_hot"+str(one_hot))
        logging.warning("x"+str(x))
    # logging.warning("one_hot"+str(one_hot))
    (xi,indices)=torch.nonzero(one_hot,as_tuple=True) # not sure why this doesnt work, and seems to require LSA  for maths to worrk! 
    index=indices.clone()
    counts=torch.zeros_like(indices)
    foundself=torch.zeros_like(indices)
    while not torch.all(foundself):
        index[:]=indices[index]#### herea 
        counts[torch.logical_not(foundself)]= counts[torch.logical_not(foundself)]+1
        foundself=torch.logical_or(foundself,indices[index]==torch.arange(indices.shape[0],device=indices.device))
    values=x*one_hot
    values=values[one_hot==1]
    return torch.sum(counts*values).item()