
import torch
import logging

def get_all_loss_fns():
    #create a dictionary of all loss functions

    return {
        #"LSABaseLoss":loss,     not working with wierd shapes?

        "LSAloss_v1":LSA_loss,# <<<<<<<< 
        "LSAloss_v2":LSA_2loss,
        "LSAloss_v3":LSA_3loss,
        "CombinedLosses_v1":combine_lossesv1,# <<<<<<<
        "CombinedLosses_v2":combine_lossesv2,
        "CombinedLosses_v3":combine_lossesv3,
        "CELoss":base_loss,
    }
def loss(one_hot,x):
    #this only works for 2d SQUARE matrices
    #so we remove rows and columns that are all zeros
    # print(x.shape)
    # print(one_hot.shape)
    locations=one_hot

    # #While the sum of all rows and columns is not equal to 1, we need to keep going
    # while torch.any(torch.sum(locations,dim=0,keepdim=False)!=1) or torch.any(torch.sum(locations,dim=1,keepdim=False)!=1):
    #     try:
    #         dim0index=torch.sum(one_hot,dim=0,keepdim=False)==1
    #         dim1index=torch.sum(one_hot,dim=1,keepdim=False)==1
    #         locations=one_hot[dim1index][:,dim0index]

    #         x= x[dim1index][:,dim0index]
    #     except Exception as e:
    #         print("error")
    #         # print(e)
    #         # print(one_hot)
    #         # print(x)
    #         break

    one_hot=locations.int()
    
    
    (xi,indices)=torch.nonzero(one_hot,as_tuple=True) # not sure why this doesnt work, and seems to require LSA  for maths to worrk! 
    i=torch.arange(indices.shape[0],device=indices.device)
    index=indices.clone()
    counts=torch.zeros_like(indices)
    foundself=torch.zeros_like(indices,dtype=torch.bool)
    while torch.any(torch.logical_not(foundself)) and (torch.sum(foundself)<indices.shape[0]):
        index[:]=indices[index]#### herea 
        counts=counts+torch.logical_not(foundself).int()
        foundself=torch.logical_or(foundself,indices[index]==i)
    values=x*one_hot
    values=values[one_hot==1]
    return torch.sum(counts*values) , ((counts*values.float()).unsqueeze(1))@ ((counts*values).unsqueeze(0).float())



def LSA_loss(indices,Tensor):
    #Take the LSA of the tensor, and then use the LSA to index the tensor
    #assert that indices is 2d and has the same number of elements as the input tensor and is boolean
    # assert indices.dtype==torch.bool
    reward=Tensor*indices.int()
    Cost=Tensor*torch.logical_not(indices.bool()).int()

    output= Cost-reward
    
    return torch.sum(output), output


from functools import reduce
def LSA_2loss(one_hot,x):
    #this only works for 2d SQUARE matrices
    #so we remove rows and columns that are all zeros
    one_hot=one_hot.int()
    #we have to do SOMETHING about empty rows and columns, and rows with duplicate items. 
    #we *could* remove them, but that would change the size of the matrix, and we want to keep the size the same.
    #step 1, replace all empty rows and columns with the logical or of the rows and columns with 2 or more items
    #step 2, remove rows and columns that are still empty.
    sums_of_rows=torch.sum(one_hot,dim=0,keepdim=False)
    sums_of_cols=torch.sum(one_hot,dim=1,keepdim=False)
    #step 1
    one_hot[sums_of_rows==0]=torch.sum(one_hot[sums_of_rows>1],dim=0).bool().int().unsqueeze(0)
    #replace with torch.select instead of indexing

    one_hot[:,sums_of_cols==0]=torch.sum(one_hot[:,sums_of_cols>2],dim=1).bool().int().unsqueeze(1)
    #step 2
    # sums_of_rows=torch.sum(one_hot,dim=0,keepdim=False)
    # sums_of_cols=torch.sum(one_hot,dim=1,keepdim=False)
    # one_hot=one_hot[sums_of_cols>0][:,sums_of_rows>0]
    # x=x[sums_of_cols>0][:,sums_of_rows>0]

    #print(one_hot.shape)
    #now we have a square matrix with no empty rows or columns
    (xi,indices)=torch.nonzero(one_hot,as_tuple=True)
    '''    
     # not sure why this doesnt work, and seems to require LSA  for maths to worrk! 

    index=indices.clone().sort().indices
    print(index)
    counts=torch.zeros_like(indices,dtype=torch.int)
    foundself=torch.zeros_like(indices,dtype=torch.int)
    for i in range(indices.shape[0]):
        print(counts)

        counts =torch.add(counts, 1-foundself)
        foundself=torch.logical_or(foundself.to(dtype=torch.bool),indices[index]==torch.arange(indices.shape[0],device=indices.device)).to(dtype=torch.int)
        index[:]=indices[index]#### herea??
    '''
    ## because our one_hot is now potentially 1 or more per row, we need to do this in a loop
    #we're going to iterate through our rows,
    #for each row, we're going to find the index of the first 1, and then find the index of the next 1, and then the next, and so on
    #step1, 
    rows=torch.zeros_like(one_hot,dtype=torch.bool,device=one_hot.device)
    #fill diagonal with 1s
    counts=torch.zeros(one_hot.shape[0],dtype=torch.int,device=one_hot.device)
    rows.fill_diagonal_(True)
    #step2
    for j,startrow in enumerate(rows):
        original=startrow.clone()
        results=torch.zeros((startrow.shape[0],startrow.shape[0]),dtype=torch.int,device=startrow.device)
        for i in range(startrow.shape[0]):
            startrow=one_hot[startrow].bool()
            #print("retrieved",startrow)

            #this may return multiple columns, so we need to do a logical or of all the columns
            startrow=torch.sum(startrow,dim=0,keepdim=False).bool()
            #print(startrow)
            results[i]=startrow
        #now we have a matrix of 1s and 0s, where each row is a row of the original matrix

        #results is the matrix of locations, 
        #we want to find the index of the first time that the original row is true

        #step 1: only select the colunms(s) of the original row
        #step 2: find the first row that is true
        #step 3: find the index of the first row that is true

        #step1:
        results=results[original]
        #step2:
        results=torch.argmax(results,dim=1)
        #step3:
        counts[j]=torch.sum(results)



        #print(counts)


    values=x*one_hot
    positives=x* one_hot
    positives=torch.abs(positives)*counts #.unsqueeze suggested?
    
    #positives[torch.nonzero(one_hot,as_tuple=True)]=positives[torch.nonzero(one_hot,as_tuple=True)]*counts #.unsqueeze suggested?
    negatives=x * (1-one_hot)
    values=values[one_hot==1]
    return torch.sum(torch.abs(positives)), torch.abs(positives)+negatives

def LSA_3loss(one_hot,x):
    one_hot=one_hot.int()
    locations=one_hot
 
        
    (xi,indices)=torch.nonzero(locations,as_tuple=True) # not sure why this doesnt work, and seems to require LSA  for maths to worrk! 

    index=indices.clone()
    counts=torch.zeros_like(indices)
    foundself=torch.zeros_like(indices)
    for i in range(indices.shape[0]):
        index[:]=indices[index]#### herea?? 
        counts += torch.logical_not(foundself)
        foundself=torch.logical_or(foundself,indices[index]==torch.arange(indices.shape[0],device=indices.device))
    values=x*one_hot
    positives=x* one_hot
    positives[xi,indices]=positives[xi,indices]*counts #.unsqueeze suggested?
    negatives=x * (1-one_hot)
    values=values[one_hot==1]
    return torch.sum(torch.abs(positives))+torch.sum(negatives), torch.abs(positives)+negatives


# Define some LSA Methods for testing
from typing import Callable, final

import torch

from scipy.optimize import linear_sum_assignment

from functools import partial
def outputconversion(func): #converts the output of a function back to 1-hot tensor
    def wrapper(*args, **kwargs):
        func=kwargs.pop("func")
        args=list(args)
        x=args.pop(0)
        output=torch.zeros_like(x)

        x1,y1=func(x.cpu().detach(), *args, **kwargs)
        try:
            output[x1,y1]=1
        except:
            output[y1,x1]=1
        return output
    
    return partial(wrapper,func=func)



CELoss=torch.nn.CrossEntropyLoss(reduction="none",)
def base_loss(indices,Tensor):
    labels=torch.arange(Tensor.shape[0],device=Tensor.device)
    #do CE loss in both directions
    loss1=CELoss(Tensor,labels)
    loss2=CELoss(Tensor.T,labels)
    loss= loss1+loss2
    return loss.mean(), loss1.unsqueeze(1)@loss2.unsqueeze(0)

def combine_lossesv1(indices,Tensor):
    loss,logits=base_loss(indices,Tensor)
    loss2,logits2=LSA_loss(indices,Tensor)
    return loss+loss2,logits+logits2
def combine_lossesv2(indices,Tensor):
    loss,logits=base_loss(indices,Tensor)
    loss2,logits2=LSA_2loss(indices,Tensor)
    return loss+loss2,logits+logits2
def combine_lossesv3(indices,Tensor):
    loss,logits=base_loss(indices,Tensor)
    loss2,logits2=LSA_3loss(indices,Tensor)
    return loss+loss2,logits+logits2


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