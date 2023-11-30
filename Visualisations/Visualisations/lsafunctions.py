import numpy as np
from typing import Callable

import torch
from functools import reduce
'''
This is research code... it is not clean and it is not commented

If you wish to use it for TPUs, I strongly recommend you refactor your code to use this style of function factory.
 Otherwise your runs will be very slow.

This code is a copy of the LSA methods in the LSA notebook, but with the following changes:
modified to return 1-hot tensors * input so we get a sense of values returned.

'''
from scipy.optimize import linear_sum_assignment
import logging

from functools import partial
def outputconversion(func): #converts the output of a function back to 1-hot tensor
    def wrapper(*args, **kwargs):
        func=kwargs.pop("func")
        args=list(args)
        x=args.pop(0)
        output=torch.zeros_like(x)

        x1,y1=func(x, *args, **kwargs)
        try:
            output[x1,y1]=1
        except:
            output[y1,x1]=1
        return output
    
    return partial(wrapper,func=func)

def forcehigh(func):
    def wrapper(*args, **kwargs):
        func=kwargs.pop("func")
        args=list(args)
        x=args.pop(0)
        output=func(x, *args, **kwargs)
        output=torch.nonzero(output,as_tuple=True)
        return output
    
    return partial(wrapper,func=func)

def doFlip(func):
    #MyLSA works well on 300,20, but not on 20,300
    def wrapper(*args, **kwargs):
        func=kwargs.pop("func")
        args=list(args)
        x=args.pop(0)
        out= func(x.T, *args,**kwargs).T if x.shape[0]<x.shape[1] else func(x,*args,**kwargs)
        return out
    return partial(wrapper,func=func)

def get_all_LSA_fns():
    #returns list of all other fns in this file that take a tensor as input.
    functions={
        "my function": MyLSA,
        #outputconversion(no_for_loop_MyLinearSumAssignment),
        #outputconversion(no_for_loop_triu_MyLinearSumAssignment),
        #outputconversion(no_for_loop_v2_MyLinearSumAssignment),
        #outputconversion(no_for_loop_v2_triu_MyLinearSumAssignment),
        #outputconversion(no_for_loop_v3_MyLinearSumAssignment),
        #outputconversion(no_for_loop_v3_triu_MyLinearSumAssignment),
        "recursive fn":outputconversion(recursiveLinearSumAssignment),
        "recursive fn2 ":outputconversion(recursiveLinearSumAssignment_v2),
        "recursive fn5":recursiveLinearSumAssignment_v5,
        #outputconversion(recursiveLinearSumAssignment_v3),
        #outputconversion(recursiveLinearSumAssignment_v4),
        "stock":outputconversion(linear_sum_assignment),

    }

    return functions



def MyLSA(TruthTensor, maximize=True,lookahead=2):
    '''
    If Maximize is False, I'm trying to minimize the costs. 
    This means that the mask must instead make all the weights far above all the others - 'inf' kind of thing. 
    '''
    #assert truthtensor is 2d and nonzero
    # assert len(TruthTensor.shape)==2
    # assert TruthTensor.shape[0]>0 and TruthTensor.shape[1]>0
    # assert lookahead>0
    # assert torch.sum(TruthTensor==0)==0

    mask=torch.zeros(TruthTensor.shape,device=TruthTensor.device,dtype=torch.int8)
    results=torch.zeros_like(TruthTensor)

    finder=torch.argmax if maximize else torch.argmin
    
    #subtract the min value from all values so that the min value is 0
    TruthTensor=TruthTensor-torch.min(torch.min(TruthTensor,dim=1,keepdim=True).values,dim=0).values
    replaceval=torch.tensor([float(-1)]) if maximize else torch.max(TruthTensor).to(dtype=torch.float32)+1
    replaceval=replaceval.to(TruthTensor.device)
    dimsizes=torch.tensor(TruthTensor.shape)
    #select index of the smallest value
    bigdim=torch.argmax(dimsizes)   # 0 
    small_dim=1-bigdim          # 1

    for i in range(TruthTensor.shape[small_dim]): # number of rows 
        #print("masked input is: ")
        #draw(torch.where(mask==0,TruthTensor,replaceval))
        array=torch.where(mask==0,TruthTensor,replaceval)
        deltas=torch.diff(torch.topk(array,lookahead,dim=bigdim,largest=maximize).values,n=lookahead-1,dim=bigdim).squeeze()
        #print(deltas)
        #draw(deltas.unsqueeze(1))
        col_index=torch.argmax(torch.abs(deltas)) # this is the column to grab,  Note this measures step so its not important to do argmin...
        #print(str(col_index.item()) + " selected ")
        if small_dim==1:
            row_index=finder(array[:,col_index]) 
            results[row_index,col_index]=1
            mask[:,col_index]=1 #mask out the column 
            mask[row_index]=1
        else: 
            row_index=finder(array[col_index])
            #now we have to swap the row and column index 
            # results[col_index,row_index]=1
            results[col_index,row_index]=1

            mask[:,row_index]=1 #mask out the column 
            mask[col_index]=1
        #results[row_index,col_index]=1
        #print("mask is now")
        #draw(mask)
    return results

def MyLinearSumAssignment(TruthTensor, maximize=True,lookahead=2):
    return MyLSA(TruthTensor, maximize=maximize,lookahead=lookahead).nonzero(as_tuple=True)

def no_for_loop_triu_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True):
    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
    remove=torch.ones_like(rewards,dtype=torch.bool).triu().unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights,dim=0).values
    Locations=comb_fn(rewards,Costs)
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)
    
    return torch.arange(Locations.shape[0],device=Locations.device),col_index

def no_for_loop_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):

    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
    remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights,dim=0).values
    #plt.show(plt.imshow(Costs.cpu().numpy()))
    
    Locations=comb_fn(rewards,Costs)
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)

    return torch.arange(Locations.shape[0],device=Locations.device),col_index
def no_for_loop_v2_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):
    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize] 
    remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)

    Costs=next_highest_fn(weights1,dim=1).values
    Costs2=next_highest_fn(weights2,dim=0).values

    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    #find the dim with the smallest value
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)

    return torch.arange(Locations.shape[0],device=Locations.device),col_index

def no_for_loop_v2_triu_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):

    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
   
    remove=torch.ones_like(rewards,dtype=torch.bool).tril().unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights1,dim=1).values 
    Costs2=next_highest_fn(weights2,dim=0).values
    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)
    return torch.arange(Locations.shape[0],device=Locations.device),col_index


def no_for_loop_v3_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):
    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize] 
    remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)

    Costs=next_highest_fn(weights1,dim=1).values
    Costs2=next_highest_fn(weights2,dim=0).values

    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)

    return torch.arange(Locations.shape[0],device=Locations.device),col_index

def no_for_loop_v3_triu_MyLinearSumAssignment(rewards:torch.Tensor,maximize=True,tril=False):

    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
   
    remove=torch.ones_like(rewards,dtype=torch.bool).tril().unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights1,dim=1).values 
    Costs2=next_highest_fn(weights2,dim=0).values
    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmax(dimsizes)
    col_index=final_fn(Locations,dim=dim)
    return torch.arange(Locations.shape[0],device=Locations.device),col_index


def reduceLinearSumAssignment(rewards:torch.Tensor,cost_neg:torch.Tensor,next_highest_fn: Callable,remove,dim=1):
    removehw,removehwT=remove
    if dim==0:
        removehw,removehwT=removehwT,removehw

    # rewards is HW, weights is  B(H) H W 
    weights=rewards.unsqueeze(0).repeat(*tuple([rewards.shape[0]]+ [1]*len(rewards.shape)))
    #rewards is shape hw, weights is shape h w w
    weights=weights.masked_fill(removehw,cost_neg)#.permute(1,2,0)
    #draw(weights.cpu())
    Costs=next_highest_fn(weights,dim=dim).values #should not be 0  
    #draw(Costs.cpu())
    #print(Costs.shape)
    weights2=rewards.T.unsqueeze(0).repeat(*tuple([rewards.shape[1]]+ [1]*len(rewards.shape)))

    weights2=weights2.masked_fill(removehwT,cost_neg)#.permute(1,2,0)
    Costs2=next_highest_fn(weights2,dim=dim).values #should not be 0

    Cost_total= torch.add(Costs,Costs2.T)
    return Cost_total

def reduceLinearSumAssignment_vm(rewards:torch.Tensor,cost_neg:torch.Tensor,next_highest_fn: Callable,remove:torch.Tensor):
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)
    
    Costs=next_highest_fn(weights1,dim=1).values #should not be 0  
    Costs2=next_highest_fn(weights2,dim=0).values #should not be 0

    #Cost_total=Costs+Costs2 # max,min or plus? min max seem to be worse than plus
    Cost_total= torch.add(Costs,Costs2)
    
    return Cost_total
def reduceLinearSumAssignment_v2(rewards:torch.Tensor,maximize=False):
    Topv,topi=rewards.topk(k=2,dim=1,largest=maximize)
    costs=Topv[:,0].unsqueeze(1).repeat(1,rewards.shape[-1])
    #print(costs.shape)
    one_hot=torch.zeros_like(rewards, dtype=torch.bool).scatter_(1,topi[:,0].unsqueeze(1),1)
    #draw(one_hot.to(dtype=torch.float,device="cpu"))
    costs[one_hot]=Topv[:,1]
    #draw(costs.cpu())
    topv2,topi2=rewards.topk(k=2,dim=0,largest=maximize)
    costs2=topv2[0].unsqueeze(0).repeat(rewards.shape[0],1)
    one_hot2 = torch.zeros_like(rewards, dtype=torch.bool).scatter_(0, topi2[0].unsqueeze(0), 1)
    costs2[one_hot2]=topv2[1]
    #draw(costs2.cpu())
    Cost_total= costs2+costs
    #draw(Cost_total.cpu())

    return Cost_total


def reduceLinearSumAssignment_v3(rewards:torch.Tensor,maximize=True):

    #30,32
    TotalCosts= torch.max(rewards,dim=1,keepdim=True).values + torch.max(rewards,dim=0,keepdim=True).values
    #30,32
    diffs= torch.diff(rewards.topk(k=2,dim=1,largest=maximize).values,dim=1)
    #30,1
    diffs2= torch.diff(rewards.topk(k=2,dim=0,largest=maximize).values,dim=0)
    #1,32
    one_hot=torch.nn.functional.one_hot(torch.argmax(rewards,dim=1),num_classes=rewards.shape[1])
    #30,32
    one_hot=one_hot*diffs
    #30,32
    one_hot2=torch.nn.functional.one_hot(torch.argmax(rewards,dim=0),num_classes=rewards.shape[0])
    #32,30

    one_hot2=one_hot2.T * diffs2
    deltas=one_hot+one_hot2
    totalCosts=TotalCosts+deltas
    return totalCosts


def reduceLinearSumAssignment_v4(rewards:torch.Tensor,maximize=True):

    #30,32
    TotalCosts= torch.max(rewards,dim=1,keepdim=True).values + torch.max(rewards,dim=0,keepdim=True).values
    #30,32
    #diffs= torch.diff(rewards.topk(k=2,dim=1,largest=maximize).values,dim=1)
    #30,1
    diffs2= torch.diff(rewards.topk(k=2,dim=0,largest=maximize).values,dim=0)
    #1,32
    #one_hot=torch.nn.functional.one_hot(torch.argmax(rewards,dim=1),num_classes=rewards.shape[1])
    #30,32
    #one_hot=one_hot*diffs
    #30,32
    one_hot2=torch.nn.functional.one_hot(torch.argmax(rewards,dim=0),num_classes=rewards.shape[0])
    #32,30

    one_hot2=one_hot2.T * diffs2
    #deltas=one_hot+one_hot2
    totalCosts=TotalCosts+one_hot2#deltas
    return totalCosts

def recursiveLinearSumAssignment(rewards:torch.Tensor,maximize=False,factor=0.8):
    cost_neg,next_highest_fn,comb_fn,final_fn=((torch.tensor(float('inf')),torch.min,torch.add,torch.argmin),(torch.tensor(float('-inf')),torch.max,torch.sub,torch.argmax))[maximize] 
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    #we need to make a mask that holds the diagonal of a H x H matrix repeated B times, and then one with the diagonal of a BxB matrix repeated H times
    # rewards=rewards-torch.min(torch.min(rewards,dim=1,keepdim=True).values,dim=0).values
    #^^ should make no difference.....but always worth checking! 
    #rewards=rewards-  (rewards.min())
    #col_index=None
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    bigdim=torch.argmax(dimsizes)
    small_dim=torch.argmin(dimsizes)
    output=torch.zeros_like(rewards,dtype=torch.int8)
    removeHHB=torch.zeros((rewards.shape[small_dim],rewards.shape[small_dim]),dtype=torch.bool,device=rewards.device).fill_diagonal_(1).unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape) + [rewards.shape[bigdim]]))
    removeBBH=torch.zeros((rewards.shape[bigdim],rewards.shape[bigdim]),dtype=torch.bool,device=rewards.device).fill_diagonal_(1).unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[small_dim]]))
    for i in range(10):
        cost=reduceLinearSumAssignment(rewards,cost_neg,next_highest_fn,(removeHHB,removeBBH),dim=bigdim)
        rewards=rewards - (cost/factor)
    col_index=final_fn(rewards,dim=bigdim)
    #return torch.arange(rewards.shape[0],device=rewards.device),col_index
    output=(torch.arange(rewards.shape[small_dim],device=rewards.device),col_index) if small_dim==1 else (col_index,torch.arange(rewards.shape[small_dim],device=rewards.device))
    return output

def recursiveLinearSumAssignment_v2(rewards:torch.Tensor,maximize=True,factor=1):
    cost_neg,next_highest_fn,comb_fn,final_fn=((torch.tensor(float('inf')),torch.min,torch.add,torch.argmin),(torch.tensor(float('-inf')),torch.max,torch.sub,torch.argmax))[maximize] 
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    # remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    # col_index=None
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    bigdim=torch.argmax(dimsizes)
    small_dim=torch.argmin(dimsizes)
    for i in range(min(rewards.shape[-2:])):
        cost2=reduceLinearSumAssignment_v2(rewards,maximize=maximize)
        rewards=rewards- (cost2/factor)# can remove
    col_index=final_fn(rewards,dim=bigdim)
        #return torch.arange(rewards.shape[0],device=rewards.device),col_index
    # logging.warning("small dim"+str(small_dim))
    output=(torch.arange(rewards.shape[small_dim],device=rewards.device),col_index) if small_dim==1 else (col_index,torch.arange(rewards.shape[small_dim],device=rewards.device))
    return output
def recursiveLinearSumAssignment_v5(rewards:torch.Tensor,maximize=True,factor=10):
    #create tensor of ints
    output=torch.zeros_like(rewards,dtype=torch.int8)
    #print("out1")
    #draw(output)
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    # remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    # col_index=None
    rewards=rewards.clone()
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    bigdim=torch.argmax(dimsizes)

    small_dim=torch.argmin(dimsizes)
    for i in range(10):
        cost2=reduceLinearSumAssignment_v2(rewards,maximize=maximize)
        rewards=rewards- (cost2/factor)# can remove
    
    #draw(output)
    #draw(rewards)
    cutoff=torch.topk(rewards.flatten(),rewards.shape[small_dim]+1,largest=maximize,sorted=True).values[-1]
    if maximize:
        output[(rewards>cutoff)]=1
    else:
        output[(rewards<cutoff)]=1
    return output.nonzero(as_tuple=True)


def recursiveLinearSumAssignment_v3(rewards:torch.Tensor,maximize=True,factor=1):
    final_fn=torch.argmax if maximize else torch.argmin
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    #remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    # col_index=None
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmax(dimsizes)
    for i in range(rewards.shape[-1]):
        cost=reduceLinearSumAssignment_v3(rewards,maximize=maximize)
        rewards=rewards-(cost/factor)# can remove
        #draw(rewards.cpu())
        #? why is this suggested??? rewards,_,_=torch.svd(rewards)
        #rewards=rewards ** (rewards-cost/factor) #times here makes it very spiky! 
    col_index=final_fn(rewards,dim=dim)
        #x,y=torch.arange(rewards.shape[0],device=rewards.device),col_index    
        
    return torch.arange(rewards.shape[0],device=rewards.device),col_index

def recursiveLinearSumAssignment_v4(rewards:torch.Tensor,maximize=True,factor=1):
    final_fn=torch.argmax if maximize else torch.argmin
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    #remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    #y_values=[]
    col_index=None
    dimsizes=torch.tensor(rewards.shape)
    #select index of the smallest value
    dim=torch.argmin(dimsizes)
    for i in range(rewards.shape[-1]):
        cost=reduceLinearSumAssignment_v3(rewards,maximize=maximize)
        rewards=rewards-(cost/factor)# can remove
        #draw(rewards.cpu())
        #? why is this suggested??? rewards,_,_=torch.svd(rewards)
        #rewards=rewards ** (rewards-cost/factor) #times here makes it very spiky! 
    col_index=final_fn(rewards,dim=dim)
    #x,y=torch.arange(rewards.shape[0],device=rewards.device),col_index
    #y_values.append(col_index)

    return torch.arange(rewards.shape[dim],device=rewards.device),col_index





#run each method on the same data and compare the results
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for i,func in get_all_LSA_fns().items():
        print("method:",i)
        input=torch.rand([10,10])
        try:
            results=func(input)

        except Exception as e:
            print("method {} failed".format(i))
            print(e)
            continue
        



    def LSA_loss(Tensor):
        #Take the LSA of the tensor, and then use the LSA to index the tensor
        
        #return sum of Tensor(LSA(Tensor)) - sum of Tensor[!LSA(Tensor)] 
        #so we need to get the indices of the LSA
        indices=MyLSA(Tensor)
        #assert that indices is 2d and has the same number of elements as the input tensor and is boolean
        assert len(indices.shape)==2
        assert indices.shape[0]==indices.shape[1]
        assert indices.shape[0]==Tensor.shape[0]
        assert indices.shape[1]==Tensor.shape[1]
        # assert indices.dtype==torch.bool
        reward=Tensor[indices.bool()]
        Cost=Tensor[torch.logical_not(indices.bool())]
        #plt.imshow(indices.cpu().numpy())
        


        output= (Tensor*indices) + (Tensor*(indices-1))
        
        return torch.sum(reward)-torch.sum(Cost), output
    


    def LSA_2loss(x):
        #this only works for 2d SQUARE matrices
        #so we remove rows and columns that are all zeros
        one_hot=MyLSA(x.clone()).int()
        print(one_hot.dtype)
        assert len(one_hot.shape)==2
        assert one_hot.shape[0]==one_hot.shape[1]
        assert one_hot.shape[0]==x.shape[0]
        assert one_hot.shape[1]==x.shape[1]
        dim0index=torch.sum(one_hot,dim=0,keepdim=False)==1
        dim1index=torch.sum(one_hot,dim=1,keepdim=False)==1
        locations=one_hot[dim1index][:,dim0index]
        # logging.warning("one_hot"+str(one_hot))

        (xi,indices)=torch.nonzero(locations,as_tuple=True) # not sure why this doesnt work, and seems to require LSA  for maths to worrk! 

        index=indices.clone()
        counts=torch.zeros_like(indices)
        foundself=torch.zeros_like(indices)
        while not torch.all(foundself):
            index[:]=indices[index]#### herea 
            counts[torch.logical_not(foundself)]= counts[torch.logical_not(foundself)]+1
            foundself=torch.logical_or(foundself,indices[index]==torch.arange(indices.shape[0],device=indices.device))
        values=x*one_hot
        positives=x* one_hot*counts
        #negatives = not one_hot * input
        negatives=x * (1-one_hot)


        # plt.subplot(1,3,1)
        # plt.imshow(positives.cpu().numpy())
        # plt.subplot(1,3,2)
        # plt.imshow(negatives.cpu().numpy())
        # plt.subplot(1,3,3)
        # plt.imshow((positives+negatives).cpu().numpy())
        # plt.show()

        values=values[one_hot==1]

        return torch.sum(counts*values).item(), positives+negatives




    size=20
    #show comparison of CELoss to LSA Loss side by side for an array like 
    #create a 2d array of size x size between -1 and 1
    input=torch.rand([size,size]) *2 -1

    # #input=torch.tensor([[1,0.2,0.3],[0.4,0.5,1],[0.3,1,0.2]])
    # input=torch.tensor([[1,0.2,0.3],[0.4,0.5,1],[1,0.3,0.2]])
    loss_stock=torch.nn.CrossEntropyLoss(reduction="none")
    labels=torch.arange(size)

    dim1_loss_stock=loss_stock(input,labels).unsqueeze(1)
    dim0_loss_stock=loss_stock(input.T,labels).unsqueeze(0)


    LossToDraw=dim1_loss_stock @ dim0_loss_stock
    dim1_loss_LSA,output=LSA_loss(input)
    LSA_2l,output2=LSA_2loss(input)

    plt.subplot(2,3,1)

    #in the first plot, draw the input

    plt.imshow(input.cpu().numpy())
    plt.subplot(2,3,2)
    plt.imshow(LossToDraw.cpu().numpy())
    plt.subplot(2,3,3)
    plt.imshow(output.cpu().numpy())
    plt.subplot(2,3,4)
    plt.imshow(output2.cpu().numpy())
    plt.subplot(2,3,5)
    matrix= (torch.sum(output,dim=0,keepdim=True)) 
    matrix1=(torch.sum(output,dim=1,keepdim=True))
    #print(torch.sum(output2,dim=1),torch.sum(output2,dim=0))
    plt.imshow((matrix.T@matrix1.T).T.cpu().numpy())



    print("dim0 loss stock: \n {} \n {}".format(dim0_loss_stock.tolist(), torch.sum(dim0_loss_stock).item()/size))

    print("dim1 loss stock: \n {} \n {}".format(dim1_loss_stock.tolist(), torch.sum(dim1_loss_stock).item()/size))
    total=torch.sum(dim0_loss_stock).item()/size+torch.sum(dim1_loss_stock).item()/size
    print("total CE loss: ",total)
    print("dim1 loss LSA: ",dim1_loss_LSA)
    print("dim1 loss LSA 2: ",LSA_2l)

    # edit titles and make them small enough to fit

    
    plt.subplot(2,3,1)

    plt.title("input Random Tensor \n size {}x{}".format(size,size),fontsize=8)
    
    plt.subplot(2,3,2)
    plt.title("CE Loss attribution of input: \n Total = {}".format(total),fontsize=8)
    plt.subplot(2,3,3)
    plt.title("LSA Loss attribution of input:\n Total = {}".format(dim1_loss_LSA),fontsize=8)
    plt.subplot(2,3,4)
    plt.title("LSA Loss attribution of with loopscaling:\n Total = {}".format(LSA_2l),fontsize=8)
    plt.subplot(2,3,5)
    plt.title("LSA2 Loss aggregate {}".format(dim1_loss_LSA),fontsize=8)
    #lt.plot(output.cpu().numpy().flatten())

    plt.show()


   