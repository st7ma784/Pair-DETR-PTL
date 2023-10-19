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


from functools import partial
def outputconversion(func):
    def wrapper(x,func=func):
        x1,y1=func(x)
        output=torch.zeros_like(x)
        output[x1,y1]=1
        return output*x
    
    return partial(wrapper,func=func)


def get_all_LSA_fns():
    #returns list of all other fns in this file that take a tensor as input.
    functions=[
        MyLinearSumAssignment,
        #outputconversion(no_for_loop_MyLinearSumAssignment),
        #outputconversion(no_for_loop_triu_MyLinearSumAssignment),
        #outputconversion(no_for_loop_v2_MyLinearSumAssignment),
        #outputconversion(no_for_loop_v2_triu_MyLinearSumAssignment),
        #outputconversion(no_for_loop_v3_MyLinearSumAssignment),
        #outputconversion(no_for_loop_v3_triu_MyLinearSumAssignment),
        outputconversion(recursiveLinearSumAssignment),
        outputconversion(recursiveLinearSumAssignment_v2),
        #outputconversion(recursiveLinearSumAssignment_v3),
        #outputconversion(recursiveLinearSumAssignment_v4),
        outputconversion(linear_sum_assignment),

    ]

    return functions



def MyLinearSumAssignment(TruthTensor, maximize=True,lookahead=2):
    '''
    If Maximize is False, I'm trying to minimize the costs. 
    This means that the mask must instead make all the weights far above all the others - 'inf' kind of thing. 
    '''
 
    mask=torch.ones(TruthTensor.shape,device=TruthTensor.device)
    results=torch.zeros(TruthTensor.shape,device=TruthTensor.device)

    finder=torch.argmax if maximize else torch.argmin
    replaceval=0 if maximize else float(1e9)

    for i in range(min(TruthTensor.shape[-2:])): # number of rows
        deltas=torch.diff(torch.topk(torch.clamp(TruthTensor*mask,max=100),lookahead,dim=0,largest=maximize).values,n=lookahead-1,dim=0)
        col_index=torch.argmax(torch.abs(deltas)) # this is the column to grab,  Note this measures step so its not important to do argmin...
        row_index=finder(TruthTensor[:,col_index])
        mask[:,col_index]=replaceval #mask out the column
        mask[row_index]=replaceval
        results[row_index,col_index]=1
    return results*TruthTensor

def no_for_loop_triu_MyLinearSumAssignment(rewards:torch.Tensor,maximize=False):
    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
    remove=torch.ones_like(rewards,dtype=torch.bool).triu().unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights,dim=0).values
    Locations=comb_fn(rewards,Costs)
    col_index=final_fn(Locations,dim=1)
    return torch.arange(Locations.shape[0],device=Locations.device),col_index

def no_for_loop_MyLinearSumAssignment(rewards:torch.Tensor,maximize=False,tril=False):

    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
    remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights,dim=0).values
    #plt.show(plt.imshow(Costs.cpu().numpy()))
    
    Locations=comb_fn(rewards,Costs)
    col_index=final_fn(Locations,dim=1)

    return torch.arange(Locations.shape[0],device=Locations.device),col_index
def no_for_loop_v2_MyLinearSumAssignment(rewards:torch.Tensor,maximize=False,tril=False):
    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize] 
    remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)

    Costs=next_highest_fn(weights1,dim=1).values
    Costs2=next_highest_fn(weights2,dim=0).values

    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    col_index=final_fn(Locations,dim=1)

    return torch.arange(Locations.shape[0],device=Locations.device),col_index

def no_for_loop_v2_triu_MyLinearSumAssignment(rewards:torch.Tensor,maximize=False,tril=False):

    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
   
    remove=torch.ones_like(rewards,dtype=torch.bool).tril().unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights1,dim=1).values 
    Costs2=next_highest_fn(weights2,dim=0).values
    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    col_index=final_fn(Locations,dim=1)
    return torch.arange(Locations.shape[0],device=Locations.device),col_index
def no_for_loop_v3_MyLinearSumAssignment(rewards:torch.Tensor,maximize=False,tril=False):
    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize] 
    remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)

    Costs=next_highest_fn(weights1,dim=1).values
    Costs2=next_highest_fn(weights2,dim=0).values

    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    col_index=final_fn(Locations,dim=1)

    return torch.arange(Locations.shape[0],device=Locations.device),col_index

def no_for_loop_v3_triu_MyLinearSumAssignment(rewards:torch.Tensor,maximize=False,tril=False):

    cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(0,torch.max,torch.sub,torch.argmax))[maximize]
   
    remove=torch.ones_like(rewards,dtype=torch.bool).tril().unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    weights=rewards.unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[-1]]))
    weights1=weights.masked_fill(remove,cost_neg)#.permute(1,2,0)
    weights2=weights.masked_fill(remove.permute(1,0,2),cost_neg)#.permute(1,2,0)
    Costs=next_highest_fn(weights1,dim=1).values 
    Costs2=next_highest_fn(weights2,dim=0).values
    Cost_total=Costs+Costs2
    Locations=rewards - Cost_total/2
    col_index=final_fn(Locations,dim=1)
    return torch.arange(Locations.shape[0],device=Locations.device),col_index


def reduceLinearSumAssignment(rewards:torch.Tensor,cost_neg:torch.Tensor,next_highest_fn: Callable,remove):
    removehw,removehwT=remove
    # rewards is HW, weights is  B(H) H W 
    weights=rewards.unsqueeze(0).repeat(*tuple([rewards.shape[0]]+ [1]*len(rewards.shape)))
    #rewards is shape hw, weights is shape h w w
    weights=weights.masked_fill(removehw,cost_neg)#.permute(1,2,0)
    #draw(weights.cpu())
    Costs=next_highest_fn(weights,dim=1).values #should not be 0  
    #draw(Costs.cpu())
    #print(Costs.shape)
    weights2=rewards.T.unsqueeze(0).repeat(*tuple([rewards.shape[1]]+ [1]*len(rewards.shape)))

    weights2=weights2.masked_fill(removehwT,cost_neg)#.permute(1,2,0)
    Costs2=next_highest_fn(weights2,dim=1).values #should not be 0
    #d#raw(Costs2.cpu())
    #print(Costs2.shape)
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


def reduceLinearSumAssignment_v3(rewards:torch.Tensor,maximize=False):

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


def reduceLinearSumAssignment_v4(rewards:torch.Tensor,maximize=False):

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

def recursiveLinearSumAssignment(rewards:torch.Tensor,maximize=False,factor=1):
    cost_neg,next_highest_fn,comb_fn,final_fn=((torch.tensor(float('inf')),torch.min,torch.add,torch.argmin),(torch.tensor(float('-inf')),torch.max,torch.sub,torch.argmax))[maximize] 
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    #we need to make a mask that holds the diagonal of a H x H matrix repeated B times, and then one with the diagonal of a BxB matrix repeated H times

    removeHHB=torch.zeros((rewards.shape[0],rewards.shape[0]),dtype=torch.bool,device=rewards.device).fill_diagonal_(1).unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape) + [rewards.shape[1]]))
    removeBBH=torch.zeros((rewards.shape[1],rewards.shape[1]),dtype=torch.bool,device=rewards.device).fill_diagonal_(1).unsqueeze(-1).repeat(*tuple([1]*len(rewards.shape)+[rewards.shape[0]]))
    #rewards=rewards-  (rewards.min())
    for i in range(rewards.shape[-1]):
        cost=reduceLinearSumAssignment(rewards,cost_neg,next_highest_fn,(removeHHB,removeBBH))
        rewards=rewards - cost/factor
        col_index=final_fn(rewards,dim=1)
        #x,y=torch.arange(rewards.shape[0],device=rewards.device),col_index    
    return torch.arange(rewards.shape[0],device=rewards.device),col_index

def recursiveLinearSumAssignment_v2(rewards:torch.Tensor,maximize=False,factor=1):
    cost_neg,next_highest_fn,comb_fn,final_fn=((torch.tensor(float('inf')),torch.min,torch.add,torch.argmin),(torch.tensor(float('-inf')),torch.max,torch.sub,torch.argmax))[maximize] 
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    # remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    for i in range(rewards.shape[-1]):
        cost2=reduceLinearSumAssignment_v2(rewards,maximize=maximize)
        rewards=rewards- (cost2/factor)# can remove
        col_index=final_fn(rewards,dim=1)
        #return torch.arange(rewards.shape[0],device=rewards.device),col_index
    
    return torch.arange(rewards.shape[0],device=rewards.device),col_index

def recursiveLinearSumAssignment_v3(rewards:torch.Tensor,maximize=False,factor=1):
    final_fn=torch.argmax if maximize else torch.argmin
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    #remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    for i in range(rewards.shape[-1]):
        cost=reduceLinearSumAssignment_v3(rewards,maximize=maximize)
        rewards=rewards-(cost/factor)# can remove
        #draw(rewards.cpu())
        #? why is this suggested??? rewards,_,_=torch.svd(rewards)
        #rewards=rewards ** (rewards-cost/factor) #times here makes it very spiky! 
        col_index=final_fn(rewards,dim=1)
        #x,y=torch.arange(rewards.shape[0],device=rewards.device),col_index    
    return torch.arange(rewards.shape[0],device=rewards.device),col_index

def recursiveLinearSumAssignment_v4(rewards:torch.Tensor,maximize=False,factor=1):
    final_fn=torch.argmax if maximize else torch.argmin
    #cost_neg,next_highest_fn,comb_fn,final_fn=((1e9,torch.min,torch.add,torch.argmin),(-1e9,torch.max,torch.sub,torch.argmax))[maximize] 
    #remove=torch.zeros_like(rewards,dtype=torch.bool).fill_diagonal_(1).unsqueeze(0).repeat(*tuple([rewards.shape[-1]]+[1]*len(rewards.shape)))
    y_values=[]
    for i in range(rewards.shape[-1]):
        cost=reduceLinearSumAssignment_v3(rewards,maximize=maximize)
        rewards=rewards-(cost/factor)# can remove
        #draw(rewards.cpu())
        #? why is this suggested??? rewards,_,_=torch.svd(rewards)
        #rewards=rewards ** (rewards-cost/factor) #times here makes it very spiky! 
        col_index=final_fn(rewards,dim=1)
        #x,y=torch.arange(rewards.shape[0],device=rewards.device),col_index
        y_values.append(col_index)
    
    return torch.arange(rewards.shape[0],device=rewards.device),col_index





#run each method on the same data and compare the results
if __name__ == "__main__":
    for i,func in enumerate(get_all_LSA_fns()):
        print("method:",i)
        input=torch.rand([10,10])
        try:
            results=func(input)

        except Exception as e:
            print("method {} failed".format(i))
            print(e)
            continue
        