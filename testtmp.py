

from scipy.optimize import linear_sum_assignment
import torch
B=20
n_queries=80
chunks=torch.randint(1,4,(B,))
cost_Array=torch.rand((chunks.sum(),B*n_queries))
f=torch.arange(B)
g=torch.stack([f,f,f])
h=g.flatten()
#print(h)
# for chunk,costsplit in zip(chunks,cost_Array.split(chunks.tolist(),-1)):
#     print("split shape",costsplit.shape) #n, 1600
#     x1,y1=linear_sum_assignment(costsplit)
#     print(costsplit[x1,y1]) #print the index of the #chunk biggest elements

#     #this is the same as the above and is faster?
#     alt=costsplit.flatten().topk(chunk,sorted=True,largest=False).indices
#     x,y=alt//1600,alt%1600
#     print(costsplit[x,y]) #print the index of the #chunk bigestest elements

#     #this is the same as the above and is faster?

C = cost_Array.view(B, n_queries, -1).cpu()


indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(chunks.tolist(), -1))]
print([c.shape for c in C.split(chunks.tolist(), -1)])
X,Y=zip(*indices)
X=torch.cat([torch.as_tensor(x,dtype=torch.int64) for x in X])
Y=torch.cat([torch.as_tensor(y,dtype=torch.int64) for y in Y])
batch_indices = torch.cat([torch.full_like(x, i) for i, x in enumerate(X)])
