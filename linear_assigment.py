import torch
def assign2D(C):
    #convert to pytorch the function in linear_assigment.m

       
    numRow,numCol=C.shape
    
    didFlip=numCol>numRow
    if didFlip:
        C=C.transpose(1,0)
        numRow,numCol=C.shape
    col4row=torch.zeros(numRow,dtype=torch.int64,device=C.device)
    row4col=torch.zeros(numCol,dtype=torch.int64,device=C.device)
    u=torch.zeros(numCol,device=C.device)#The dual variable for the columns
    v=torch.zeros(numRow,device=C.device)#The dual variable for the rows.
    
    #Initially, none of the columns are assigned.
    for curUnassCol in range(numCol):       
        #This finds the shortest augmenting path starting at k and returns
        #the last node in the path.
        [sink,pred,u,v]=ShortestPath(curUnassCol,u,v,C,col4row,row4col)
        #We have to remove node k from those that must be assigned.
        # i=-1
        
        # while(i!=curUnassCol):
        #     i=pred[sink]
        #     print("repeating for across \n {}  \n until we reach {}, currently on {}".format(pred,curUnassCol,i))

        #     col4row[sink]=i
        #     h=row4col[i]
        #     row4col[i]=sink
        #     sink=h   
        col4row[sink]=pred[sink]
        row4col[pred[sink]]=sink
    if didFlip:
        temp=row4col.clone()
        row4col=col4row
        col4row=temp
        temp=u
        u=v
        v=temp
    return row4col#,u,vcol4row
@torch.jit.script
def ShortestPath(curUnassCol:int ,u: torch.Tensor ,v:torch.Tensor,C:torch.Tensor,col4row:torch.Tensor,row4col:torch.Tensor): #-> Tuple[int,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    numRow,numCol=C.shape
    #numCol=size(C,2);
    pred=torch.zeros(numRow,dtype=torch.int64,device=C.device)
    ScannedCols=torch.zeros(numCol,dtype=torch.bool,device=C.device)
    ScannedRow=torch.zeros(numRow,dtype=torch.bool,device=C.device)
    Row2Scan=torch.arange(0,numRow,device=C.device)
    sink=0
    delta=torch.tensor(0)
    curCol=int(curUnassCol)
    #set array of inf
    shortestPathCost=torch.full((numRow,),100.0,device=C.device)    
    while torch.all(ScannedRow.nonzero()) and sink==0:
        #Mark the current column as having been visited.
        ScannedCols[curCol]=1
        minVal=torch.tensor(float('inf'),device=C.device)
        reducedCosts=C[:,curCol] -u[curCol] -v
        reducedCosts=reducedCosts[ScannedRow==0]
        smallestReducedCost_v,smallestReducedCost_i=torch.min(reducedCosts.unsqueeze(0),dim=1)
        shortestPathCost[ScannedRow==0][smallestReducedCost_i]=smallestReducedCost_v
        pred[ScannedRow==0][smallestReducedCost_i]=curCol
        minVal=torch.min(smallestReducedCost_v,minVal)   
        closestRow=Row2Scan[ScannedRow==0][smallestReducedCost_i]
        ScannedRow[closestRow]=1
        # Row2Scan=Row2Scan[ScannedRow==0]
        ##extra
        delta=smallestReducedCost_v
        
        #If we have reached an unassigned column.
        if col4row[closestRow]==0:
            sink=closestRow.item()
            curCol=col4row[closestRow].item()
    ###not sure if this is needed
    u[curUnassCol]=u[curUnassCol]+delta.item()
    sel= ScannedCols
    sel[curUnassCol]=0
    u[sel]=u[sel] -shortestPathCost[row4col[sel]]+delta
    sel=ScannedRow
    v[sel]=v[sel]-delta+shortestPathCost[sel]
    return (sink,pred,u,v)


def myLinearSum(C):

       
    numRow,numCol=C.shape
    
    # didFlip=numCol>numRow
    # if didFlip:
    #     C=C.transpose(1,0)
    #     numRow,numCol=C.shape
    
    col4row=torch.zeros(numRow,dtype=torch.int64,device=C.device)
    row4col=torch.zeros(numCol,dtype=torch.int64,device=C.device)
    u=torch.zeros(numCol,device=C.device)#The dual variable for the columns
    v=torch.zeros(numRow,device=C.device)#The dual variable for the rows.
    
    #Initially, none of the columns are assigned.
    for curCol in range(numCol):       
        pred=torch.zeros(numRow,dtype=torch.int64,device=C.device)
        #ScannedCols=torch.zeros(numCol,dtype=torch.bool,device=C.device)
        ScannedRow=torch.zeros(numRow,dtype=torch.bool,device=C.device)
        Row2Scan=torch.arange(0,numRow,device=C.device)
        sink=torch.tensor(0,device=C.device)
        #curCol=int(curUnassCol)
        #set array of inf
        #minVal=torch.tensor(float('inf'),device=C.device)
        ScannedCols=torch.zeros(numCol,dtype=torch.bool,device=C.device)
        shortestPathCost=torch.full((numRow,),float('inf'),device=C.device)    
        while torch.all(ScannedRow.nonzero()) and torch.all(sink==0):
            reducedCosts=C[ScannedRow==0][curCol] # extra ? #-u[curCol]-v[Row2Scan[ScannedRow==0]]
            smallestReducedCost_i=torch.argmin(reducedCosts)

            smallestReducedCost_v=torch.min(reducedCosts)
            shortestPathCost[ScannedRow==0][smallestReducedCost_i]=smallestReducedCost_v
            pred[ScannedRow==0][smallestReducedCost_i]=curCol
            #minVal=torch.min(smallestReducedCost_v,minVal)
            
            ####surely this is wrong??! or at least REALLY inefficient
            closestRowScan=Row2Scan[ScannedRow==0][smallestReducedCost_i]      
            closestRow=Row2Scan[ScannedRow==0][closestRowScan]
            ScannedRow[closestRow]=1
            Row2Scan=Row2Scan[ScannedRow==0]
            delta=smallestReducedCost_v
            if col4row[closestRow]==0:
                sink=closestRow
                curCol=col4row[closestRow]
        u[curCol]=u[curCol]+delta
        u[curCol]=u[curCol]+delta
        sel= ScannedCols 
        sel[curCol]=0
        u[sel]=u[sel] -shortestPathCost[row4col[sel]]+delta
        sel=ScannedRow
        v[sel]=v[sel]-delta+shortestPathCost[sel]
        col4row[sink]=pred[sink]
        row4col[pred[sink]]=sink
    # if didFlip:
    #     temp=row4col.clone()
    #     row4col=col4row
    #     col4row=temp
    #     temp=u
    #     u=v
    #     v=temp
    return row4col#,u,vcol4row



B=20
n_queries=80
chunks=torch.randint(1,4,(B,))
cost_Array=torch.rand((chunks.sum(),B*n_queries)).cuda().view(B, n_queries, -1)
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

from scipy.optimize import linear_sum_assignment

with torch.autograd.profiler.profile() as prof:
    B = cost_Array.clone().cpu()

    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(B.split(chunks.tolist(), -1))]

print(prof.key_averages().table(sort_by="self_cpu_time_total"))


C = cost_Array.clone()

myindices = [assign2D(c[i]) for i, c in enumerate(C.split(chunks.tolist(), -1))]
myindices = [myLinearSum(c[i]) for i, c in enumerate(C.split(chunks.tolist(), -1))]

with torch.autograd.profiler.profile() as prof:

    C = cost_Array.clone()
    myindices = [assign2D(c[i]) for i, c in enumerate(C.split(chunks.tolist(), -1))]
print(prof.key_averages().table(sort_by="self_cuda_time_total"))
with torch.autograd.profiler.profile() as prof:

    C = cost_Array.clone()
    myindices = [myLinearSum(c[i]) for i, c in enumerate(C.split(chunks.tolist(), -1))]
print(prof.key_averages().table(sort_by="self_cuda_time_total"))

print(indices)
print(myindices)