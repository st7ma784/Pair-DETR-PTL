import torch

def assign2D(C):
    #convert to pytorch the function in linear_assigment.m

       
    numRow,numCol=C.shape
    
    didFlip=numCol>numRow
    if didFlip:
        C=C.transpose(1,0)
        numRow,numCol=C.shape
    C=C.sigmoid()
    #These store the assignment as it is made.
    col4row=torch.zeros(numRow,dtype=torch.int64)
    row4col=torch.zeros(numCol,dtype=torch.int64)
    u=torch.zeros(numCol)#The dual variable for the columns
    v=torch.zeros(numRow)#The dual variable for the rows.
    
    #Initially, none of the columns are assigned.
    for curUnassCol in range(numCol):       
        #This finds the shortest augmenting path starting at k and returns
        #the last node in the path.
        [sink,pred,u,v]=ShortestPath(curUnassCol,u,v,C,col4row,row4col)
        assert sink!=-1,"The problem is not feasible."
        #We have to remove node k from those that must be assigned.
        i=-1
        while(i!=curUnassCol):
            i=pred[sink]
            col4row[sink]=i
            h=row4col[i]
            row4col[i]=sink
            sink=h   
    
    if didFlip:
        temp=row4col
        row4col=col4row
        col4row=temp
        
        temp=u
        u=v
        v=temp
    return col4row,row4col#,u,v

def ShortestPath(curUnassCol,u,v,C,col4row,row4col):
    numRow,numCol=C.shape
    #numCol=size(C,2);
    pred=torch.zeros(numRow,dtype=torch.int64)
    
    #Initially, none of the rows and columns have been scanned.
    #This will store a 1 in every column that has been scanned.
    ScannedCols=torch.zeros(numCol)
    
    #This will store a 1 in every row that has been scanned.
    ScannedRow=torch.zeros(numRow)
    Row2Scan=[*range(0,numRow)]
    numRow2Scan=numRow
    
    sink=0
    delta=0
    curCol=int(curUnassCol)
    #set array of inf
    shortestPathCost=torch.full((numRow,),100)    
    while sink==0:
        #Mark the current column as having been visited.
        ScannedCols[curCol]=1
        
        #Scan all of the rows that have not already been scanned.
        minVal=torch.tensor(100,dtype=torch.float64)
        for i,curRowScan in enumerate(Row2Scan):
            curRow=int(curRowScan)
            
            reducedCost=delta+C[curRow,curCol]-u[curCol]-v[curRow]
            if reducedCost<shortestPathCost[curRow]:
                pred[curRow]=curCol
                shortestPathCost[curRow]=reducedCost
            
            
            #%Find the minimum unassigned row that was scanned.
            if shortestPathCost[curRow]<minVal:
                minVal=shortestPathCost[curRow]
                closestRowScan=i
            
        

        if minVal.isinf():
           #If the minimum cost row is not finite, then the problem is
           #not feasible
           return (-1,pred,u,v)
        
        
        closestRow=Row2Scan[closestRowScan]
        
        #%Add the row to the list of scanned rows and delete it from
        #%the list of rows to scan.
        ScannedRow[closestRow]=1
        numRow2Scan=numRow2Scan-1
        Row2Scan.remove(Row2Scan[closestRowScan])
        
        delta=shortestPathCost[closestRow]
        
        #If we have reached an unassigned column.
        if col4row[closestRow]==0:
            sink=closestRow
            curCol=int(col4row[closestRow])
        
    
    
    #%Dual Update Step
    
    #%#Update the first column in the augmenting path.
    u[curUnassCol]=u[curUnassCol]+delta
    #%Update the rest of the columns in the augmenting path.
    #print("curUnassCol",curUnassCol)
    #print("ScannedCols {} {}".format(ScannedCols.shape,ScannedCols))# 1,1
    sel= ScannedCols!=0 #find(ScannedCols~=0)
    #print("sel {} {}".format(sel.shape,sel))# 1,1 
    sel[curUnassCol]=torch.zeros([])
    #print("u {} {}".format(u.shape ,u))# 1,1
    #print(row4col.shape)# 1)
    u[sel]=u[sel] -shortestPathCost[row4col[sel]]+delta
    
    #%Update the scanned rows in the augmenting path.
    #in torch we can't do ScannedRow~=0 - so we do this
    sel=sel.nonzero()
    v[sel]=v[sel]-delta+shortestPathCost[sel]

    return (sink,pred,u,v)




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

from scipy.optimize import linear_sum_assignment
indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(chunks.tolist(), -1))]
myindices = [assign2D(c[i]) for i, c in enumerate(C.split(chunks.tolist(), -1))]

print(indices)
print(myindices)