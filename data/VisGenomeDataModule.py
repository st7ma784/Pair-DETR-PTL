#So I think you just need to update that line to:
import torchvision
#torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms
#from torchvision.transforms import v2 as transforms

import torch
from transformers import CLIPTokenizer
import os
import pytorch_lightning as pl
from torch.utils.data import Dataset,IterableDataset
from datasets import load_dataset
from PIL import Image
import random
import requests
from io import BytesIO
from torchvision import datapoints
prep=transforms.Compose([
        transforms.Resize(224, interpolation=Image.NEAREST),
        transforms.CenterCrop(224),
        #Note: the standard  lambda function here is not supported by pytorch lightning
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

class VisGenomeIterDataset(IterableDataset):
    def __init__(self,split="train",dir="HF",T=prep,tokenizer=None):
        #print('Loading COCO dataset')
        self.data=load_dataset("visual_genome", "relationships_v1.2.0",streaming=True,cache_dir=dir)[split]
        print("got datast")
        self.T=T
        self.tokenizer=tokenizer
        if tokenizer is None:
            self.tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=dir) 
        self.tokenize=lambda x:self.tokenizer(x,return_tensors="pt",padding="max_length", truncation=True,max_length=77)['input_ids']
        self.data =self.data.map(self.process,remove_columns=['image_id' , 'image','relationships','width', 'height', 'coco_id', 'flickr_id','url' ])

    def process(self,item):
        objects=[]
        subjects=[]
        captions=[]
        obj_classes=[]
        subj_classes=[]

        for r in item["relationships"]:
            obj_classes.append(" ".join(["a", r["object"]["names"][0]]))
            subj_classes.append(" ".join(["a", r["subject"]["names"][0]]))  
            caption=" ".join(["a",r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]])
            captions.append(self.tokenize(caption))
            objects.append(datapoints.BoundingBox([r["subject"]["x"],r["subject"]["y"],r["subject"]["w"],r["subject"]["h"]], format=datapoints.BoundingBoxFormat.XYWH, spatial_size=[item["width"],item["height"]]).to_xyxy_array())
            subjects.append(datapoints.BoundingBox([r["object"]["x"],r["object"]["y"],r["object"]["w"],r["object"]["h"]], format=datapoints.BoundingBoxFormat.XYWH, spatial_size=[item["width"],item["height"]]).to_xyxy_array())
            #captions.append(caption)
        try:
            img,boxes= prep(item["image"],boxes=objects+subjects)
        except FileNotFoundError as e:
            response = requests.get(item["url"])
            img,boxes = prep(Image.open(BytesIO(response.content)),boxes=objects+subjects)

        finally:
            
            
            return {"img":img,"relation":captions,"objects":boxes[:len(boxes//2)],"subjects":boxes[len(boxes//2):], "obj_classes":torch.stack([self.tokenize(x) for x in obj_classes]),"subj_classes":torch.stack([self.tokenize(x) for x in subj_classes])}
        
    def __len__(self):
        return 108077
    def __iter__(self):
        return self.data.__iter__()

    

class VisGenomeDataset(Dataset):
    def __init__(self, split="train",dir="HF",T=prep,tokenizer=None):
        #print('Loading COCO dataset')
        self.data=load_dataset("visual_genome", "relationships_v1.2.0",streaming=False,cache_dir=dir)[split]
        self.T=T
        self.tokenizer=tokenizer
        if tokenizer is None:
            self.tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=dir) 
        self.tokenize=lambda x:self.tokenizer(x,return_tensors="pt",padding="max_length", truncation=True,max_length=77)['input_ids']
        # self.data =self.data.map(self.process,remove_columns=['image_id' , 'image','relationships','width', 'height', 'coco_id', 'flickr_id','url' ],num_proc=8).filter(lambda example: example is not None)
        
    def process(self,item):
        objects=[]
        subjects=[]
        captions=[]
        obj_classes=[]
        subj_classes=[]
        img,boxes=None,None
        for r in item["relationships"]:
            obj_classes.append(" ".join(["a", r["object"]["names"][0]]))
            subj_classes.append(" ".join(["a", r["subject"]["names"][0]]))  
            caption=" ".join(["a",r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]])
            captions.append(self.tokenize(caption))
            objects.append(torch.tensor([r["subject"]["x"],r["subject"]["y"],r["subject"]["x"]+r["subject"]["w"],r["subject"]["y"]+r["subject"]["h"]]))
            subjects.append(torch.tensor([r["object"]["x"],r["object"]["y"],r["object"]["x"]+r["object"]["w"],r["object"]["y"]+r["object"]["h"]]))
            #captions.append(caption)
        try:
            img,boxes= prep(item["image"],objects+subjects)
        except FileNotFoundError as e:
            response = requests.get(item["url"])
            img,boxes = prep(Image.open(BytesIO(response.content)),objects+subjects)
        
        return {"img":img,"relation":captions,"objects":boxes[:(len(boxes)//2)],"subjects":boxes[(len(boxes)//2):], "obj_classes":torch.stack([self.tokenize(x) for x in obj_classes]),"subj_classes":torch.stack([self.tokenize(x) for x in subj_classes])}
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.process(self.data.__getitem__(idx))
def Collate(batch):
    #batch is a list of dicts with keys img, relation, objects, subjects, obj_classes, subj_classes
    #we're boing to stack the images, relations, obj_classes, subj_classes as these are all to be handed to CLIP
    #we're going to cat the objects and subjects as these boxes are to be handed to DETR
    #we're going to make out batch_idx with torch.cat( torch.fill(idx in batch, len(batch[idx])))

    batch=zip(*batch)
    batch["img"]=torch.stack(batch["img"])
    batch["relation"]=torch.cat(batch["relation"])
    batch["obj_classes"]=torch.cat(batch["obj_classes"])
    batch["subj_classes"]=torch.cat(batch["subj_classes"])
    batch["objects"]=torch.cat(batch["objects"])
    batch["subjects"]=torch.cat(batch["subjects"])
    batch["batch_idx"]=torch.cat([torch.full((len(x),),i) for i,x in enumerate(batch[0])])
    
    #batch_idx=torch.cat([torch.full((len(x),),i) for i,x in enumerate(batch[0])])

    return batch
class VisGenomeDatasetBigBoxes(VisGenomeDataset):
    def process(self,item):
        if len(item["relationships"])==0:
            return None
        #this is probably awful,  Why not look at the Pairwise DETR for how it does contrastive loss on multiple objects? 
        boxes=[]
        captions=[]
        obj_classes=[]
        subj_classes=[]

        for r in item["relationships"]:
            obj_classes.append(" ".join(["a", r["object"]["names"][0]]))
            subj_classes.append(" ".join(["a", r["subject"]["names"][0]]))      
            caption=" ".join(["a",r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]])
            captions.append(self.tokenize(caption))
           
            original_bbox=[min(r["object"]["x"],r["subject"]["x"]),
                                min(r["object"]["y"],r["subject"]["y"]), # these find the top left corner
                                max(r["object"]["x"]+r["object"]["w"],r["subject"]["x"]+r["subject"]["w"]), # find the bottom right corner with max of x ys and add the whs.  
                            max(r["object"]["y"]+r["object"]["h"],r["subject"]["y"]+r["subject"]["h"])]
            #captions.append(caption)
            boxes.append(original_bbox)
        try:
            img,boxes= prep(item["image"],boxes=boxes)
        except FileNotFoundError as e:
            response = requests.get(item["url"])
            img,boxes = prep(Image.open(BytesIO(response.content)),boxes=boxes)

        finally:
            
            
            return {"img":img,"relation":torch.stack(captions),"boxes":torch.stack(boxes),"obj_classes":torch.stack([self.tokenize(x) for x in obj_classes]),"subj_classes":torch.stack([self.tokenize(x) for x in subj_classes])}
        
   
            
class VisGenomeDatasetIterBigBoxes(VisGenomeDataset):
    def process(self,item):
        if len(item["relationships"])==0:
            return None
        #this is probably awful,  Why not look at the Pairwise DETR for how it does contrastive loss on multiple objects? 
        boxes=[]
        captions=[]
        obj_classes=[]
        subj_classes=[]

        for r in item["relationships"]:
            obj_classes.append(" ".join(["a", r["object"]["names"][0]]))
            subj_classes.append(" ".join(["a", r["subject"]["names"][0]]))

            caption=" ".join(["a",r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]])
            captions.append(self.tokenize(caption))
           
            original_bbox=[min(r["object"]["x"],r["subject"]["x"]),
                                min(r["object"]["y"],r["subject"]["y"]), # these find the top left corner
                                max(r["object"]["x"]+r["object"]["w"],r["subject"]["x"]+r["subject"]["w"]), # find the bottom right corner with max of x ys and add the whs.  
                            max(r["object"]["y"]+r["object"]["h"],r["subject"]["y"]+r["subject"]["h"])]
            #captions.append(caption)
            boxes.append(original_bbox)
        try:
            img,boxes= prep(item["image"],boxes=boxes)
        except FileNotFoundError as e:
            response = requests.get(item["url"])
            img,boxes = prep(Image.open(BytesIO(response.content)),boxes=boxes)

        finally:
            return {"img":img,"relation":torch.stack(captions),"boxes":torch.stack(boxes),"obj_classes":torch.stack([self.tokenize(x) for x in obj_classes]),"subj_classes":torch.stack([self.tokenize(x) for x in subj_classes])}
        
# Dataset
class VisGenomeDataModule(pl.LightningDataModule):

    def __init__(self, Cache_dir='.', T=prep, batch_size=256,stream=False, fullBoxes=False):
        super().__init__()
        self.data_dir = Cache_dir
        self.batch_size = batch_size
        self.stream=stream
        self.tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32") 
        self.collate_fn=None
        if fullBoxes:
            self.dataConstructor=VisGenomeDatasetBigBoxes
            if self.stream:
                self.dataConstructor=VisGenomeDatasetIterBigBoxes
        else:
            self.dataConstructor=VisGenomeDataset
            self.collate_fn=Collate
            if self.stream:
                self.dataConstructor=VisGenomeIterDataset
        self.T=T

    def train_dataloader(self, B=None):
        if B is None:
            B=self.batch_size 
        if self.collate_fn is not None:
            return torch.utils.data.DataLoader(self.train, batch_size=B, shuffle=not self.stream, num_workers=4, prefetch_factor=3, pin_memory=True,drop_last=True,collate_fn=self.collate_fn)
        return torch.utils.data.DataLoader(self.train, batch_size=B, shuffle=not self.stream, num_workers=4, prefetch_factor=3, pin_memory=True,drop_last=True)
    def val_dataloader(self, B=None):
        if B is None:
            B=self.batch_size
        if self.collate_fn is not None:
            return torch.utils.data.DataLoader(self.val, batch_size=B, shuffle=not self.stream, num_workers=1, prefetch_factor=1, pin_memory=True,drop_last=True,collate_fn=self.collate_fn)       
        return torch.utils.data.DataLoader(self.val, batch_size=B, shuffle=not self.stream, num_workers=1, prefetch_factor=1, pin_memory=True,drop_last=True)
    def test_dataloader(self,B=None):
        if B is None:
            B=self.batch_size
        if self.collate_fn is not None:
            return torch.utils.data.DataLoader(self.test, batch_size=B, shuffle=not self.stream, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True,collate_fn=self.collate_fn)

        return torch.utils.data.DataLoader(self.test, batch_size=B, shuffle=not self.stream, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        #print("Entered COCO datasetup")
        
        #if stage == 'fit' or stage is None:
        self.train=self.dataConstructor(T=self.T,dir=self.data_dir,split="train",tokenizer=self.tokenizer)
    



if __name__ == "__main__":
    #run this to test the dataloader
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--Cache_dir', type=str, default='.', help='path to download and cache data')
    dir=os.path.join(parser.parse_args().Cache_dir,"data")
    dm =VisGenomeDataModule(Cache_dir="HF",batch_size=3)
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        print(batch)
        break