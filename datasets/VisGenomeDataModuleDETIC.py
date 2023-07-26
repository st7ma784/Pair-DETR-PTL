from PIL import Image
import torch    

import numpy as np
import os
# from pySmartDL import SmartDL
import pytorch_lightning as pl
from transformers import CLIPTokenizer
import time

#So I think you just need to update that line to:
import io
from torchvision.transforms import *
import torch
import numpy as np
import os
import pytorch_lightning as pl
import time
from datasets import load_dataset
from PIL import Image
import json
import requests
import time
import random
prep=Compose([
        Resize(224, interpolation=Image.NEAREST),
        CenterCrop(224),
        #Note: the standard  lambda function here is not supported by pytorch lightning
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
T= Compose([Resize((224,224),interpolation=Image.NEAREST),ToTensor()])


class VisGenomeIterDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer,split="train",dir="HF",T=prep):
        #print('Loading COCO dataset')
        data=load_dataset("visual_genome", "relationships_v1.2.0",streaming=True,cache_dir=dir)[split]
        print("got datast")
        self.data = data.map(self.process)
        self.T=T
        self.tokenizer = tokenizer
        self.tokenize=lambda x:self.tokenizer(x,return_tensors="pt",padding="max_length", truncation=True,max_length=77)['input_ids']
    def process(self,item):
        r=random.choice(item["relationships"])

        url     = 'http://localhost:5001/predictions'  #Point at the docker running on the same VM
        Classes= " ,".join(list(set([*[r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]]])))
        payload = json.dumps({ "input":{
            "image": item["url"],    #URL to image to search
            "vocabulary": "custom",
            "custom_vocabulary": Classes # Your own prompts.. (note that by default "a " is added to the front, so the first 2 examples here are wrong)
          }})
        headers = {}
        res = requests.post(url, data=payload, headers=headers)
        try:
          resp_dict = res.json()
        except:
          time.sleep(10)
          res = requests.post(url, data=payload, headers=headers)
          resp_dict=res.json
        image =io.BytesIO( base64.b64decode(resp_dict["output"]["path"][21:])) #the 21: allows skipping of header info
        img_array = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img =Image.fromarray(img_array)


        r=self.tokenize(" ".join(["a",r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]]), return_tensors="pt",padding="max_length", truncation=True,max_length=77)
        return {"img":self.T(img),"relation":r}
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        return self.data.__iter__()

class VisGenomeDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer,split="train",dir="HF",T=prep):
        #print('Loading COCO dataset')
        self.data=load_dataset("visual_genome", "relationships_v1.2.0",streaming=False,cache_dir=dir)[split] #,download_mode="force_redownload"
        print("got datast")
        self.data = dataset.map(self.process)
        self.__getitem__ = self.data.__getitem__
        self.T=T
        self.tokenizer = tokenizer
        self.tokenize=lambda x:self.tokenizer(x,return_tensors="pt",padding="max_length", truncation=True,max_length=77)['input_ids']
    def process(self,item):
        r=random.choice(item["relationships"])

        url     = 'http://localhost:5001/predictions'  #Point at the docker running on the same VM
        Classes= " ,".join(list(set([*[r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]]])))
        payload = json.dumps({ "input":{
            "image": item["url"],    #URL to image to search
            "vocabulary": "custom",
            "custom_vocabulary": Classes # Your own prompts.. (note that by default "a " is added to the front, so the first 2 examples here are wrong)
          }})
        headers = {}
        res = requests.post(url, data=payload, headers=headers)
        try:
          resp_dict = res.json()
        except:
          time.sleep(10)
          res = requests.post(url, data=payload, headers=headers)
          resp_dict=res.json
        image =io.BytesIO( base64.b64decode(resp_dict["output"]["path"][21:])) #the 21: allows skipping of header info
        img_array = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img =Image.fromarray(img_array)


        r=self.tokenize(" ".join(["a",r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]]), return_tensors="pt",padding="max_length", truncation=True,max_length=77)
        return {"img":self.T(img),"relation":r}
    def __len__(self):
        return len(self.data)

# Dataset
class VisGenomeDataModule(pl.LightningDataModule):

    def __init__(self, Cache_dir='.', T=prep, batch_size=256):
        super().__init__()
        self.data_dir = Cache_dir
        self.batch_size = batch_size
        self.T=T
        self.tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=Cache_dir)

    def train_dataloader(self, B=None):
        if B is None:
            B=self.batch_size 
        return torch.utils.data.DataLoader(self.train, batch_size=B, shuffle=True, num_workers=1, prefetch_factor=1, pin_memory=True,drop_last=True)
    def val_dataloader(self, B=None):
        if B is None:
            B=self.batch_size
       
        return torch.utils.data.DataLoader(self.val, batch_size=B, shuffle=True, num_workers=1, prefetch_factor=1, pin_memory=True,drop_last=True)
    def test_dataloader(self,B=None):
        if B is None:
            B=self.batch_size


        return torch.utils.data.DataLoader(self.test, batch_size=B, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        #print("Entered COCO datasetup")
        
        if stage == 'fit' or stage is None:
            self.train=VisGenomeDataset(tokenizer=self.tokenizer,T=self.T,split="train")
            self.val=VisGenomeDataset(tokenizer=self.tokenizer,T=self.T,split="dev")
        self.test=VisGenomeDataset(tokenizer=self.tokenizer,T=self.T,split="test")




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








