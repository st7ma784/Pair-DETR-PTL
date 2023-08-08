from PIL import Image
import torch    

import numpy as np
import os
# from pySmartDL import SmartDL
import pytorch_lightning as pl
from transformers import CLIPTokenizer
import time
import os
import logging
import json
import numpy
import joblib
import sys
import cv2
import tempfile
import cog
import time
from cog import Input,File,Path 
from PIL import Image
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
import os 
# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder
import wget
from typing import Optional,List
# This file defines how the DETIC model is exposed as an AZUREML model endpoint. 
# We have a couple of options here for this. 

import base64,json,io
from base64 import b64decode
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO
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
    def __init__(self, tokenizer,mask_predictor,split="train",dir="HF",T=prep):
        #print('Loading COCO dataset')
        self.data=load_dataset("visual_genome", "relationships_v1.2.0",streaming=False,cache_dir=dir)[split] #,download_mode="force_redownload"
        print("got datast")
        self.mask_predictor=mask_predictor
        self.data = self.data.map(self.process)
        self.__getitem__ = self.data.__getitem__
        self.T=T
        self.tokenizer = tokenizer
        self.tokenize=lambda x:self.tokenizer(x,return_tensors="pt",padding="max_length", truncation=True,max_length=77)['input_ids']
    def process(self,item):
        #for all the relationships in the image
        #  extract the classes of the subject, object 
        #  predictions=self.predict(image,[subject,object])
        #  check predictions for the masks within bboxes. 
        #  if there is a mask, then add the relationship to the list of relationships
        #  return the list of relationships tokenized and clip embedded. 

        #  return coco style annotations per ovject.
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

        self.cfg = get_cfg()
        add_centernet_config(self.cfg)
        add_detic_config(self.cfg)
        self.cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        if not os.path.exists("./models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"):
            url = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
            filename = wget.download(url)
            print("fetched to {}".format(filename))
            self.cfg.MODEL.WEIGHTS = filename
        else:
            self.cfg.MODEL.WEIGHTS = './models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        self.cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        self.cfg.MODEL.DEVICE='cpu' 
        self.text_encoder = build_text_encoder(pretrain=True)
        self.text_encoder.eval()
        self.predictor = DefaultPredictor(self.cfg)
        self.model=self.predictor.model
        BUILDIN_CLASSIFIER = {
            'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
            'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
            'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
            'coco': 'datasets/metadata/coco_clip_a+cname.npy',
        }
        BUILDIN_METADATA_PATH = {
            'lvis': 'lvis_v1_val',
            'objects365': 'objects365_v2_val',
            'openimages': 'oid_val_expanded',
            'coco': 'coco_2017_val',
        }
        
        logging.info("Init complete")
    

    def predict(self,image,classes): 
            
            #assert custom_vocabulary is not None and len(custom_vocabulary.split(',')) > 0
            metadata = MetadataCatalog.get(str(time.time()))
            metadata.thing_classes = classes
            classifier = self.get_clip_embeddings(metadata.thing_classes)
            num_classes = len(classes)
            self.model.roi_heads.num_classes = num_classes
    
            zs_weight = classifier
            zs_weight = torch.cat(
                [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))], 
                dim=1) # D x (C + 1)
            if self.model.roi_heads.box_predictor[0].cls_score.norm_weight:
                zs_weight = torch.nn.functional.normalize(zs_weight, p=2, dim=0)
            zs_weight = zs_weight.to(self.model.device)
            for k in range(len(self.model.roi_heads.box_predictor)):
                del self.model.roi_heads.box_predictor[k].cls_score.zs_weight
                self.model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight
            # Reset visualization threshold
            output_score_threshold = 0.3
            for cascade_stages in range(len(self.predictor.model.roi_heads.box_predictor)):
                self.predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold

            outputs = self.predictor(image)
 
            #So - Idea - What if I could use the score to add noise to the output class. 
            return dict(boxes=outputs['instances'].get_fields()["pred_boxes"].tensor,
                        masks=outputs['instances'].get_fields()["pred_masks"],
                        scores=outputs['instances'].get_fields()["scores"],
                        pred_classes=outputs['instances'].get_fields()["pred_classes"])


    def get_clip_embeddings(self,vocabulary, prompt='a '):
        
        texts = [prompt + x for x in vocabulary]
        if "{}" in prompt:
            texts=[prompt.format(x) for x in vocabulary]
        return self.text_encoder(texts).detach().permute(1, 0).contiguous()
    
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
            self.train=VisGenomeDataset(tokenizer=self.tokenizer,mask_predictor=self.predict, T=self.T,split="train")
            self.val=VisGenomeDataset(tokenizer=self.tokenizer,mask_predictor=self.predict,T=self.T,split="dev")
        self.test=VisGenomeDataset(tokenizer=self.tokenizer,mask_predictor=self.predict,T=self.T,split="test")




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






'''





