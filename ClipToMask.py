#this file is an experiment to train a ptl model.
'''
The plan is to take the clip embeddings of a relationship found in the visual genome datase. This embedding is hopefully going to inform the mask

We're going to do 2 tests here. 

The first is to purely evaluate a DETIC model. 

Experiment 1: how good are the detic predictions compared to the gt mask on MSCOCO
    potential output: The DETIC model is a good indicator of the masks for Vis Genome. 
Experiment 2: using the masks from COCO, can CLIP predict them?
    potential output: CLIP is a good indicator of the masks for objects.
Experiment 3: using the masks from VisGenome, can CLIP predict them? 
    potential output: CLIP is a good indicator of the masks for Vis Genome relationships. 


'''

from typing import Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import clip
from model import MLP
class Exp2CLIPtoCOCOMask(pl.LightningModule):
    def __init__(self,layers:int,version:int,outputsize=224):
        super().__init__()
        self.clip,_ =clip.load("ViT-B/32", device="cuda")
        self.outputsize=outputsize
        #We'ere going to build a model that takes the clip embeddings and tries to predict the mask
        #mask shape is BxHxW
        # clip shape is BxL
        # this means we need to expand the information there...
        #to do this we're going to take our B,F and train a row of pixels for x and y, multiply them and then pass that through a third mlp
        #this will give us a mask of the same size as the input. 
        self.weight=nn.Parameter(torch.tensor(0.5))
        if version==1:
            self.xmlp=MLP(input_size=512,hidden_size=512,output_size=512,depth=layers)
            self.ymlp=MLP(input_size=512,hidden_size=512,output_size=512,depth=layers)
            self.finalmlp=MLP(input_size=512,hidden_size=512,output_size=outputsize,depth=layers)
            self.finalcat=MLP(input_size=2*outputsize,hidden_size=768,output_size=outputsize,depth=layers)
            self.forward=self.from_encoding_to_maskv1
        elif version==2:
            self.mlp=MLP(input_size=512,hidden_size=1024,output_size=outputsize*outputsize,depth=layers*2)
            self.forward=self.from_encoding_to_maskv2
        else:
            raise NotImplementedError("Version must be 1 or 2")
    def from_encoding_to_maskv1(self,encoding):
        xrow=self.xmlp(encoding) # B#512
        ycol=self.ymlp(encoding)
        imageFeatures=torch.bmm(xrow.unsqueeze(-1),ycol.unsqueeze(1))#B,512,512
        mask1= self.finalmlp(imageFeatures)
        imageFeatures1=torch.bmm(xrow.unsqueeze(1),ycol.unsqueeze(-1))
        mask2=self.finalmlp(imageFeatures)
        both_masks=self.finalcat(torch.cat([mask1,mask2],dim=-1))
        return both_masks
    def from_encoding_to_maskv2(self,encoding):
        return self.mlp(encoding).view(encoding.shape[0],self.outputsize,self.outputsize)
    def train_step(self,batch,batch_idx):

        image,caption,summed_masks=batch
        encoding=self.clip.encode_text(caption) #B,512
        maskcap=self(encoding) #B,224,224
        encoding=self.clip.encode_image(image) #B,512
        mask_im=self(encoding) #B,224,224
        
        lossa=nn.functional.binary_cross_entropy_with_logits(maskcap,summed_masks)
        lossb=nn.functional.binary_cross_entropy_with_logits(mask_im,summed_masks)
        #weight will be a leant param between 0 and 1

        weight=self.weight.sigmoid()
        self.log("train_loss_cap",lossa)
        self.log("train_loss_img",lossb)
        mask=maskcap*(weight)+mask_im*(1-weight)
        loss=nn.functional.binary_cross_entropy_with_logits(mask,summed_masks)
        return loss
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

from detectron2.engine import DefaultPredictor,build_model
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

class Exp3ClipToVisGenomeMask(Exp2CLIPtoCOCOMask):
    #this is going to be a bit more complicated - including a DEtic model to generate the masks needed
     def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.clip,_ =clip.load("ViT-B/32", device="cuda")
        self.cfg=get_cfg()
        add_detic_config(self.cfg)
        add_centernet_config(self.cfg)
        add_detic_config(self.cfg)
        self.cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        filename="./models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        if not os.path.exists("./models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"):
            if os.path.exists("./Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"):
                os.system("cp ./Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth ./models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
            else:
                url = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
                filename = wget.download(url)
                print("fetched to {}".format(filename))
        self.cfg.MODEL.WEIGHTS = filename
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.12  # set threshold for this model
        self.cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        self.cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        self.cfg.MODEL.DEVICE='cuda'
        self.text_encoder = build_text_encoder(pretrain=True)
        self.text_encoder.eval()
        self.detic = build_model(self.cfg)
        self.detic.eval()
    def detic_forward(self,images,classes,boxes,class_idx):
        #takes a list of images and classes and returns the masks
        #classes are a list of all the names to go and search for with get_clip_embeddings()
        #boxes, are the annotations that visgenome provides to make sure we have the right things
        
        #step one: make big list of all classes, subj, obj, 
        #step one-half: repeat for all " ".join([sub,rel,obj])
        #step two: do predict (im,all_classes)
        #step three: process and cat output, where the output doesnt return for certain masks, then we ought to see if the whole tag DID get returned and send that back instead, otherwise we';ll remove that caption .



        #summed_masks is the masks that we're going to use to train the model, which is the set of boxes of each sub,obj pair  in captions
        #captions is the plain text sub,rel,obj that we're going to use to get the embeddings
        returns captions,summed_masks

    def train_step(self,batch, batch_idx):
        image,targets,tgt_idx=batch
        #in visual genome, we have a set of relations for an image. Boxes are provided for sub and obj but still pin to each relationship. 
        
        captions,summed_masks=self.detic_forward(image,targets,tgt_idx)

        encodingcap=self.clip.encode_text(captions)
        encodingim=self.clip.encode_image(image[tgt_idx])
        maska=self.forward(encodingcap)
        maskb=self.forward(encodingim)
        mask=maska*(self.weight.sigmoid())+maskb*(1-self.weight.sigmoid())
        loss=nn.functional.binary_cross_entropy_with_logits(mask,summed_masks)
        self.log("train_loss",loss)
        return loss

        
        #B,512
        #we're going to take the classes and search them through detic

