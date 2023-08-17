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
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
import torch
import os ,sys 
# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
import time
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder
import wget
from typing import Optional,List
import torchvision
class Exp2CLIPtoCOCOMask(pl.LightningModule):
    def __init__(self,layers:int,version:int,outputsize=224):
        super().__init__()
        self.clip,_ =clip.load("ViT-B/32", device="cuda")
        self.clip.eval()
        self.outputsize=outputsize
        #We'ere going to build a model that takes the clip embeddings and tries to predict the mask
        #mask shape is BxHxW
        # clip shape is BxL
        # this means we need to expand the information there...
        #to do this we're going to take our B,F and train a row of pixels for x and y, multiply them and then pass that through a third mlp
        #this will give us a mask of the same size as the input. 
        self.version=version
        self.w=torch.nn.Parameter(torch.tensor(0.5))
        self.threshold=torch.nn.Parameter(torch.tensor(0.5))
        if version==1:
            self.xmlp=MLP(input_dim=512,hidden_dim=512,output_dim=512,num_layers=layers)
            self.ymlp=MLP(input_dim=512,hidden_dim=512,output_dim=512,num_layers=layers)
            self.finalmlp=MLP(input_dim=512,hidden_dim=512,output_dim=outputsize,num_layers=layers)
            self.finalcat=MLP(input_dim=2*outputsize,hidden_dim=768,output_dim=outputsize,num_layers=layers)
            self.finalcat2=MLP(input_dim=512,hidden_dim=512,output_dim=outputsize,num_layers=layers)

            self.forward=self.from_encoding_to_maskv1
        elif version==2:
            self.mlp=MLP(input_dim=512,hidden_dim=1024,output_dim=outputsize*outputsize,num_layers=layers*2)
            self.forward=self.from_encoding_to_maskv2
        else:
            raise NotImplementedError("Version must be 1 or 2")
        
        self.loss=nn.BCEWithLogitsLoss(reduction="mean")
    def from_encoding_to_maskv1(self,encoding):
        xrow=self.xmlp(encoding) # B#512
        ycol=self.ymlp(encoding)
        # print("xrow",xrow.shape)
        # print("ycol",ycol.shape)

        imageFeatures=torch.bmm(xrow.unsqueeze(-1),ycol.unsqueeze(1))#B,512,512
        mask1= self.finalmlp(imageFeatures)
        imageFeatures1=torch.bmm(ycol.unsqueeze(-1),xrow.unsqueeze(1))
        # print("imageFeatures1",imageFeatures1.shape)
        mask2=self.finalmlp(imageFeatures1)
        both_masks=self.finalcat(torch.cat([mask1,mask2],dim=-1))
        #print("both_masks",both_masks.shape)
        both_masks=self.finalcat2(both_masks.permute(0,2,1))
        return both_masks.view(encoding.shape[0],self.outputsize,self.outputsize)
    def from_encoding_to_maskv2(self,encoding):
        return self.mlp(encoding).view(encoding.shape[0],self.outputsize,self.outputsize)
    def training_step(self,batch,batch_idx):
        #assume coco objects with images, and boxes and masks
        image, targets ,classencodings,masks,batch_idx,(tgt_ids,tgt_bbox,tgt_masks,tgt_sizes)= batch
        encoding=self.clip.encode_image(image) #B,512
        mask_im=self(encoding) #B,224,224
        maskcap=self(classencodings[tgt_ids]) #B,512

        threshold=self.threshold.sigmoid()
        mask_im=mask_im>threshold
        maskcap=maskcap>threshold

        lossa= self.loss(maskcap.float(),tgt_masks)
        lossb= self.loss(mask_im.float(),masks)
        #weight will be a leant param between 0 and 1

        weight=self.w
        self.log("weight",weight)
        self.log("train_loss_cap",lossa)
        self.log("train_loss_img",lossb)
        loss=lossa*weight+lossb*(1-weight)
        return loss
    
    def configure_optimizers(self) -> Any:
        if self.version==2:
            optimizer = torch.optim.AdamW([p for p in self.mlp.parameters()]+
                                          [self.w,self.threshold]
                                          , lr=1e-3)
            return optimizer
        elif self.version==1:
            optimizer=torch.optim.AdamW([p for p in self.xmlp.parameters()]+
            [p for p in self.ymlp.parameters()]+
            [p for p in self.finalmlp.parameters()]+
            [p for p in self.finalcat.parameters()]+
            [self.threshold]+
            [p for p in self.finalcat2.parameters()]+
            [self.w],lr=1e-3)
            return optimizer
        else:

            raise NotImplementedError("Version must be 1 or 2")


class TensorWrapper():
    def __init__(self,tensor):
        self.tensor=tensor
        self.tensor.image_sizes=torch.tensor(tensor.shape[1:]).unsqueeze(0).repeat(tensor.shape[0],1)
    def size(self):
        return self.tensor.size()
class Exp3ClipToVisGenomeMask(Exp2CLIPtoCOCOMask):
    #this is going to be a bit more complicated - including a DEtic model to generate the masks needed
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cfg=get_cfg()
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
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
        self.cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        self.cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        self.cfg.MODEL.DEVICE='cuda'

        self.detic = DefaultPredictor(self.cfg)
        self.detic.model.eval()
        self.loss=nn.MSELoss(reduction="mean")
    @torch.no_grad()
    def detic_forward(self,**batch):
        #This is going to assume we're pulling the Relation info from VisGenomeDataModule.py 
        # so we'll receive a list of "img", "relation","objects","subjects","obj_classes","subj_classes",batch_idx
        # inputs=[{"image":i} for i in batch["img"]]
        img=batch["img"]
        obj_classes=batch["obj_classes"].squeeze()
        subj_classes=batch["subj_classes"].squeeze()
        objects=batch["objects"]
        subjects=batch["subjects"]
        batch_idx=batch["batch_idx"]
        obj_classes_encodings=self.clip.encode_text(obj_classes)
        subj_classes_encodings=self.clip.encode_text(subj_classes)
        classifier=torch.cat([obj_classes_encodings,subj_classes_encodings],dim=0)
        featuresOUT = self.detic.model.backbone(img)
        img.image_sizes=[(224,224)]*img.shape[0]
        setattr(img,"image_sizes",[(224,224)]*img.shape[0])

        proposals, _ = self.detic.model.proposal_generator(img, featuresOUT,None)
        outputs, _ = self.detic.model.roi_heads(None, featuresOUT, proposals,classifier_info=(classifier.clone().detach().float(),None,None))
        
        found_masks=[outputs[i].get('pred_masks') for i in range(len(outputs))]
        found_boxes=[outputs[i].get('pred_boxes').tensor for i in range(len(outputs))] #these are in xyxy format
    
        #split the gt up by batch_idx
        splits=torch.bincount(batch_idx,minlength=img.shape[0]).tolist()
        per_img_objects=objects.split(splits)
        per_img_subjects=subjects.split(splits)
        all_masks,spans,index_arrays=[],[],[]

        for obj,subj,boxes,masks in zip(per_img_objects,per_img_subjects,found_boxes,found_masks):
            object_box_ious=torchvision.ops.box_iou(obj,boxes)
            subject_box_ious=torchvision.ops.box_iou(subj,boxes)
            if object_box_ious.shape[0]==0 or subject_box_ious.shape[0]==0 or boxes.shape[0]==0:
                all_masks.append(torch.zeros((max(obj.shape[0],subj.shape[0]),1,28,28),device=self.device))
                spans.append(max(obj.shape[0],subj.shape[0]))
                #pass
            else:    
                best_obj_boxes=torch.nn.functional.gumbel_softmax(object_box_ious, tau=1, hard=False, eps=1e-10, dim=0)
                best_subj_boxes=torch.nn.functional.gumbel_softmax(subject_box_ious,tau=1, hard=False, eps=1e-10,dim=0)
                #these are one hot foundxB matrixes that represent the best found thing per annotated box

                #print(masks.shape)# torch.Size([41, 1, 28, 28])
                masks=masks.flatten(1)
                #print(best_obj_boxes.shape)# torch.Size()
                masks=torch.logical_or(best_obj_boxes@masks,best_subj_boxes@masks).float()
                #combine the masks for the best obj and subj boxes
                masks=masks.unflatten(1,(1, 28,28))
                all_masks.append(masks)
                spans.append(len(masks))
                index_arrays.append(torch.logical_or(best_obj_boxes,best_subj_boxes))
                #should result in a single #annotations x found boxes where every row in dim0 has 2 entries...
        
        masks_per_caption=torch.cat(all_masks,dim=0)
        #print("idx_masks_per_caption",idx_masks_per_caption.shape)
        masks_per_image=torch.stack([torch.sum(masks,dim=0).clamp(0,1) for masks in all_masks])

        assert masks_per_caption.shape[0]==sum(spans)
        #COMBINE INDEX ARRAYS :
        #each index array represents the one hot of which pair of annotations goes to which output mask. 
        #we need to combine them into a single index array that represents the one hot of which annotations have an output mask. 
        #we can do this by summing the index arrays and then taking the index of non-zero rows
        #the result should be a 1d tensor of size #num annotations filled with bool values to indicate if the annotation has a mask.
        # for i in range(len(index_arrays)):
        #     print("Ann Len",splits[i])
        #     print("spans[{}]".format(i),spans[i])

        #     print("index_arrays[{}]".format(i),index_arrays[i].shape)
        # cap_indexes=torch.cat(index_arrays,dim=0).sum(dim=1).nonzero().squeeze(1)
        # assert torch.sum(cap_indexes)==masks_per_caption.shape[0]
        return masks_per_caption,masks_per_image,spans#,cap_indexes


    def training_step(self,batch, batch_idx):
        #in visual genome, we have a set of relations for an image. Boxes are provided for sub and obj but still pin to each relationship. 
        images=batch["img"]
        captions=batch["relation"].squeeze()
        tgt_idx=batch["batch_idx"]
        encodingcap=self.clip.encode_text(captions).float()
        encodingim=self.clip.encode_image(images).float()
              
        maska=self.forward(encodingcap)
        maskb=self.forward(encodingim)
        #print("maskb",maskb.shape)

        masks_per_caption,masks_per_image,detic_splits=self.detic_forward(**batch)
        #print("masks_per_caption",masks_per_caption.shape)
        masks_per_caption=torch.nn.functional.interpolate(masks_per_caption,size=maska.shape[-2:]).squeeze(1)
        #print("masks_per_image",masks_per_image.shape)
        masks_per_image=torch.nn.functional.interpolate(masks_per_image,size=maskb.shape[-2:]).squeeze(1)
       
        
        threshold=self.threshold.sigmoid()
        maska=maska>threshold
        maskb=maskb>threshold
        self.log("threshold",self.threshold,prog_bar=True)
        lossa=self.loss(maska.float(),masks_per_caption)
        lossb=self.loss(maskb.float(),masks_per_image)
        self.log("weight",self.w,prog_bar=True)

        self.log("caption_loss",lossa,prog_bar=True)
        self.log("image_loss",lossb,prog_bar=True)

        loss=lossa*(self.w.sigmoid())+lossb*(1-self.w.sigmoid())

        self.log("train_loss",loss,prog_bar=True)
        return loss

        
        #B,512
        #we're going to take the classes and search them through detic
    def validation_step(self,batch,batch_idx):
        images=batch["img"]
        captions=batch["relation"].squeeze()
        tgt_idx=batch["batch_idx"]
        encodingcap=self.clip.encode_text(captions).float()
        encodingim=self.clip.encode_image(images).float()
              
        maska=self.forward(encodingcap)
        maskb=self.forward(encodingim)

        masks_per_caption,masks_per_image, splits=self.detic_forward(**batch)
        #
        #print("masks_per_caption",masks_per_caption.shape)
        masks_per_caption=torch.nn.functional.interpolate(masks_per_caption,size=maska.shape[-2:]).squeeze(1)
        #print("masks_per_image",masks_per_image.shape)
        masks_per_image=torch.nn.functional.interpolate(masks_per_image,size=maskb.shape[-2:]).squeeze(1)
        objects=batch["objects"] # bbox in xyxy format
        subjects=batch["subjects"] # bbox in xyxy format  
        #tgt_bbox is all the concatenates bboxes
        stacked=torch.stack([objects,subjects],dim=-1)
        tgt_bbox=torch.stack([torch.min(stacked[:,0],dim=-1).values,
                              torch.min(stacked[:,1],dim=-1).values, # these find the top left corner
                                torch.max(stacked[:,2],dim=-1).values, # find the bottom right corner with max of x ys and add the whs.  
                            torch.max(stacked[:,3],dim=-1).values],dim=1)

        batch_ann_counts=torch.bincount(tgt_idx,minlength=images.shape[0]).tolist()
        # print("batch_ann_counts",batch_ann_counts)
        # print("splits",splits)
        # print("masks_per_caption",masks_per_caption.shape)
        # print("masksa",maska.shape)
        self.logger.log_image(key="validation samples",
            images=[i for i in images],
            masks=[{
                "prediction": {"mask_data": torch.argmax(torch.cat([torch.full((1,*maska.shape[1:]),0.2,device=maska.device),maska],dim=0),dim=0).cpu().numpy().astype(int),"class_labels": {i:str(i) for i in range(ca)}},
                "ground_truth": {"mask_data":torch.argmax(torch.cat([torch.full((1,*masks_per_caption.shape[1:]),0.2,device=masks_per_caption.device),masks_per_caption],dim=0),dim=0).cpu().numpy().astype(int),"class_labels": {i:str(i) for i in range(cb)}}
            } for maska,masks_per_caption,ca,cb in zip(maska.to(torch.int).split(batch_ann_counts),masks_per_caption.to(torch.int).split(splits),batch_ann_counts,splits)],
            # boxes=[{
            #     "ground_truth": {"box_data": tgt_bbox.tolist()},# "class_labels": c},
            # } for tgt_bbox,c in zip(tgt_bbox.split(batch_ann_counts),torch.arange(tgt_bbox.shape[0]).split(batch_ann_counts))],
        )

        
if __name__ == "__main__":

    # we're going to need to look at datatest 3 first... which means loading the visual genome dataset.
    from data.VisGenomeDataModule import VisGenomeDataModule
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--Cache_dir', type=str, default='.', help='path to download and cache data')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--stream', default=False, type=bool,help='stream data',)
    parser.add_argument("--COCO", default=False, type=bool,help="Use COCO style data")
    args=parser.parse_args()
    dir=os.path.join(args.Cache_dir,"data")
    dm =VisGenomeDataModule(Cache_dir=dir,batch_size=args.batch_size)
    dm.prepare_data()
    dm.setup()

    model=Exp3ClipToVisGenomeMask(layers=2,version=2)
    logger=pl.loggers.WandbLogger(project="ClipToMask",entity="st7ma784",name="Exp3ClipToVisGenomeMask")
    trainer = pl.Trainer(gpus=1,precision=32,max_epochs=1,fast_dev_run=False,logger=logger)
    trainer.fit(model, dm)

    