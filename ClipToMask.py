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
        print("both_masks",both_masks.shape)
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
        lossa= self.loss(maskcap,tgt_masks)
        lossb= self.loss(mask_im,masks)
        #weight will be a leant param between 0 and 1

        weight=self.w
        self.log("weight",weight)
        self.log("train_loss_cap",lossa)
        self.log("train_loss_img",lossb)
        loss=lossa*weight+lossb*(1-weight)
        return loss
    
    def configure_optimizers(self) -> Any:
        if self.version==2:
            optimizer = torch.optim.AdamW([p for p in self.mlp.parameters()]+[self.w], lr=1e-3)
            return optimizer
        elif self.version==1:
            optimizer=torch.optim.AdamW([p for p in self.xmlp.parameters()]+
            [p for p in self.ymlp.parameters()]+
            [p for p in self.finalmlp.parameters()]+
            [p for p in self.finalcat.parameters()]+
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
        #do predictions on all images with DETIC 
        #classifier = self.get_clip_embeddings(self,classes)
        #self.detic.model.roi_heads.num_classes =  obj_classes_encodings.shape[0]+subj_classes_encodings.shape[0]
        # metadata = MetadataCatalog.get(str(time.time()))
        # metadata.thing_classes = self.detic.model.roi_heads.num_classes
        classifier=torch.cat([obj_classes_encodings,subj_classes_encodings],dim=0)
        # zs_weight = classifier.T# torch.cat([classifier, classifier.new_zeros((classifier.shape[0], 1))], dim=1) # D x (C + 1)
        # if self.detic.model.roi_heads.box_predictor[0].cls_score.norm_weight:
        #     classifier = torch.nn.functional.normalize(classifier, p=2, dim=1)
        # zs_weight = zs_weight.to(self.detic.model.roi_heads.box_predictor[0].cls_score.zs_weight)
        # for k in range(len(self.detic.model.roi_heads.box_predictor)):
        #     #print(self.detic.model.roi_heads.box_predictor[k].cls_score.zs_weight.__dir__())
        #     del self.detic.model.roi_heads.box_predictor[k].cls_score.zs_weight
        #     self.detic.model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight
            # print("zs_weight",zs_weight.shape)
            # print("cls_score",self.detic.model.roi_heads.box_predictor[k].cls_score.zs_weight.shape)
            #self.detic.model.roi_heads.box_predictor[k].cls_score.zs_weight.copy_(zs_weight)
        
        # output_score_threshold = self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        # for cascade_stages in range(len(self.detic.model.roi_heads.box_predictor)):
        #     self.detic.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold        
        #So - Idea - What if I could use the score to add noise to the output class. 
        featuresOUT = self.detic.model.backbone(img)
        img.image_sizes=[(224,224)]*img.shape[0]
        setattr(img,"image_sizes",[(224,224)]*img.shape[0])

        # features = [featuresOUT[f] for f in self.detic.model.proposal_generator.in_features]
        # _, reg_pred_per_level, agn_hm_pred_per_level = self.detic.model.proposal_generator.centernet_head(features)
        # grids = self.detic.model.proposal_generator.compute_grids(features)
        # agn_hm_pred_per_level = [x.sigmoid() if x is not None else None \
        #     for x in agn_hm_pred_per_level]

        # proposals = self.detic.model.proposal_generator.predict_instances(
        #     grids, agn_hm_pred_per_level, reg_pred_per_level, 
        #     [(224,224)]*img.shape[0], [None for _ in agn_hm_pred_per_level])
        # for p in range(len(proposals)):
        #         proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
        #         proposals[p].objectness_logits = proposals[p].get('scores')
        #         proposals[p].remove('pred_boxes')
        #         proposals[p].remove('scores')
        #         proposals[p].remove('pred_classes')

        proposals, _ = self.detic.model.proposal_generator(img, featuresOUT,None)
        #fault is after here
        outputs, _ = self.detic.model.roi_heads(None, featuresOUT, proposals,classifier_info=(classifier.clone().detach().float(),None,None))
        #So heres what I don't understand.... 
        '''
        In this roi_heads function, we take the set of proposals, and then we run them through the roi_heads.
        each ROi_head is a detic_roi_head we take the max value from each head. Each head has a DETIC ROI Predictor, that essentially just does f @ zs_weight?
        So - we're going to need to do a few things here.

        1. We need to take the zs_weight and the f and do a matrix multiply
        2. We need to take the max value from each head
        '''


        #     print("outputs",outputs.keys())
        #outputs is a list of Instances, each instance has pred_boxes, pred_classes, pred_masks, scores
        found_masks=[outputs[i].get('pred_masks') for i in range(len(outputs))]
        splits=[len(outputs[i].get('pred_boxes')) for i in range(len(outputs))]
        found_boxes=[outputs[i].get('pred_boxes').tensor for i in range(len(outputs))] #these are in xyxy format
        #check outputs for bounding boxes that are close to the subject and object boxes.
        #do box iou between outputs and inputs split by batch_idx


        #These are Boxes, we need to convert them to tensors,
        found_boxes=torch.cat(found_boxes,dim=0)
        found_masks=torch.cat(found_masks,dim=0)
        #found_boxes torch.Size([45, 4])
        #found_masks torch.Size([45, 1, 28, 28])
        object_box_ious=torchvision.ops.box_iou(found_boxes,objects)
        subject_box_ious=torchvision.ops.box_iou(found_boxes,subjects)

        
        bestobjboxes=torch.max(object_box_ious,dim=1).indices
        bestsubjboxes=torch.max(subject_box_ious,dim=1).indices #draw out which dim is which

        print("bestobjboxes",bestobjboxes.shape)
        print("splits",splits)
        obj_masks_per_caption= found_masks[bestobjboxes]# select masks corresponding to best boxes,
        sub_masks_per_caption= found_masks[bestsubjboxes]
        
        #do matcher based on box iou between outputs and inputs split by batch_idx 
        batch_one_hot=torch.nn.functional.one_hot(batch_idx,num_classes=img.shape[0])
        #print("obj masks_per_caption",obj_masks_per_caption.shape)
        #print("masks_per_caption",obj_masks_per_caption)
        masks_per_caption=torch.logical_or(obj_masks_per_caption,sub_masks_per_caption).float()
        idx_masks_per_caption= masks_per_caption*batch_one_hot.unsqueeze(-1).unsqueeze(-1).float() 

        #print("idx_masks_per_caption",idx_masks_per_caption.shape)
        masks_per_image=torch.sum(idx_masks_per_caption,dim=0)

        assert masks_per_caption.shape[0]==sum(splits)
        return masks_per_caption,masks_per_image,splits


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

        masks_per_caption,masks_per_image=self.detic_forward(**batch)
        print("masks_per_caption",masks_per_caption.shape)
        masks_per_caption=torch.nn.functional.interpolate(masks_per_caption,size=maska.shape[-2:]).squeeze(1)
        masks_per_image=torch.nn.functional.interpolate(masks_per_image.unsqueeze(1),size=maskb.shape[-2:]).squeeze(1)
        #mask=maska*(self.weight.sigmoid())+maskb*(1-self.weight.sigmoid())
        


        lossa=self.loss(maska,masks_per_caption)
        lossb=self.loss(maskb,masks_per_image)/(224*224)
        self.log("weight",self.w,prog_bar=True)

        self.log("caption_loss",lossa,prog_bar=True)
        self.log("image_loss",lossb,prog_bar=True)

        loss=lossa*(self.w)+lossb*(1-self.w)

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
        #print("masks_per_caption",masks_per_caption.shape)
        masks_per_caption=torch.nn.functional.interpolate(masks_per_caption,size=maska.shape[-2:]).squeeze(1)
        #print("masks_per_image",masks_per_image.shape)
        masks_per_image=torch.nn.functional.interpolate(masks_per_image.unsqueeze(1),size=maskb.shape[-2:]).squeeze(1)
        objects=batch["objects"] # bbox in xyxy format
        subjects=batch["subjects"] # bbox in xyxy format  
        #tgt_bbox is all the concatenates bboxes
        stacked=torch.stack([objects,subjects],dim=-1)
        tgt_bbox=torch.stack([torch.min(stacked[:,0],dim=-1).values,
                              torch.min(stacked[:,1],dim=-1).values, # these find the top left corner
                                torch.max(stacked[:,2],dim=-1).values, # find the bottom right corner with max of x ys and add the whs.  
                            torch.max(stacked[:,3],dim=-1).values],dim=1)

        batch_ann_counts=torch.bincount(tgt_idx,minlength=images.shape[0]).tolist()
        #print("masks_per_caption",masks_per_caption.shape)
        self.logger.log_image(key="validation samples",
            images=[i for i in images],
            masks={
                "prediction": {"mask_data": maska.cpu().split(batch_ann_counts), "class_labels":  torch.arange(tgt_bbox.shape[0])},
                "ground_truth": {"mask_data": masks_per_caption.cpu().split(splits), "class_labels": torch.arange(sum(splits))}
            },
            boxes={
                "ground_truth": {"box_data": tgt_bbox.split(batch_ann_counts), "class_labels": torch.arange(tgt_bbox.shape[0])},
            }
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

    model=Exp3ClipToVisGenomeMask(layers=2,version=1)
    logger=pl.loggers.WandbLogger(project="ClipToMask",entity="st7ma784",name="Exp3ClipToVisGenomeMask")
    trainer = pl.Trainer(gpus=1,precision=32,max_epochs=1,fast_dev_run=False,logger=logger)
    trainer.fit(model, dm)

    