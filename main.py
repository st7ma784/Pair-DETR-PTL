# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------


from pathlib import Path
from data.coco_eval import CocoEvaluator
# from datasets.panoptic_eval import PanopticEvaluator
from functools import reduce
# from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
# from torch.utils.data import DataLoader
from util.misc import inverse_sigmoid
import evaluate
from model import *
import pytorch_lightning as pl
from transformers import CLIPTokenizer
import numpy as np
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
# class DETRsegm(nn.Module):
#     def __init__(self, detr, freeze_detr=False):
#         super().__init__()
#         self.detr = detr

#         if freeze_detr:
#             for p in self.parameters():
#                 p.requires_grad_(False)

#         hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
#         self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
#         self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

#     def forward(self, samples: NestedTensor):
        
#         bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)

#         seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
#         outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])

#         out["pred_masks"] = outputs_seg_masks
#         return out
class PairDETR(pl.LightningModule):
    def __init__(self,**args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.learning_rate = args['lr']
        self.Tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        #posmethod=PositionEmbeddingLearned
        #if args['position_embedding'] in ('v2', 'sine'):
            #TODO find a better way of exposing other arguments
        posmethod = PositionEmbeddingSine
        self.backbone = Backbone(args['backbone'], False, True,False)
        self.num_queries = args['num_queries']
        hidden_dim = args['hidden_dim']
        self.transformer = PositionalTransformer(
            d_model=args['hidden_dim'],
            dropout=args['dropout'],
            nhead=args['nheads'],
            dim_feedforward=args['dim_feedforward'],
            num_encoder_layers=args['enc_layers'],
            num_decoder_layers=args['dec_layers'],
            normalize_before=args['pre_norm'],
            return_intermediate_dec=args['intermediate_layer'],
            device=self.device,
        )
        self.positional_embedding = posmethod(args['hidden_dim'] // 2,device=self.device)
       
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 6)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        self.aux_loss = args['aux_loss']
        self.loss=nn.CrossEntropyLoss(reduction="mean")
        self.clip_projection=nn.Linear(512,hidden_dim)

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.input_proj.weight.data, 0)
        # if args.masks:
        #     model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

        self.weight_dict = {'loss_out_iou':0.5,
                            'loss_gt_iou':0.5,
                            'class_loss':1,
                            "class_mask_loss":2,
                            'loss_bbox_acc': 0.0001*args['bbox_loss_coef'],
                            'loss_giou': 0.01*args['giou_loss_coef'],
                            'loss_dice': 1000*args['dice_loss_coef'], #  last unc
                            'loss_mask': 100*args['mask_loss_coef'], # 
                            'CELoss':1}

        # self.criterion = SetCriterion( 
        #                               weight_dict=self.weight_dict,
        #                             focal_alpha=args['focal_alpha'],
        #                              losses=['labels', 'boxes','masks'], # final cardinality
        #                              logger=self.logger
        #                             )
        self.criterion= FastCriterion(weight_dict=self.weight_dict,logger=self.logger)
        # TO DO : CREATE SECOND CRTIERION FOR THE SECOND HEAD
        
        
        self.postprocessors = {'bbox': PostProcess()}
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, args['nheads'], dropout=0.01)
        self.mask_head = MaskHeadSmallConv(hidden_dim + args['nheads'], [1024, 512, 256], hidden_dim)
        self.threshold = 0.75

    def tokenize(self,x):
            return self.Tokenizer(x, # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids']
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.args['weight_decay'])
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args['lr_drop'])
        return [optimizer]

    def forward(self, samples,classencoding,masks=None):

        # if isinstance(samples, (list, torch.Tensor)):
        #     samples = nested_tensor_from_tensor_list(samples)
        #     print("damn here! ")
        #features, pos = self.backbone(samples) this called Joiner which is a wrapper for the backbone:

        src = self.backbone(samples)
        src,feats=src[-1],src[:-1]
        # print(masks.shape) #B,800,800
        # print(src.shape) 
    
        mask=F.interpolate(masks[None].float(),size=src.shape[-2:]).bool()[-1]        

        classencodings=self.clip_projection(classencoding)
        classencodings=classencodings.repeat(1,self.num_queries,1).flatten(0,1)


        outputs_coord, outputs_class ,outputs_coord2, outputs_class2= self.transformer( self.input_proj(src) , classencodings, self.positional_embedding(src,mask), mask)
        #filter these outputs by the cosine similarity of the class encodings, we want to find the most similar class encoding to each of the outputs, and ensure that its above a threshold

        similarities=torch.einsum('bqf,gf->bqg', outputs_class[-1], classencoding.squeeze())# this takes output classes of shape (1,24,240,512) and class encodings of shape (240,512) and returns (1,24,240,n_classes)
        #use this as a mask to filter the outputs
        pred_to_keep_mask=torch.max(similarities,dim=-1).values>self.threshold
        # print("filter",pred_to_keep_mask.shape)  #B, NQ* N_classes
        # print("outputs",outputs_class[-1].shape) #B, NQ*N_classes, 512
        # print("mask",mask.shape) #B,25,25
        bbox_mask = self.bbox_attention(outputs_class[-1], classencodings, mask=mask)
        #it feels like this should have references and not class encodings, but I'm not sure how to get them
        feats=feats[::-1]
        seg_masks = self.mask_head(self.input_proj(src), bbox_mask, feats)
        outputs_seg_masks = seg_masks.view(src.shape[0], -1, seg_masks.shape[-2], seg_masks.shape[-1])
        #print("seg_masks",outputs_seg_masks.shape) # B, NQ* N_classes, 200,200
        #print("outputs_seg_masks",outputs_seg_masks.shape) # B, NQ* N_classes, 200,200
        #outputs_seg_masks=outputs_seg_masks[pred_to_keep_mask]

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}#,'pred_masks':masks}
        out2= {'pred_logits': outputs_class2[-1], 'pred_boxes': outputs_coord2[-1]}#,'pred_masks':masks}
        out["pred_masks"] = outputs_seg_masks
        out2["pred_masks"] = outputs_seg_masks

        if self.aux_loss:
            out['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            out2['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class2[:-1], outputs_coord2[:-1])]
        
        return out,out2

    def training_epoch_start(self, *args, **kwargs):
        for k,v in self.weight_dict.items():
            #convert v to tensor and put it on the device
            self.weight_dict[k]=torch.as_tensor(v,device=self.device)

    def training_step(self, batch, batch_idx):
        samples, targets ,classencodings,masks,batch_idx,(tgt_ids,tgt_bbox,tgt_masks,tgt_sizes)= batch
        #targets = [{k: v.to(self.device,non_blocking=True) for k, v in t.items()} for t in targets]
        #print("classencodings",classencodings.keys())
        class_to_tensor=torch.zeros(max(list(classencodings.keys()))+1,device=self.device,dtype=torch.long) # find what the biggest index of classes is then make that many zeros. 
        for i,c in enumerate(classencodings.keys()):
            class_to_tensor[c]=i
            #we have to make this because it's not a given that the dictionary of classes has every key, nor that they're the same size
        # tensor_index_to_class=torch.as_tensor(list(classencodings.keys()),device=self.device)
        classencodings = torch.stack([v.squeeze() for v in classencodings.values()]).to(self.device,non_blocking=True)
        outputs,out2 = self(samples,classencodings,masks)

        # num_boxes = max(tgt_ids.shape[0], 1)
        embedding_indices=class_to_tensor[tgt_ids]

        tgt_embs= classencodings[embedding_indices] 
        tgt_embs=tgt_embs/torch.norm(tgt_embs,dim=-1,keepdim=True)
        tgt_masks=interpolate(tgt_masks.unsqueeze(1),outputs['pred_masks'].shape[-2:]).squeeze(1).to(outputs['pred_masks']) # BB,W,H
        masks=interpolate(masks.to(outputs['pred_masks']).unsqueeze(1),outputs['pred_masks'].shape[-2:]).squeeze(1) # B,W,H
        num_boxes = max(tgt_ids.shape[0], 1)
        #loss_dict, predictions= self.criterion(classencodings,outputs,num_boxes=num_boxes,tgt_sizes=tgt_sizes,tgt_ids=tgt_ids,tgt_bbox=tgt_bbox,class_lookup=class_to_tensor)
  





        loss_dict, predictions,boxes= self.criterion(classencodings=classencodings,targets=targets,num_boxes=num_boxes,outputs=outputs, tgt_masks=tgt_masks,tgt_embs=tgt_embs,tgt_sizes=tgt_sizes,tgt_ids=tgt_ids,tgt_bbox=tgt_bbox,im_masks=masks,batch_idx=batch_idx)
#        losses=sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        loss_dict2,predictions2,boxes2 = self.criterion(classencodings=classencodings,targets=targets,num_boxes=num_boxes, outputs=outputs, tgt_masks=tgt_masks,tgt_embs=tgt_embs,tgt_sizes=tgt_sizes,tgt_ids=tgt_ids,tgt_bbox=tgt_bbox,im_masks=masks,batch_idx=batch_idx)
        


        #log the images with boxes 
        if batch_idx%100==0:
            #images need unnormalized, to  be in 0-255, and in CHW
            images=samples[0].cpu().numpy()*255
            #then multiple boxes up by HW to get the right size
            boxes=boxes.cpu().numpy()*200
            #then draw boxes
            # this..... is not how to do it ! 
            self.log( "train_images", [wandb.Image(Images=images, boxes=boxes[i]) for i in range(images.shape[0])],prog_bar=False,rank_zero_only=True)

        logits=predictions/torch.norm(predictions,dim=-1,keepdim=True)
        logits2=predictions2/torch.norm(predictions2,dim=-1,keepdim=True)
        # print("logits",logits.shape)
        # print("logits2",logits2.shape)
        CELoss=self.loss(logits@logits2.T,torch.arange(logits.shape[0],device=self.device))
        
        
        loss_dict['CELoss']=CELoss
        loss_dict2['CELoss']=CELoss
        #losses2 = sum(loss_dict2[k] * self.weight_dict[k] for k in loss_dict2.keys() if k in self.weight_dict)
        
        losses = reduce(torch.add, [loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict])
        losses2= reduce(torch.add, [loss_dict2[k] * self.weight_dict[k] for k in loss_dict2.keys() if k in self.weight_dict])
        for k, v in loss_dict.items():
            if k in self.weight_dict:
                self.log(k,v * self.weight_dict[k],prog_bar=True,enable_graph=False,rank_zero_only=True)
        
        self.log("loss",losses,enable_graph=False,rank_zero_only=True)
        return losses +losses2
   
    def test_epoch_start(self,*args):
        self.cocoann=self.trainer.datamodule.test.coco
        #model = AutoModelForObjectDetection.from_pretrained("MariaK/detr-resnet-50_finetuned_cppe5")
        self.evalmodule = evaluate.load("ybelkada/cocoevaluate", coco=self.coco_ann)
    

    def test_step(self,batch,batch_idx):

        samples, targets ,classencodings,masks = batch
        class_to_tensor=torch.zeros(max(list(classencodings.keys()))+1,device=self.device,dtype=torch.long) # find what the biggest index of classes is then make that many zeros. 
        for i,c in enumerate(classencodings.keys()):
            class_to_tensor[c]=i
            #we have to make this because it's not a given that the dictionary of classes has every key, nor that they're the same size
        tensor_index_to_class=torch.as_tensor(list(classencodings.keys()),device=self.device)
    
        samples = samples.to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        classencodings = {k: v.to(self.device,non_blocking=True) for k, v in  classencodings.items()}
        classencodings=torch.stack(list(classencodings.values()))
        outputs, _ = self(samples, classencodings,masks)# we need to find coco classes for this!?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        scores,labels,boxes = self.postprocessors['bbox'](outputs, orig_target_sizes,classencodings)    
        # lookup labels against the main class inx labels=
        labels=tensor_index_to_class[labels]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        if 'segm' in self.postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = self.postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        # I Probably want .evaluate() here

        # if self.panoptic_evaluator is not None:
        #     res_pano = self.postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     self.panoptic_evaluator.update(res_pano)
        # outputs={target['image_id']: output for target, output in zip(targets, results)}
        #self.coco_evaluator.update(outputs)
        self.evalmodule.add(prediction=results, reference=targets)
        # return outputs
    def test_epoch_end(self,outputs):
        #self.coco_evaluator.synchronize_between_processes()
        results = self.evalmodule.compute()
        self.log("mAP",results["mAP"], on_epoch=True, prog_bar=True, logger=True)
        self.print(results)
        #         iou_types = tuple(k for k in ('segm', 'bbox') if k in self.postprocessors.keys())
        #         #print(self.trainer.datamodule.__dir__())
        #         #point the coco eval at the underlying dataset
        #         self.coco_evaluator = CocoEvaluator(self.trainer.datamodule.val.coco, iou_types)
                                                    
        #         self.panoptic_evaluator = None
        #         self.coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

        #         for output in outputs:
        #             self.coco_evaluator.update(output)
        #         self.coco_evaluator.synchronize_between_processes()
        #         self.coco_evaluator.accumulate()
        #         self.coco_evaluator.summarize()
        #         #self.log("mAP",self.coco_evaluator.coco_eval["bbox"].stats[0], prog_bar=True, logger=True)
        #         all_ids={}

        #         if self.panoptic_evaluator is not None:
        #             panoptic_res = self.panoptic_evaluator.summarize()
        #             self.log("PQ_all", panoptic_res["All"])
        #             self.log("PQ_th", panoptic_res["Things"])
        #             self.log("PQ_st", panoptic_res["Stuff"])
        # #        if self.coco_evaluator is not None:
        #         #if 'bbox' in self.postprocessors.keys():
        #         #    self.log('coco_eval_bbox',self.coco_evaluator.coco_eval['bbox'].stats.tolist())
        #         if 'segm' in self.postprocessors.keys():
        #             self.log('coco_eval_masks',self.coco_evaluator.coco_eval['segm'].stats.tolist())
import clip


class VisGenomeModule(PairDETR):
    #in this module, we're going to do the same as above, only we need to also process the masks first with DETIC as in the ClipToMask module
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
        self.clip, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.detic = DefaultPredictor(self.cfg)
        self.detic.model.eval()

    @torch.no_grad()
    def detic_forward(self,**batch):
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
            else:    
                best_obj_boxes=torch.nn.functional.gumbel_softmax(object_box_ious, tau=1, hard=False, eps=1e-10, dim=0)
                best_subj_boxes=torch.nn.functional.gumbel_softmax(subject_box_ious,tau=1, hard=False, eps=1e-10,dim=0)
                masks=masks.flatten(1)
                masks=torch.logical_or(best_obj_boxes@masks,best_subj_boxes@masks).float()
                masks=masks.unflatten(1,(1, 28,28))
                all_masks.append(masks)
                spans.append(len(masks))
                index_arrays.append(torch.logical_or(best_obj_boxes,best_subj_boxes))
               
        masks_per_caption=torch.cat(all_masks,dim=0)
        masks_per_image=torch.stack([torch.sum(masks,dim=0).clamp(0,1) for masks in all_masks])
        assert masks_per_caption.shape[0]==sum(spans)
        return masks_per_caption,masks_per_image,spans

    def do_batch(self,batch):
        #in visual genome, we have a set of relations for an image. Boxes are provided for sub and obj but still pin to each relationship. 
        img=batch["img"]
        obj_classes=batch["obj_classes"].squeeze()
        subj_classes=batch["subj_classes"].squeeze()
        objects=batch["objects"] # bbox in xyxy format
        subjects=batch["subjects"] # bbox in xyxy format
        batch_idx=batch["batch_idx"]
        captions=batch["relation"].squeeze()
        masks_per_caption,masks_per_image=self.detic_forward(**batch)
        classencodings=captions if captions.shape[-1] ==512 else self.clip.encode_text(captions)
        #targets are the coco format annotations  
        #tgt_ids is which encoding each caption has. 
        tgt_ids=torch.arange(classencodings.shape[0],device=self.device)       
        #tgt_bbox is all the concatenates bboxes
        tgt_bbox=torch.stack([torch.min(objects[:,0],subjects[:,0]).values,
                                    torch.min(objects[:,1],subjects[:,1]), # these find the top left corner
                                torch.max(objects[:,2],subjects[:,2]), # find the bottom right corner with max of x ys and add the whs.  
                            torch.max(objects[:,3],subjects[:,3])],dim=1)
        #tgt_sizes is the number of labels per image
        
        tgt_sizes= torch.nn.funtional.one_hot(batch_idx,num_classes=img.shape[0]).sum(dim=0)
        targets = [dict(zip(["labels","boxes","masks","boxes"],v)) for v in zip(tgt_ids,tgt_bbox,masks_per_caption,tgt_sizes)]

        return img, targets , classencodings, masks_per_image, batch_idx, (tgt_ids,tgt_bbox,masks_per_caption,tgt_sizes)
    def training_step(self,batch,batch_idx):
       
        return super().training_step(self.do_batch(batch),batch_idx)
    def test_step(self,batch,batch_idx):
        return super().test_step(self.do_batch(batch),batch_idx)
    def validation_step(self,batch,batch_idx):
        return super().validation_step(self.do_batch(batch),batch_idx)
    


if __name__ == '__main__':
    from argparser import get_args_parser
    import argparse
    import wandb
    parser = argparse.ArgumentParser('Conditional DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    savepath=args.coco_path
    args = vars(args)


    # #make wandb logger
    # run=wandb.init(project="SPARC",entity="st7ma784",name="VRE",config=args)

    # logtool= pl.loggers.WandbLogger( project="SPARC",entity="st7ma784",experiment=run, save_dir=savepath,log_model=True)

    # #wandb_logger = WandbLogger(project='pairdetr',entity="st7ma784",log_model=True)
    # from data.coco import COCODataModule
    # data=COCODataModule(Cache_dir=args.coco_path,batch_size=4)
    # #convert to dict
    # model=PairDETR(**args)


    #or use VisGenomeForTraining....
    from data.VisGenomeDataModule import VisGenomeDataModule
    data =VisGenomeDataModule(Cache_dir=savepath,batch_size=4)
    data.prepare_data()
    data.setup()
    model=VisGenomeModule(**args)
    run=wandb.init(project="SPARC-VisGenome",entity="st7ma784",name="VRE-Vis",config=args)

    logtool= pl.loggers.WandbLogger( project="SPARC-VisGenome",entity="st7ma784",experiment=run, save_dir=savepath,log_model=True)

    

    trainer = pl.Trainer(
                         precision=16,
                         max_epochs=20,#args['epochs'], 
                         num_sanity_val_steps=0,
                         gradient_clip_val=0.25,
                         accumulate_grad_batches=1,
                         logger=logtool,
                         #callbacks=[ModelCheckpoint(dirpath=args['output_dir'],save_top_k=1,monitor='val_loss',mode='min')],
                         accelerator='auto',
                         fast_dev_run=False,  
                         devices="auto",
                            )
    trainer.fit(model,data)
    trainer.test(model,data)
