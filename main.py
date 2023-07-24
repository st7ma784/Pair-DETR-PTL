# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------


from pathlib import Path
# from datasets.coco_eval import CocoEvaluator
# from datasets.panoptic_eval import PanopticEvaluator

# from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
# from torch.utils.data import DataLoader
from util.misc import inverse_sigmoid

from model import *
import pytorch_lightning as pl
from transformers import CLIPTokenizer

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
        self.matcher = HungarianMatcher(cost_class=args['set_cost_class'], 
                                        cost_bbox=args['set_cost_bbox'],
                                        cost_giou=args['set_cost_giou'])

        self.weight_dict = {'loss_ce': args['cls_loss_coef'],
                       'loss_bbox': args['bbox_loss_coef'],
                       'loss_giou': args['giou_loss_coef'],
                       'loss_dice': args['dice_loss_coef'],
                       'loss_mask': args['mask_loss_coef'],
                       'CELoss':0.25}

        self.criterion = SetCriterion(matcher=self.matcher, 
                                      weight_dict=self.weight_dict,
                                    focal_alpha=args['focal_alpha'],
                                     losses=['labels', 'boxes', 'cardinality','masks'],
                                     device=self.device)
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

    def forward(self, samples: NestedTensor,classencoding):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        #features, pos = self.backbone(samples) this called Joiner which is a wrapper for the backbone:

        src = self.backbone(samples.tensors)
        src,feats=src[-1],src[:-1]
        mask=F.interpolate(samples.mask[None].float(),size=src.shape[-2:]).bool()[-1]        
        #my class encoding is going to be shape (n_classes, 512) I want this query embed to be (n_queries,hidden_dim)

        #print(self.query_embed.shape)
        classencodings=self.clip_projection(classencoding)
        #repeat by the number of queries
        classencodings=classencodings.repeat(1,self.num_queries,1).flatten(0,1)
        #print("classencs",classencoding.shape)
        #print("src shape",src.shape,src.device)
        

        outputs_coord, outputs_class ,outputs_coord2, outputs_class2= self.transformer( self.input_proj(src) , classencodings, self.positional_embedding(src,mask), mask)
        #print("outputs_coord",outputs_coord.shape) # 1 ,24,240,4
        #print("outputs_class",outputs_class.shape) # 1,24,240,512
        #filter these outputs by the cosine similarity of the class encodings, we want to find the most similar class encoding to each of the outputs, and ensure that its above a threshold

        #similarities=torch.einsum('bqf,gf->bqg', outputs_class[-1], classencoding.squeeze())# this takes output classes of shape (1,24,240,512) and class encodings of shape (240,512) and returns (1,24,240,n_classes)
        #use this as a mask to filter the outputs
        #pred_to_keep_mask=torch.max(similarities,dim=-1).values>self.threshold

        #print("max_similarities",pred_to_keep_mask.shape) #B,C
        bbox_mask = self.bbox_attention(outputs_class[-1], classencodings, mask=mask)
        #reverse the order of feats 
        feats=feats[::-1]
        seg_masks = self.mask_head(self.input_proj(src), bbox_mask, feats)
        #print("seg_masks",seg_masks.shape)
        outputs_seg_masks = seg_masks.view(src.shape[0], -1, seg_masks.shape[-2], seg_masks.shape[-1])
        #print("outputs_seg_masks",outputs_seg_masks.shape)
#         return out





        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}#,'pred_masks':masks}
        out2= {'pred_logits': outputs_class2[-1], 'pred_boxes': outputs_coord2[-1]}#,'pred_masks':masks}
        out["pred_masks"] = outputs_seg_masks
        out2["pred_masks"] = outputs_seg_masks

        if self.aux_loss:
            out['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            out2['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class2[:-1], outputs_coord2[:-1])]
        
        return out,out2



    def training_step(self, batch, batch_idx):
        samples, targets ,classencodings = batch
        targets = [{k: v.to(self.device,non_blocking=True) for k, v in t.items()} for t in targets]
        classencodings = {k: v.to(self.device,non_blocking=True) for k, v in  classencodings.items()}
        outputs,out2 = self(samples,torch.stack(list(classencodings.values())))
        count=0
        for k in outputs.keys():
            
            if isinstance(outputs[k],Tensor) and torch.any(torch.isnan(outputs[k])):
                #we should really work out how and why this happens! 
                print("{} has nan".format(k))
                outputs[k]=torch.nan_to_num(outputs[k],nan=1e-8,posinf=0.0,neginf=0.0)
                count+=1
        if count>1:
            return None

        # I also want a list of the class labels from COCO to be comparing against in the first instance, this is essentially the classes we're told to go and grab, and the labels are the GT, which when deployed will come from Text to as well.
        loss_dict, predictions= self.criterion(classencodings,outputs, targets)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        loss_dict2,predictions2 = self.criterion(classencodings,out2, targets)
        # print("predictions",predictions.shape)
        # print("predictions2",predictions2.shape)
        logits=predictions/torch.norm(predictions,dim=-1,keepdim=True)
        logits2=predictions2/torch.norm(predictions2,dim=-1,keepdim=True)
        # logits=logits.flatten(0,1)
        # logits2=logits2.flatten(0,1)
        CELoss=self.loss(logits@logits2.T,torch.arange(logits.shape[0],device=self.device))
        
        
        loss_dict['CELoss']=CELoss
        loss_dict2['CELoss']=CELoss
        losses2 = sum(loss_dict2[k] * self.weight_dict[k] for k in loss_dict2.keys() if k in self.weight_dict)
        loss_dict_scaled = {k: v * self.weight_dict[k]
                                    for k, v in loss_dict.items() if k in self.weight_dict}
        for k,v in loss_dict_scaled.items():
            self.log(k,v,prog_bar=True,enable_graph=False)
        #self.log('train_loss', loss_value)
        return losses+losses2
   
    def on_test_start(self) -> None:
        print(" Looking for Datamodule :\n",self.__dir__())     
    #     iou_types = tuple(k for k in ('segm', 'bbox') if k in self.postprocessors.keys())
    #     # print(self.data.__dir__())
    #     self.coco_evaluator = CocoEvaluator(self.val_dataloader().dataset, iou_types)
    #     self.panoptic_evaluator = None
          # self.coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]



    # def test_step(self, batch, batch_idx):
    #
    #     samples, targets ,classencodings = batch
    #     samples = samples.to(self.device)
    #     targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

    #     outputs, out2= self(samples, torch.stack(list(classencodings.values())) )# we need to find coco classes for this!?
    #     loss_dict = self.criterion(outputs, targets)
    #     weight_dict = self.criterion.weight_dict
    #     orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    #     results = self.postprocessors['bbox'](outputs, orig_target_sizes)
    #     if 'segm' in self.postprocessors.keys():
    #         target_sizes = torch.stack([t["size"] for t in targets], dim=0)
    #         results = self.postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
    #     res = {target['image_id'].item(): output for target, output in zip(targets, results)}
    #     if self.coco_evaluator is not None:
    #         self.coco_evaluator.update(res)

    #     if self.panoptic_evaluator is not None:
    #         res_pano = self.postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
    #         for i, target in enumerate(targets):
    #             image_id = target["image_id"].item()
    #             file_name = f"{image_id:012d}.png"
    #             res_pano[i]["image_id"] = image_id
    #             res_pano[i]["file_name"] = file_name

    #         self.panoptic_evaluator.update(res_pano)
    # def on_test_end(self):
    #     if self.coco_evaluator is not None:
    #         self.coco_evaluator.accumulate()
    #         self.coco_evaluator.summarize()
    #     panoptic_res = None
    #     if self.panoptic_evaluator is not None:
    #         panoptic_res = self.panoptic_evaluator.summarize()
        
    #     if self.coco_evaluator is not None:
    #         if 'bbox' in self.postprocessors.keys():
    #             self.log('coco_eval_bbox',self.coco_evaluator.coco_eval['bbox'].stats.tolist())
    #         if 'segm' in self.postprocessors.keys():
    #             self.log('coco_eval_masks',self.coco_evaluator.coco_eval['segm'].stats.tolist())
    #     if panoptic_res is not None:
    #         self.log("PQ_all", panoptic_res["All"])
    #         self.log("PQ_th", panoptic_res["Things"])
    #         self.log("PQ_st", panoptic_res["Stuff"])
    #     return super().on_test_end()


if __name__ == '__main__':
    from argparser import get_args_parser
    import argparse
    parser = argparse.ArgumentParser('Conditional DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    from datasets.coco import COCODataModule
    data=COCODataModule(Cache_dir=args.coco_path,batch_size=4,return_masks=True)
    #convert to dict
    args = vars(args)
    model=PairDETR(**args)
    trainer = pl.Trainer(
                         precision=16,
                         max_epochs=args['epochs'], 
                         num_sanity_val_steps=0,
                         gradient_clip_val=0.25,
                         accumulate_grad_batches=4,
                         #callbacks=[ModelCheckpoint(dirpath=args['output_dir'],save_top_k=1,monitor='val_loss',mode='min')],
                         accelerator='auto',
                         fast_dev_run=False,  
                         devices="auto",
                            )
    trainer.fit(model,data)
    #trainer.test(model,data)
