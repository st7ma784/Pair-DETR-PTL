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

# import util.misc as utils
from model import *
import pytorch_lightning as pl
import clip
from transformers import CLIPTokenizer
class PairDETR(pl.LightningModule):
    def __init__(self,**args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        #self.kwargs = kwargs
        self.learning_rate = args['lr']
        # num_classes = 20 if args['dataset_file'] != 'coco' else 91
        # if args['dataset_file'] == "coco_panoptic":
        #     num_classes = 250
        # self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.Tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenize=lambda x: self.Tokenizer(x, # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids']
        myclip,_=clip.load('ViT-B/32',device=self.device)
        self.clip_projection= myclip.text_projection
        posmethod=PositionEmbeddingLearned
        if args['position_embedding'] in ('v2', 'sine'):
            #TODO find a better way of exposing other arguments
            posmethod = PositionEmbeddingSine
        self.positional_embedding = posmethod(args['hidden_dim'] // 2,device=self.device)
        self.backbone = Backbone(args['backbone'], True, False, args['dilation'])
        #self.backbone.num_channels = self.backbone.num_channels
        self.transformer = PositionalTransformer(
            d_model=args['hidden_dim'],
            dropout=args['dropout'],
            nhead=args['nheads'],
            num_queries=args['num_queries'],
            dim_feedforward=args['dim_feedforward'],
            num_encoder_layers=args['enc_layers'],
            num_decoder_layers=args['dec_layers'],
            normalize_before=args['pre_norm'],
            return_intermediate_dec=args['intermediate_layer'],
            device=self.device,
        )
        self.num_queries = args['num_queries']
        hidden_dim = self.transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 6)
        self.query_embed = nn.Parameter(nn.Embedding(self.num_queries, hidden_dim).weight)
        self.query_embed.to(self.device) # changing to nn,Parameter(*)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        self.aux_loss = args['aux_loss']

        # init prior_prob setting for focal loss
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        #self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        # if args.masks:
        #     model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
        self.matcher = HungarianMatcher(cost_class=args['set_cost_class'], cost_bbox=args['set_cost_bbox'], cost_giou=args['set_cost_giou'])

        self.weight_dict = {'loss_ce': args['cls_loss_coef'],
                       'loss_bbox': args['bbox_loss_coef'],
                       'loss_giou': args['giou_loss_coef']}

        self.criterion = SetCriterion(matcher=self.matcher, weight_dict=self.weight_dict,
                                focal_alpha=args['focal_alpha'], losses=['labels', 'boxes', 'cardinality'])
        self.postprocessors = {'bbox': PostProcess()}
        
    #config optimizer

    

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        #features, pos = self.backbone(samples) this called Joiner which is a wrapper for the backbone:

        src,mask = self.backbone(samples.tensors,samples.mask)
        #print(src.shape)
        src,mask=src[-1],mask[-1]
        #src,mask=xs["0"].decompose() # pull it into the tensor, mask. 
        #print(src.shape,mask.shape) #orch.Size([16, 2048, 38, 35]) torch.Size([16, 38, 35])
        assert mask is not None


        # print("self.input_proj(src)",self.input_proj(src).shape)#([8, 256, 29, 34])
        # print("self.query_embed.weight",self.query_embed.weight.shape)#([300, 256])
        # print("self.positional_embedding(src,mask).to(src.dtype)",self.positional_embedding(src,mask).to(src.dtype).shape)#([8, 256, 29, 34])
        # print("mask",mask.shape)#mask torch.Size([8, 29, 34])

        #print("h,w",src.shape[-2],src.shape[-1])# now regular size! 
        hs, reference = self.transformer(self.input_proj(src), mask, self.query_embed, self.positional_embedding(src).to(src.dtype))
        #
        #print("reference",reference.shape) # B, n_q, 2
        #print("reference_before_sigmoid",reference_before_sigmoid.shape) #   B, n_q, 2
        # outputs_coords = []
        #hs torch.Size([1, 200, 14, 512])
        #torch.Size([14, 200, 2])
        #print("hs",hs.shape) # torch.Size([layers, B, n_queries, F]
        #print(reference.shape) # torch.Size([layers, B, n_queries, 2]
        tmp=self.bbox_embed(hs)
        
        tmp[..., :2] +=  inverse_sigmoid(reference) 
        outputs_coord = tmp.sigmoid()
        
        outputs_class = hs@self.clip_projection # torch.Size([9, B, n_queries, C])
        #print("outputs_class",outputs_class.shape)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out
    # this can be optimized better! to remove the above if statement! 
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def configure_optimizers(self):
            
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                    weight_decay=self.args['weight_decay'])
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args['lr_drop'])
        return [optimizer]

    def training_step(self, batch, batch_idx):
        samples, targets ,classencodings = batch
        #print("batch targets",targets)
        #print("batch classes",classes.shape)
        #targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        outputs = self(samples)

        # I also want a list of the class labels from COCO to be comparing against in the first instance, this is essentially the classes we're told to go and grab, and the labels are the GT, which when deployed will come from Text to as well.
        loss_dict = self.criterion(classencodings,outputs, targets)

        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

        loss_dict_scaled = {k: v * self.weight_dict[k]
                                    for k, v in loss_dict.items() if k in self.weight_dict}
        for k,v in loss_dict_scaled.items():
            self.log(k,v,prog_bar=True,enable_graph=False)
        #self.log('train_loss', loss_value)
        return losses
   
    def on_test_start(self) -> None:
        print(" Looking for Datamodule :\n",self.__dir__())     
    #     iou_types = tuple(k for k in ('segm', 'bbox') if k in self.postprocessors.keys())
    #     # print(self.data.__dir__())
    #     self.coco_evaluator = CocoEvaluator(self.val_dataloader().dataset, iou_types)
    #     self.panoptic_evaluator = None
    #     if 'panoptic' in self.postprocessors.keys():
    #         self.panoptic_evaluator = PanopticEvaluator(
    #             self.val_dataloader().dataset.ann_file,
    #             self.val_dataloader().dataset.ann_folder,
    #     )
    #     return super().on_test_start()
    # def test_step(self, batch, batch_idx):
    


    # # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

        
    #     samples, targets = batch
    #     samples = samples.to(self.device)
    #     targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

    #     outputs = self.model(samples)
    #     loss_dict = self.criterion(outputs, targets)
    #     weight_dict = self.criterion.weight_dict

    #     # reduce losses over all GPUs for logging purposes
    #     loss_dict_reduced = utils.reduce_dict(loss_dict)
    #     loss_dict_reduced_scaled = {k: v * weight_dict[k]
    #                                 for k, v in loss_dict_reduced.items() if k in weight_dict}
    #     loss_dict_reduced_unscaled = {f'{k}_unscaled': v
    #                                   for k, v in loss_dict_reduced.items()}
    #     self.log("loss",sum(loss_dict_reduced_scaled.values()))
    #     #  **loss_dict_reduced_scaled,
    #     #  **loss_dict_reduced_unscaled)
    #     self.log("class_error",loss_dict_reduced['class_error'])

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
    from pytorch_lightning.callbacks import ModelCheckpoint
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    from datasets.coco import COCODataModule
    data=COCODataModule(Cache_dir=args.coco_path,batch_size=8)
    #convert to dict
    args = vars(args)
    model=PairDETR(**args)
    trainer = pl.Trainer(
                         precision=32,
                         max_epochs=args['epochs'], 
                         num_sanity_val_steps=0,
                        #  gradient_clip_val=0.25,
                         accumulate_grad_batches=4,
                         #callbacks=[ModelCheckpoint(dirpath=args['output_dir'],save_top_k=1,monitor='val_loss',mode='min')],
                         accelerator='cuda',
                         fast_dev_run=False,  
                         devices="auto",
    )
    trainer.fit(model,data)
    #trainer.test(model,data)
