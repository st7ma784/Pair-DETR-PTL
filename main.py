# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------


from pathlib import Path
from typing import Optional 
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from model import *
import pytorch_lightning as pl

class ConditionalDETR(nn.Module):
    """ This is the Conditional DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

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
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs, reference = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
        
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed(hs[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)

        outputs_class = self.class_embed(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class PairDETR(pl.LightningModule):
    def __init__(self, *args,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.kwargs = kwargs
        num_classes = 20 if args.dataset_file != 'coco' else 91
        if args.dataset_file == "coco_panoptic":
            num_classes = 250
        posmethod=PositionEmbeddingLearned
        if args.position_embedding in ('v2', 'sine'):
            # TODO find a better way of exposing other arguments
            posmethod = PositionEmbeddingSine
        position_embedding = posmethod(args.hidden_dim // 2)
        backbone = Joiner(Backbone(args.backbone, True, False, args.dilation), position_embedding)
        backbone.num_channels = backbone.num_channels
        
        transformer = PositionalTransformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            num_queries=args.num_queries,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
        #Dear future me , we really want to be extending this class as it's where our positional encodings are found! 
        #good Luck! 
        self.model = ConditionalDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
        )
        # if args.masks:
        #     model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
        self.matcher = HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)

        self.weight_dict = {'loss_ce': args.cls_loss_coef,
                       'loss_bbox': args.bbox_loss_coef,
                       'loss_giou': args.giou_loss_coef}

        losses = ['labels', 'boxes', 'cardinality']
        # if args.masks:
        #     losses += ["masks"]
        self.criterion = SetCriterion(num_classes, matcher=self.matcher, weight_dict=self.weight_dict,
                                focal_alpha=args.focal_alpha, losses=losses)
        self.postprocessors = {'bbox': PostProcess()}
        
    #config optimizer
    def configure_optimizers(self):
            
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_drop)
        return [optimizer], [lr_scheduler]
    
    def setup(self, stage = None) -> None:
        self.train_dataset = build_dataset(image_set='train', args=self.args)
        self.val_dataset = build_dataset(image_set='val', args=self.args)
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, 
                          collate_fn=utils.collate_fn, 
                          num_workers=args.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          collate_fn=utils.collate_fn, 
                          #can add some speed ups here
                          num_workers=args.num_workers)
    def training_step(self, batch, batch_idx):
        samples, targets = batch
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        
        self.log('loss',losses_reduced_scaled)
        # **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        self.log('class_error',loss_dict_reduced['class_error'])
        self.log('train_loss', loss_value)
        return losses
    def on_validation_start(self) -> None:
        
        iou_types = tuple(k for k in ('segm', 'bbox') if k in self.postprocessors.keys())
        
        self.coco_evaluator = CocoEvaluator(self.val_dataset, iou_types)
        self.panoptic_evaluator = None
        if 'panoptic' in self.postprocessors.keys():
            self.panoptic_evaluator = PanopticEvaluator(
                self.val_dataset.ann_file,
                self.val_dataset.ann_folder,
        )
        return super().on_validation_start()
    def validation_step(self, batch, batch_idx):
    


    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

        
        samples, targets = batch
        samples = samples.to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        self.log("loss",sum(loss_dict_reduced_scaled.values()))
        #  **loss_dict_reduced_scaled,
        #  **loss_dict_reduced_unscaled)
        self.log("class_error",loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in self.postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = self.postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if self.coco_evaluator is not None:
            self.coco_evaluator.update(res)

        if self.panoptic_evaluator is not None:
            res_pano = self.postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            self.panoptic_evaluator.update(res_pano)
    def on_validation_end(self):
        if self.coco_evaluator is not None:
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
        panoptic_res = None
        if self.panoptic_evaluator is not None:
            panoptic_res = self.panoptic_evaluator.summarize()
        
        if self.coco_evaluator is not None:
            if 'bbox' in self.postprocessors.keys():
                self.log('coco_eval_bbox',self.coco_evaluator.coco_eval['bbox'].stats.tolist())
            if 'segm' in self.postprocessors.keys():
                self.log('coco_eval_masks',self.coco_evaluator.coco_eval['segm'].stats.tolist())
        if panoptic_res is not None:
            self.log("PQ_all", panoptic_res["All"])
            self.log("PQ_th", panoptic_res["Things"])
            self.log("PQ_st", panoptic_res["Stuff"])
        return super().on_validation_end()


if __name__ == '__main__':
    from argparser import get_args_parser
    import argparse
    parser = argparse.ArgumentParser('Conditional DETR training and evaluation script', parents=[get_args_parser()])
    from pytorch_lightning.callbacks import ModelCheckpoint
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model=PairDETR(*args)
    trainer = pl.Trainer(gpus=1,
                         precision=16,
                         max_epochs=args.epochs, 
                         num_sanity_val_steps=0,
                         gradient_clip_val=args.clip_grad,
                         callbacks=[ModelCheckpoint(dirpath=args.output_dir,save_top_k=1,monitor='val_loss',mode='min')],
                         accelerator='ddp',  
                            )
    trainer.fit(model)
    trainer.test(model)

