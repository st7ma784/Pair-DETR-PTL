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
from models import build_model
import pytorch_lightning as pl

class PairDETR(pl.LightningModule):
    def __init__(self, *args,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model, self.criterion, self.postprocessors = build_model(self.args)
    #config optimizer
    def configure_optimizers(self):
            
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_drop)
        return [optimizer], [lr_scheduler]
    
    def setup(self, stage: str | None = None) -> None:
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

    