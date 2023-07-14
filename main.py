# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import datetime
import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
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
def main(args):
    
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.dataset_file == "coco_panoptic":
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    output_dir = Path(args.output_dir)
   
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(
            model, criterion, data_loader_train, optimizer, epoch,
            args.clip_max_norm)
        lr_scheduler.step()

        _, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, args.output_dir
        )

       
        if args.output_dir and utils.is_main_process():

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)


if __name__ == '__main__':
    from argparser import get_args_parser
    import argparse
    parser = argparse.ArgumentParser('Conditional DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
