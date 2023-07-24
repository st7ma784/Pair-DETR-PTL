# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, interpolate)
from util.misc import inverse_sigmoid
from models.attention import MultiheadAttention

from typing import Optional, List
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
Tensor = torch.Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List, Optional
import itertools
from scipy.optimize import linear_sum_assignment

import io
from collections import defaultdict
from PIL import Image

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias



class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns):

        # print("in mask conv head")
        # print("x shape",x.shape) # B, F, H,W
        # print("bbox mask shape",bbox_mask.shape) # B, 240, n_heads, 1, 1 
        # for i,fpn in enumerate(fpns):
        #     print("fpn {} shape {}".format(i,fpn.shape))
        # #        print("fpn {} shape {}".format(1,fpns[0].shape)) # B, F, H,W
        x =torch.cat([x.repeat(bbox_mask.shape[1],1,1,1), bbox_mask.repeat(1,1,1,x.shape[-2],x.shape[-1]).flatten(0, 1)], dim=1)
        #print("x shape",x.shape) # B, F+240, H,W
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x



def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)
class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):

        #B,P 4,
        #B,P,F
        #this function takes the query and key values as batch, Predictions , F and batch, Predictions , 2 
        # and calculates where to look in an image for each query
        # print("Q shape",q.shape) # NQs,4 # BBoxs for each prediction
        # print("k shape",k.shape) # NQs,Class # Class ref for each prediction
        q = self.q_linear(q) # B,P,hidden_dim
        k = self.k_linear(k) # P,hidden_dim
        # print("q",q.shape) # B,P,hidden_dim
        # print("k",k.shape) # P,hidden_dim//numheads
        q= q.permute(1,2,0) # P,hidden_dim,B
        k=k.permute(1,0).unsqueeze(-1).unsqueeze(-1) # P,hidden_dim,1,1
        #k = F.conv2d(q,k,stride=1) # P,B,P
        # print("k",k.shape) # preds , preds , B # but I want this to be B, F, H, W
        
        # print("qh",q.shape) # preds ,B, preds,1,1")
        qh = q.reshape(q.shape[2], q.shape[0], self.num_heads, self.hidden_dim // self.num_heads) # shape B,numQ,Nheads,hidden//Nheads
        #           Preds ,     B,     Preds,           1,                  1
        kh = k.reshape(k.shape[1], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1]) # shape B,numQ,hidden//Nheads,H,W
        # print("qh",qh.shape) # B, Q  , Nheads,hidden//Nheads
        # print("kh2",kh.shape) # Q ,Nheads,hidden//Nheads,1,1
        weights = torch.einsum("bqnc,qnchw->bqnhw", qh * self.normalize_fact, kh)#.flatten(2,3)
        if mask is not None:
            # print("mask",mask.shape) # B,_1,_1, 25,25
            # print("weights",weights.shape) # B, Q  , Nheads,H,W
            weights.repeat(1,1,1,mask.shape[-2],mask.shape[-1]).masked_fill_(mask.unsqueeze(1).unsqueeze(1).repeat(1,qh.shape[1],self.num_heads,1,1), float("-inf"))
        # print("weights",weights.shape) # B, Q  , Nheads,H,W
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        # print("weights",weights.shape) # B, Q  , Nheads,H,W
        #but I want B,Preds,F,H,W
        return weights


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results


class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": a})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets,encodings):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            encodings: This is a list of encodings (len(Training Classes)), where each encoding is a dict containing:
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1)  # [batch_size * num_queries, F]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]#  this is nan somewhere ?!?!?!

        #print("out prob, out box",out_prob.shape, out_bbox.shape)#1600,F   1600,4
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]) # [n ]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_embs= torch.stack([encodings[int(i)] for i in tgt_ids],dim=0).squeeze()
        out_prob=out_prob/torch.norm(out_prob,dim=-1,keepdim=True) #can get away with no grad...
        tgt_embs=tgt_embs/torch.norm(tgt_embs,dim=-1,keepdim=True)
        cost_class=F.relu(out_prob@tgt_embs.T)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
       
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu().sigmoid()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        target_classes_o=torch.cat([encodings[int(i.item())] for t, (_, J) in zip(targets, indices) for i in t["labels"][J]])
        #print("target classes o",target_classes_o.shape)
        x,y=zip(*indices)
        src=torch.cat([torch.as_tensor(x) for x in x])
        tgt=torch.cat([torch.as_tensor(y) for y in y])
        idxs = torch.stack([src,tgt])
        #print("idxs",idxs.shape)
        batch_idx = torch.cat([torch.full_like(torch.as_tensor(x), i) for i, x in enumerate(x)]).unsqueeze(0)
        indices=torch.cat([batch_idx,idxs],dim=0)
        return indices, target_classes_o


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, focal_alpha, losses,device):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """

        super().__init__()
        self.device=device
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        if self.focal_alpha <0:
            self.focal_alpha = 0
        self.loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        self.loss=nn.CrossEntropyLoss(reduction="mean")
    def sigmoid_focal_loss(self,inputs, targets, num_boxes):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        """
        # #do similarity of inputs to targets,
        # print("inputs",inputs.shape) # inputs torch.Size([8, 200, 512])
        # print("targets",targets.shape) # targets torch.Size([8, 200, 512])
        inputs=inputs/torch.norm(inputs,dim=-1,keepdim=True)
        targets=targets/torch.norm(targets,dim=-1,keepdim=True)
        loss=1-torch.sum(inputs*targets,dim=-1)
        # loss=self.loss(match,torch.ones(match.shape,device=match.device))
        #loss=self.loss(inputs,targets)
        return loss.mean() / num_boxes

    def loss_labels(self,target_classes_o, outputs, targets, indices, num_boxes, log=True):
     
        assert 'pred_logits' in outputs

        idx= (indices[0], indices[2])
        src_idx= (indices[0], indices[1])
        src_logits = outputs['pred_logits'][src_idx]
        # target_classes = torch.zeros((*src_logits.shape[:2],512), device=src_logits.device) 
        # target_classes[idx] = target_classes_o
        # print(torch.allclose(target_classes[idx], target_classes_o))#True
        loss_ce = self.sigmoid_focal_loss(src_logits, target_classes_o, num_boxes) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, target_classes_o, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, target_classes_o, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = (indices[0],indices[1])
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.stack([targets[t]['boxes'][i] for (t, i) in zip(indices[0],indices[2])], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, target_classes_o,outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        #when I include this, I get a fail at line 641. 

        src_idx =(indices[0],indices[1])
        #print("src idx",src_idx)
        tgt_idx = (indices[0],indices[2])
        #print("tgt idx",tgt_idx)
        src_masks = outputs["pred_masks"]
        #print("src masks",src_masks.shape)
        src_masks = src_masks[src_idx]
        #print("src masks",src_masks.shape)
        
        masks = [t["masks"] for t in targets]
        #print("masks",[mask.shape for mask in masks])
        # TODO use _ to mask invalid areas due to padding in loss
        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        sc_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        return {
            "loss_mask": sigmoid_focal_loss(sc_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(sc_masks, target_masks, num_boxes),
        }

    def forward(self, encodings,outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices ,target_classes_o= self.matcher(outputs_without_aux, targets,encodings)
         
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=self.device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.loss_map[loss](target_classes_o,outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets,encodings)
                for loss in self.losses:
                    l_dict = self.loss_map[loss](target_classes_o,outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        #Jici suis? 
        
        src_idx= (indices[0], indices[1]) #0,1 was the original version, 
        # print("src idx",src_idx)
        # print("pred logits",outputs['pred_logits'].shape)
        src_logits = outputs['pred_logits'][src_idx]
        return losses, src_logits


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.Sequential(*list(itertools.chain.from_iterable([[nn.Linear(n, k),nn.ReLU()] for n, k in zip([input_dim] + h, h + [output_dim])]))[:-1])

    def forward(self, x):
        return self.layers(x)



class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None,device='cuda'):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.device=device

        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        
        dim_t = torch.arange(self.num_pos_feats//2, dtype=torch.float32,device=self.device)
        self.dim_t = self.temperature ** (2 * (dim_t) / self.num_pos_feats)
    def forward(self, x: Tensor, mask: Tensor):
        assert mask is not None
        not_mask = ~mask.to(self.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32).to(x.device)
        x_embed = not_mask.cumsum(2, dtype=torch.float32).to(x.device)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale


        pos_x = x_embed[:, :, :, None] / self.dim_t.to(x.device)
        pos_y = y_embed[:, :, :, None] / self.dim_t.to(x.device)
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # print(pos.device)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=512,device='cuda'):
        super().__init__()
        self.device=device
        self.row_embed = nn.Parameter(torch.ones(25, num_pos_feats,device=self.device))
        self.col_embed = nn.Parameter(torch.ones(25, num_pos_feats,device=self.device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed)
        nn.init.uniform_(self.col_embed)

    def forward(self, x: Tensor,mask: Tensor=None):
        return torch.cat([
            self.col_embed.unsqueeze(0).repeat(self.row_embed.shape[0], 1, 1),
            self.row_embed.unsqueeze(1).repeat(1, self.col_embed.shape[0], 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # returns [ B,F,H,W]


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool=True):

        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
    def forward(self, im):
        xs = self.body(im)
             
        return list(xs.values())

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
                                                    replace_stride_with_dilation=[False, False, dilation],
                                                    pretrained=True, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class PositionalTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,device="cuda"):
        super().__init__()
        self.bbox_embed = MLP(d_model, d_model, 4, 6)

        args=(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.bbox_attention = MHAttentionMap(d_model, d_model, nhead, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(d_model + nhead, [1024, 512, 256], d_model)

        self.encoder = PosTransformerEncoder(args, num_encoder_layers, normalize_before)

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(args,num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model)
        self.decoder2=TransformerDecoder(args,num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec,
                                            d_model=d_model)
        # self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers

    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed, mask=None):
        # flatten NxCxHxW to HWxNxC
        src2 = src.flatten(2).permute(2, 0, 1)
        pos_embed2 = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, src2.shape[1], 1)
        #print("query embed",query_embed.shape) #240 , B,F
        mask = mask.flatten(1)
    
        tgt = torch.zeros_like(query_embed) 
        #print("tgt",tgt.shape) #240 , B,F

        memory = self.encoder(src2, src_key_padding_mask=mask, pos=pos_embed2)
      
        hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed2, query_pos=query_embed)
        hs2, references2 = self.decoder2(tgt, memory, memory_key_padding_mask=mask,
                            pos=pos_embed2, query_pos=query_embed)
        tmp=self.bbox_embed(hs)
        #print(references.shape)#B,nq,512
        tmp[..., :2] +=  inverse_sigmoid(references) 
        outputs_coord = tmp.sigmoid()
        outputs_class = hs
        tmp=self.bbox_embed(hs2)
        tmp[..., :2] +=  inverse_sigmoid(references2) 
        outputs_coord2 = tmp.sigmoid()
        outputs_class2 = hs2
        # to do this, we  have a set of class features,  and a list of object coords. 
        #bbox_mask = self.bbox_attention(hs[-1], references, mask=mask)
        #WE could pair the 2... 
        # the bbox mask ideally should be a image shape with features in the object locations, I want an output C,NQ,Confidence,H,W?
        
        #make a B,h,w, c tensor, then use the bbox mask to select the features, then use the mask head to get the masks
        #seg_masks = self.mask_head(src, bbox_mask, src) #x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):

        ##outputs_seg_masks = seg_masks.view(src.shape[0], query_embed.shape[0], seg_masks.shape[-2], seg_masks.shape[-1])
        #print("bbox mask",bbox_mask.shape)
        #print("src",src.shape)
        return outputs_coord, outputs_class ,outputs_coord2, outputs_class2#,outputs_seg_masks


class PosTransformerEncoder(nn.Module):

    def __init__(self,args, num_layers, norm=None):
        super().__init__()
        self.layers =nn.Sequential(*[TransformerEncoderLayer(*args) for i in range(num_layers)])
    
        self.num_layers = num_layers
        #if norm is not None:
        self.norm =  nn.LayerNorm(args[0])
        # else:
        #     #set norm to do nothing
        #     self.norm = lambda x: x

    def forward(self, output:Tensor,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        output= self.norm(self.layers(dict(src=output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos))["src"])
       
        return output



class TransformerDecoder(nn.Module):

    def __init__(self,args, num_layers, norm=None, return_intermediate=False, d_model=256):
        super().__init__()
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.layers = nn.Sequential(*[TransformerDecoderLayer(*args,first_layer=True)]+[TransformerDecoderLayer(*args, query_scale=self.query_scale) for i in range(num_layers)])   # Can I make this squential?
        self.num_layers = num_layers
        self.norm = norm
        self.dim2_t = torch.pow(10000** (2 / (d_model/2)), torch.arange(d_model//4, dtype=torch.float32))
        self.intermediate = []
        self.return_intermediate = return_intermediate
        self.ref_point_head = MLP(d_model, d_model, 2, 2)
        self._register_hook()
        self.obj_centre=self.obj_centre_to_query

        #if is aux, then I want to generate things slightly differently


    def _register_hook(self):
        
        if self.return_intermediate:

            for layer in self.layers:
                layer.register_forward_hook(self._get_intermediate_output)
        else:
            self.layers[-1].register_forward_hook(self._get_intermediate_output)
    def _get_intermediate_output(self, layer, input, output):
        self.intermediate.append(self.norm(output["tgt"]))
    
    def gen_sineembed_for_position(self,pos_tensor):
         
        pos_tensor = torch.mul(pos_tensor,2 * math.pi)
        pos2_x = pos_tensor[:, :, 0].unsqueeze(-1) / self.dim2_t.to(pos_tensor.device)
        pos2_y = pos_tensor[:, :, 1].unsqueeze(-1) / self.dim2_t.to(pos_tensor.device)
        #shapes are 300,B,128          
        pos2_x = torch.stack((pos2_x.sin(), pos2_x.cos()), dim=3).flatten(2)
        pos2_y = torch.stack((pos2_y.sin(), pos2_y.cos()), dim=3).flatten(2)
        return torch.cat((pos2_y, pos2_x), dim=2)
    
    # def gen_learnt_embed_for_position(self,pos_tensor):
    #     class PositionEmbeddingLearned(nn.Module):
    
    #     def __init__(self, num_pos_feats=512,device='cuda'):
    #         super().__init__()
    #         self.device=device
    #         self.row_embed = nn.Parameter(torch.ones(25, num_pos_feats,device=self.device))
    #         self.col_embed = nn.Parameter(torch.ones(25, num_pos_feats,device=self.device))

    #     return torch.cat([
    #         self.col_embed.unsqueeze(0).repeat(self.row_embed.shape[0], 1, 1),
    #         self.row_embed.unsqueeze(1).repeat(1, self.col_embed.shape[0], 1),
    #     ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # returns [ B,F,H,W]



    def obj_centre_to_query(self,query_pos):
        reference_points = self.ref_point_head(query_pos).sigmoid()
        objcentres = reference_points[..., :2]#.transpose(0, 1)
        return objcentres
    
    def forward(self, output, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        self.intermediate = []
        reference_points_before_sigmoid = self.ref_point_head(query_pos)    # [num_queries, batch_size, 2]
        reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)
        query_sine_embed = self.gen_sineembed_for_position(reference_points[..., :2].transpose(0, 1))
        output=self.layers(dict( tgt=output, memory=memory,
                                tgt_mask = tgt_mask,
                                memory_mask = memory_mask,
                                tgt_key_padding_mask = tgt_key_padding_mask,
                                memory_key_padding_mask = memory_key_padding_mask,
                                pos = pos,
                                query_pos = query_pos,
                                query_sine_embed = query_sine_embed,
                                is_first = False))["tgt"]
        #print("out shape:",output.shape)
        return (torch.stack(self.intermediate).transpose(1, 2), reference_points)
        
        


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


        self.activation = _get_activation_fn(activation)
        self.forward=None
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.forward = self.forward_pre
        else:
            self.forward = self.forward_post

    def forward_post(self,kwargs):
        src = kwargs['src']
        src_mask = kwargs['src_mask']
        src_key_padding_mask = kwargs['src_key_padding_mask']
        pos = kwargs['pos']
        q = k = src+ pos
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        kwargs['src']=src
        return kwargs
              





    def forward_pre(self, kwargs):
        src = kwargs['src']
        src_mask = kwargs['src_mask']
        src_key_padding_mask = kwargs['src_key_padding_mask']
        pos = kwargs['pos']

        src2 = self.norm1(src)
        q = k = src2+ pos.to(src2.device)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        kwargs['src']=src
        return kwargs


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,first_layer=False,query_scale=None):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
        self.first_layer = first_layer
        if query_scale is not None:
            self.query_scale = query_scale
        if first_layer:
            self.ca_qpos_proj = nn.Linear(d_model, d_model)
            
        
        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.run=None
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.forward = self.forward_pre
        else:
            self.forward = self.forward_post
    def query_scale(self,query):
        return 1.0
    def forward_post(self,  kwargs):
        
        tgt=kwargs['tgt']
        memory=kwargs['memory']
        tgt_mask=kwargs['tgt_mask']
        memory_mask=kwargs['memory_mask']
        tgt_key_padding_mask=kwargs['tgt_key_padding_mask']
        memory_key_padding_mask=kwargs['memory_key_padding_mask']
        pos=kwargs['pos']
        query_pos=kwargs['query_pos']
        query_sine_embed=kwargs['query_sine_embed']             
        
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw= k_content.shape[0]

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if self.first_layer:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        #print(query_sine_embed.shape)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed* self.query_scale(tgt))
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        kwargs['tgt']=tgt
        return kwargs

    def forward_pre(self, kwargs):
        
        tgt=kwargs['tgt']
        memory=kwargs['memory']
        tgt_mask=kwargs['tgt_mask']
        memory_mask=kwargs['memory_mask']
        tgt_key_padding_mask=kwargs['tgt_key_padding_mask']
        memory_key_padding_mask=kwargs['memory_key_padding_mask']
        pos=kwargs['pos']
        query_pos=kwargs['query_pos']
        
        tgt2 = self.norm1(tgt)
        q = k = tgt2 +query_pos
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=tgt2 + query_pos,
                                   key=memory+ pos,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        kwargs['tgt']=tgt
        return kwargs



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")