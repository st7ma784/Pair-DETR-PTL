# Pair DETR

This repository is an forked implementation of Pair DETR 

We're going to move it to pytorch-lightning and then look at implementing some DETIC goodness!
## Introduction



The DETR approach applies the
transformer encoder and decoder architecture to object detection
and achieves promising performance. In this paper,
we handle the critical issue, slow training convergence,
and present a conditional cross-attention mechanism for
fast DETR training. Our approach is motivated by that <b>the
cross-attention in DETR relies highly on the content embeddings
and that the spatial embeddings make minor contributions</b>,
increasing the need for high-quality content embeddings
and thus increasing the training difficulty.

<div align=center>  
<img src='.github/attention-maps.png' width="100%">
</div>

Our conditional DETR learns a conditional
spatial query from the decoder embedding
for decoder multi-head cross-attention.
The benefit is that through the conditional spatial query,
each cross-attention head is able to 
<b>attend
to a band containing a distinct region,
e.g., one object extremity or a region inside the object box </b> (Figure 1).
This narrows down the spatial range for localizing the distinct regions
for object classification and box regression,
thus relaxing the dependence on the content embeddings and
easing the training. Empirical results show that conditional
DETR converges 6.7x faster for the backbones R50 and
R101 and 10x faster for stronger backbones DC5-R50 and
DC5-R101.

<div align=center>  
<img src='.github/conditional-detr.png' width="48%">
  <img src='.github/convergence-curve.png' width="48%">
</div>

Goals: 

- [x] - get this working 
- [x] - Get this working on custom data 
- [x] - implement DETIC from CLIP EMbs 
- [ ] - Can I modify the positional embedding to carry CLIP info? 



Sidequest: 
- [x] - Does CLIP contain enough info to straight up predict positional embeddings? 
- [x] - Understand why Linear_sum_assignment is the best way forward
- [x] - Can CLIP predict Masks for more complex items than just DETIC?
