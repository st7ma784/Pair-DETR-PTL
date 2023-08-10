#So I think you just need to update that line to:
import torchvision
#torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms
#from torchvision.transforms import v2 as transforms

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
import os,sys
import cv2
# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder
import torch
from transformers import CLIPTokenizer
import os
import pytorch_lightning as pl
from torch.utils.data import Dataset,IterableDataset
from datasets import load_dataset
from PIL import Image
import clip
import wget
import random
import numpy
import requests
import time
from io import BytesIO
from torchvision import datapoints
prep=transforms.Compose([
        transforms.Resize(224, interpolation=Image.NEAREST),
        transforms.CenterCrop(224),
        #Note: the standard  lambda function here is not supported by pytorch lightning
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

class VisGenomeIterDataset(IterableDataset):
    def __init__(self,split="train",dir="HF",T=prep,tokenizer=None,predictor=None):
        #print('Loading COCO dataset')
        self.data=load_dataset("visual_genome", "relationships_v1.2.0",streaming=True,cache_dir=dir)[split]
        print("got datast")
        self.predictor=predictor
        self.T=T
        self.tokenizer=tokenizer
        self.clip,_=clip.load("ViT-B/32",device="cpu")

        if tokenizer is None:
            self.tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=dir) 
        self.tokenize=lambda x:self.tokenizer(x,return_tensors="pt",padding="max_length", truncation=True,max_length=77)['input_ids']
        self.data =self.data.map(self.process,remove_columns=['image_id' , 'image','relationships','width', 'height', 'coco_id', 'flickr_id','url' ])

    def process(self,item):
        caption=""
        if len(item["relationships"])==0:
            caption=" an image with no objects interacting"
        else:
            r=random.choice(item["relationships"])
            caption=" ".join(["a",r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]])
        r=self.tokenize(caption)
        try:
            return {"img":prep(item["image"]),"relation":r}
        except FileNotFoundError as e:
            response = requests.get(item["url"])
            img = Image.open(BytesIO(response.content))
            print("failed : {}".format(item["image"]))
            return {"img":prep(img),"relation":r}
    def __len__(self):
        return 108077
    def __iter__(self):
        # item= self.data.next()
        # r=random.choice(item["relationships"])
        # r=self.tokenize(" ".join(["a",r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]]), return_tensors="pt",padding="max_length", truncation=True,max_length=77)['input_ids'])
        # out= {"img":prep(item["image"]),"relation":r}
        #print(self.data.__dir__())
        return self.data.__iter__()

    

class VisGenomeDataset(Dataset):
    def __init__(self, split="train",dir="HF",T=prep,tokenizer=None,predictor=None):
        #print('Loading COCO dataset')
        self.data=load_dataset("visual_genome", "relationships_v1.2.0",streaming=False,cache_dir=dir)[split]
        self.T=T
        self.predictor=predictor
        self.clip,_=clip.load("ViT-B/32",device="cpu")
        self.tokenizer=tokenizer
        if tokenizer is None:
            self.tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=dir) 
        self.tokenize=lambda x:self.tokenizer(x,return_tensors="pt",padding="max_length", truncation=True,max_length=77)['input_ids']
        # self.data =self.data.map(self.process,remove_columns=['image_id' , 'image','relationships','width', 'height', 'coco_id', 'flickr_id','url' ],num_proc=8).filter(lambda example: example is not None)
        
    def process(self,item):
        caption=" "
        if len(item["relationships"])==0:
            caption=" an image with no objects interacting"
        else:
            r=random.choice(item["relationships"])
            caption=" ".join(["a",r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]])
        r=self.tokenize(caption)
        #print(item)
        try:
            return {"img":prep(item["image"]),"relation":r}
        except FileNotFoundError as e:
            response = requests.get(item["url"])
            img = Image.open(BytesIO(response.content))
            #print("failed : {}".format(item["image"]))
            return {"img":prep(img),"relation":r}
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.process(self.data.__getitem__(idx))

class VisGenomeDatasetBigBoxes(VisGenomeDataset):
    def process(self,item):
        if len(item["relationships"])==0:
            return None
        #this is probably awful,  Why not look at the Pairwise DETR for how it does contrastive loss on multiple objects? 
        r=random.choice(item["relationships"])
        #s is the r["subject"] box
        s=datapoints.BoundingBox([r["subject"]["x"],r["subject"]["y"],r["subject"]["w"],r["subject"]["h"]], format=datapoints.BoundingBoxFormat.XYWH, spatial_size=[item["width"],item["height"]])
        o=datapoints.BoundingBox([r["object"]["x"],r["object"]["y"],r["object"]["w"],r["object"]["h"]], format=datapoints.BoundingBoxFormat.XYWH, spatial_size=[item["width"],item["height"]])

        
        l=datapoints.BoundingBox([min(r["object"]["x"],r["subject"]["x"]),
                                 min(r["object"]["y"],r["subject"]["y"]), # these find the top left corner
                                 max(r["object"]["x"],r["subject"]["x"])-min(r["object"]["x"],r["subject"]["x"]) +max(r["object"]["w"],r["subject"]["w"]), # find the bottom right corner with max of x ys and add the whs.  
                                max(r["object"]["y"],r["subject"]["y"])-min(r["object"]["y"],r["subject"]["y"])+ max(r["object"]["h"],r["subject"]["h"])], format=datapoints.BoundingBoxFormat.XYWH, spatial_size=[item["width"],item["height"]])
        r=self.tokenize(" ".join(["a",r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]]))

        img=item["image"]
        try:
            i,t=prep(img,{"boxes":[s,o,l]})
            return {"img":i,"relation":r,"targets":t}
        except FileNotFoundError as e:
            response = requests.get(item["url"])
            img = Image.open(BytesIO(response.content))
            #print("failed : {}".format(item["image"]))
            i,t=prep(img,{"boxes":[s,o,l]})
            return {"img":i,"targets":t,"relation":r}
            
class VisGenomeDatasetIterBigBoxes(VisGenomeIterDataset):
    def process(self,item):

        if len(item["relationships"])==0:
            return None
        r=random.choice(item["relationships"])
        #s is the r["subject"] box
        s=datapoints.BoundingBox([r["subject"]["x"],r["subject"]["y"],r["subject"]["w"],r["subject"]["h"]], format=datapoints.BoundingBoxFormat.XYWH, spatial_size=[item["width"],item["height"]])
        o=datapoints.BoundingBox([r["object"]["x"],r["object"]["y"],r["object"]["w"],r["object"]["h"]], format=datapoints.BoundingBoxFormat.XYWH, spatial_size=[item["width"],item["height"]])

        
        l=datapoints.BoundingBox([min(r["object"]["x"],r["subject"]["x"]),
                                 min(r["object"]["y"],r["subject"]["y"]), # these find the top left corner
                                 max(r["object"]["x"],r["subject"]["x"])-min(r["object"]["x"],r["subject"]["x"]) +max(r["object"]["w"],r["subject"]["w"]), # find the bottom right corner with max of x ys and add the whs.  
                                max(r["object"]["y"],r["subject"]["y"])-min(r["object"]["y"],r["subject"]["y"])+ max(r["object"]["h"],r["subject"]["h"])], format=datapoints.BoundingBoxFormat.XYWH, spatial_size=[item["width"],item["height"]])
        r=self.tokenize(" ".join(["a",r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]]))
        img=item["image"]

        try:
            i,t=prep(img,{"boxes":[s,o,l]})
            return {"img":i,"relation":r,"targets":t}
        except FileNotFoundError as e:
            response = requests.get(item["url"])
            img = Image.open(BytesIO(response.content))
            #print("failed : {}".format(item["image"]))
            i,t=prep(img,{"boxes":[s,o,l]})
            return {"img":i,"targets":t,"relation":r}
# Dataset
def DETICprocess(self,item):
    if len(item["relationships"])==0:
        return None
    #this process function takes an item and returns a COCO style dict with boxes and labels.


    #our bboxes come straight from the BigBoxes and masks in the VisGenome Dataset. To grab labels, we need to get the object and subject names and pass them through self.predictor to get the masks.
    # 
    out=[]
    img=item["image"]
    #convert JpegImageFile to cv2 image
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)

    try:
        i=prep(img)
    except FileNotFoundError as e:
        response = requests.get(item["url"])
        #open response as cv2 image
        imagebytes=BytesIO(response.content)
        #convert to cv2 image
        img = cv2.imdecode(numpy.frombuffer(imagebytes.read(), numpy.uint8), cv2.IMREAD_COLOR)
    #print("item:", item.keys())
    for r in item["relationships"]:
        #print(r)
        #s is the r["subject"] box
        outputs=self.predictor(img,[r["subject"]["names"][0], r["object"]["names"][0]])
        
        # print(outputs['instances'].keys())
        #print(outputs['instances'].get_fields().keys())#VVdict_keys(['pred_boxes', 'scores', 'pred_classes', 'pred_masks'])
        #print(outputs['instances'].get('pred_boxes').__dir__())
        found_masks=outputs['instances'].get('pred_masks')
        found_boxes=outputs['instances'].get('pred_boxes') #these are in xyxy format
        #check outputs for bounding boxes that are close to the subject and object boxes.
        obj_bboxes=torch.stack(
                    [torch.tensor([r["subject"]["x"],r["subject"]["y"],r["subject"]["x"]+r["subject"]["w"],r["subject"]["y"]+r["subject"]["h"]]),            
                    torch.tensor([r["object"]["x"],r["object"]["y"],r["object"]["x"]+r["object"]["w"],r["object"]["y"]+r["object"]["h"]])
                    ],dim=0)
        
        if len(found_boxes)==0:
            print("No boxes found")
            if len(found_masks)==0:
                print("No masks found")
                continue
            else:
                #convert masks to bboxes
                found_masks=found_masks.to("cpu").numpy()
                found_boxes=[torchvision.ops.masks_to_boxes(m) for m in found_masks]
                found_boxes=torch.stack([b.as_xyxy() for b in found_boxes])
        else:
            #get tensor from Boxes object
            found_boxes=found_boxes.tensor

        #convert to tensors
        print("obj_bboxes",obj_bboxes.shape)
        print("found_boxes",found_boxes)
        #found_boxes=torch.tensor(found_boxes) #######################################
        annotation_to_output_ious=torchvision.ops.box_iou(obj_bboxes,found_boxes)
        #find max iou +_idx for each annotation 
        max_ious,max_idx=torch.max(annotation_to_output_ious,dim=0)
        bboxes_to_keep=found_boxes[max_idx]
        masks_to_keep=found_masks[max_idx]
        print("bboxes_to_keep",bboxes_to_keep.shape)#torch.Size([2, 4])
        print("masks_to_keep",masks_to_keep.shape)#torch.Size([2, 800, 800])
        object_mask=torch.logical_or(masks_to_keep[0],masks_to_keep[1]).unsqueeze(0)
        print("object_mask",object_mask.shape)#torch.Size([800, 800]
        object_actual_bbox_from_mask=torchvision.ops.masks_to_boxes(object_mask)
        #if so, do a logical_and on the masks of subj and obj and get the bounding box of the result.

        original_bbox=torch.tensor([min(r["object"]["x"],r["subject"]["x"]),
                                min(r["object"]["y"],r["subject"]["y"]), # these find the top left corner
                                max(r["object"]["x"],r["subject"]["x"])-min(r["object"]["x"],r["subject"]["x"]) +max(r["object"]["w"],r["subject"]["w"]), # find the bottom right corner with max of x ys and add the whs.  
                            max(r["object"]["y"],r["subject"]["y"])-min(r["object"]["y"],r["subject"]["y"])+ max(r["object"]["h"],r["subject"]["h"])]).unsqueeze(0)
        print("original_bbox",original_bbox)
        print("object_actual_bbox_from_mask",object_actual_bbox_from_mask)
        print("Comparison of boxes: ", torchvision.ops.box_iou(original_bbox,object_actual_bbox_from_mask))


        out.append({"boxes":object_actual_bbox_from_mask,
                    "labels":self.tokenize(" ".join(["a",r["subject"]["names"][0],r["predicate"],r["object"]["names"][0]]))
,
                    "masks":object_mask})
    img=item["image"]
    target={'image_id':item.get("image_id",0),
            "iscrowd":torch.zeros(len(out)),
            'boxes':torch.tensor([o["boxes"].as_xyxy() for o in out]),
            'area':torch.tensor([o["boxes"].area() for o in out]),
            'masks':torch.stack([o["masks"] for o in out]),
            'labels':self.clip.encode_text(torch.stack([o["labels"] for o in out])),
    }

    print("target:", target.keys())
    try:
        i,t=prep(img,target)
    except FileNotFoundError as e:
        response = requests.get(item["url"])
        img = Image.open(BytesIO(response.content))
        #print("failed : {}".format(item["image"]))
        i,t=prep(img,target)
    summed_mask=torch.sum(target["masks"],dim=0).bool().int()
    classes=target["labels"]
    return i, t, classes ,summed_mask

class VisGenomeDatasetCOCOBoxes(VisGenomeDataset):
    def process(self,item):
        return DETICprocess(self,item)
class VisGenomeDatasetIterCOCOBoxes(VisGenomeDataset):
    def process(self, item):
        return DETICprocess(self,item)
def DETIC_collate_fn(batch):
    batch = list(zip(*batch))
    #this is the test function for collating these,
    # we still need to combine the class embeddings and return a label for each one. 
    batch[0] = torch.stack(batch[0], dim=0) # stack images together
    #batch[1] is all the targets

    batch[2] = torch.unique(torch.stack(batch[2]))
    batch[3]= torch.stack(batch[3], dim=0) # the mask.

    #ids,boxes,masks,sizes=zip(*[(v["labels"],v["boxes"],v["masks"],v["boxes"].shape[0]) for v in batch[1]])
    #ids is the index of ids in batch[2]
    #ids=torch.as_tensor([torch.where(batch[2]==i)[0][0] for i in ids])
    #batch[1]=(torch.cat(ids),torch.cat(boxes),torch.cat(masks),torch.as_tensor(sizes))
    for target in batch[1]:
        target["labels"]=torch.stack([torch.unique(torch.where(batch[2]==i,torch.arange(batch[2].shape[0]),-1)) for i in target["labels"]])
        target["labels"]=target["labels"][target["labels"]!=-1]
    #stack all the classes and return the replace labels with the indices of the classes

    #create a batch indices tensor
    # batch.append(torch.cat([torch.full((t,),i) for i,t in enumerate(sizes)],dim=0))
    #print("idxs",batch[-1])
    return tuple(batch)


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0], dim=0) # stack images together
    #batch[1] is all the targets
    ids,boxes,masks,sizes=zip(*[(v["labels"],v["boxes"],v["masks"],v["boxes"].shape[0]) for v in batch[1]])
    batch[1]=(torch.cat(ids),torch.cat(boxes),torch.cat(masks),torch.as_tensor(sizes))
    batch[2] = batch[2][0] # this is all the classnames, we'll just take the first set
    batch[3]= torch.stack(batch[3], dim=0) # the mask.
    #create a batch indices tensor
    batch.append(torch.cat([torch.full((t,),i) for i,t in enumerate(sizes)],dim=0)) # batch[4] is idxs for which box goes back to which image
    #print("idxs",batch[-1])
    return tuple(batch)

import wandb

class VisGenomeDataModule(pl.LightningDataModule):

    def __init__(self, Cache_dir='.', T=prep, batch_size=256,stream=False, fullBoxes=False):
        super().__init__()
        self.data_dir = Cache_dir
        self.batch_size = batch_size
        self.stream=stream
        self.tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32") 

        if fullBoxes:
            self.dataConstructor=VisGenomeDatasetCOCOBoxes
            if self.stream:
                self.dataConstructor=VisGenomeDatasetIterCOCOBoxes
        else:
            self.dataConstructor=VisGenomeDataset
            if self.stream:
                self.dataConstructor=VisGenomeIterDataset
        self.T=T
        self.cfg = get_cfg()
        add_centernet_config(self.cfg)
        add_detic_config(self.cfg)
        self.cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        filename="./models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        if not os.path.exists("./models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"):
            if os.path.exists("./Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"):
                #copy it over
                os.system("cp ./Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth ./models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
            else:
                url = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
                filename = wget.download(url)
                print("fetched to {}".format(filename))
        self.cfg.MODEL.WEIGHTS = filename
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        self.cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        self.cfg.MODEL.DEVICE='cpu' 
        self.text_encoder = build_text_encoder(pretrain=True)
        self.text_encoder.eval()
        self.predictor = DefaultPredictor(self.cfg)
        #build in a wandb for logging images
        if hasattr(self,"trainer") and self.trainer is not None:

            self.wandb=self.trainer.logger
        else:
            self.wandb=wandb.init(project="clip-detector",entity="st7ma784",name=str(time.time()))

    def predict(self,image,classes): 
        print("predicting...") 
        classifier = self.get_clip_embeddings(classes)
        self.predictor.model.roi_heads.num_classes =  len(classes)
        metadata = MetadataCatalog.get(str(time.time()))
        metadata.thing_classes = classes
        #print("cshape",classifier.shape) #F,2
        # zs_weight = torch.cat([classifier, classifier.new_zeros((classifier.shape[0], 1))], dim=1) # D x (C + 1)
        # if self.predictor.model.roi_heads.box_predictor[0].cls_score.norm_weight:
        #     zs_weight = torch.nn.functional.normalize(zs_weight, p=2, dim=0)
        # zs_weight = zs_weight.to(self.predictor.model.device)
        # for k in range(len(self.predictor.model.roi_heads.box_predictor)):
        #     del self.predictor.model.roi_heads.box_predictor[k].cls_score.zs_weight
        #     self.predictor.model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight
        # # Reset visualization threshold
        reset_cls_test(self.predictor.model, classifier, len(classes))
        output_score_threshold = 0.5
        for cascade_stages in range(len(self.predictor.model.roi_heads.box_predictor)):
            self.predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold
        #print("image shape",image.shape)
        #convert to np array
        # image=image.permute(1,2,0).numpy()
        outputs = self.predictor(image)

        v = Visualizer(image[:, :, ::-1], metadata)
        #print(outputs.keys())
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #print(outputs['instances'].get_fields()["pred_masks"].shape)
        out_path = "out{}.png".format(time.time())
        cv2.imwrite(str(out_path), out.get_image()[:, :, ::-1])

        self.wandb.log({"image":wandb.Image(out_path)})


        #So - Idea - What if I could use the score to add noise to the output class. 
        return outputs


    def get_clip_embeddings(self,vocabulary, prompt='a origami  '):
        
        texts = [prompt + x for x in vocabulary]
        if "{}" in prompt:
            texts=[prompt.format(x) for x in vocabulary]
        return self.text_encoder(texts).detach().permute(1, 0).contiguous()
    
    def train_dataloader(self, B=None):
        if B is None:
            B=self.batch_size 
        return torch.utils.data.DataLoader(self.train, batch_size=B, shuffle=not self.stream, num_workers=4, prefetch_factor=3, pin_memory=True,drop_last=True)
    def val_dataloader(self, B=None):
        if B is None:
            B=self.batch_size
       
        return torch.utils.data.DataLoader(self.val, batch_size=B, shuffle=not self.stream, num_workers=1, prefetch_factor=1, pin_memory=True,drop_last=True)
    def test_dataloader(self,B=None):
        if B is None:
            B=self.batch_size


        return torch.utils.data.DataLoader(self.test, batch_size=B, shuffle=not self.stream, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        #print("Entered COCO datasetup")
        
        #if stage == 'fit' or stage is None:
        self.train=self.dataConstructor(T=self.T,dir=self.data_dir,split="train",tokenizer=self.tokenizer,predictor=self.predict)
    



if __name__ == "__main__":
    #run this to test the dataloader
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--Cache_dir', type=str, default='.', help='path to download and cache data')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size')
    parser.add_argument('--stream', default=False, type=bool,help='stream data',)
    parser.add_argument("--COCO", default=False, type=bool,help="Use COCO style data")
    args=parser.parse_args()
    dir=os.path.join(args.Cache_dir,"data")
    dm =VisGenomeDataModule(Cache_dir=dir,batch_size=args.batch_size,stream=args.stream,fullBoxes=args.COCO)
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        print(batch)
        break