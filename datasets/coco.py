from pathlib import Path
import torch     
import os
import zipfile
from pySmartDL import SmartDL
import pytorch_lightning as pl
import time
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from typing import Optional, List
from torchvision.transforms import Resize
import torch
from torch import Tensor
import datasets.transforms as T
from PIL import Image
import torchvision.transforms.v2 as T2
from transformers import CLIPTokenizer
import clip
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.Tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.myclip,_=clip.load('ViT-B/32',device="cpu")
        self.myclip.eval()
        self._transforms = transforms
        self.tokenized_classnames={int(i):self.tokenize(" ".join(["a",c["name"]])) for i,c in self.coco.cats.items()}
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def tokenize(self, x):
        return self.myclip.encode_text(self.Tokenizer(x, # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 77,           # Pad & truncate all sentences.
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )['input_ids'].squeeze(1)).detach().cpu()
    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        #print("target",target)
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        #pr
        return img, target, self.tokenized_classnames


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def collate_fn(batch):
    batch = list(zip(*batch))

    batch[0] = nested_tensor_from_tensor_list(batch[0])
    #collate images
    
    batch[2] = batch[2][0]
    return tuple(batch)

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
        if self.tensors.isnan().any():
            raise ValueError("NaN in tensor")
    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target




class COCODataModule(pl.LightningDataModule):

    def __init__(self, Cache_dir='.', batch_size=256):
        super().__init__()
        self.data_dir = Cache_dir
        self.ann_dir=os.path.join(self.data_dir,"annotations")
        self.batch_size = batch_size
        self.splits={"train":[],"val":[],"test":[]}
        self.prepare_data()
    def train_dataloader(self, B=None):
        if B is None:
            B=self.batch_size 
        return torch.utils.data.DataLoader(self.train, batch_size=B, shuffle=True, num_workers=4, collate_fn=collate_fn, prefetch_factor=4, pin_memory=True,drop_last=True)
    def val_dataloader(self, B=None):
        if B is None:
            B=self.batch_size
       
        return torch.utils.data.DataLoader(self.val, batch_size=B, shuffle=True, collate_fn=collate_fn,num_workers=1, prefetch_factor=1, pin_memory=True,drop_last=True)
    def test_dataloader(self,B=None):
        if B is None:
            B=self.batch_size


        return torch.utils.data.DataLoader(self.test, batch_size=B, shuffle=True, num_workers=4, collate_fn=collate_fn, prefetch_factor=4, pin_memory=True,drop_last=True)
    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # # download data
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir,exist_ok=True)
        if not os.path.exists(self.ann_dir):
            os.makedirs(self.ann_dir,exist_ok=True)
        urls=[
                #'https://images.cocodataset.org/zips/test2017.zip',
                'https://images.cocodataset.org/zips/train2017.zip',
                'https://images.cocodataset.org/zips/val2017.zip',
                'https://images.cocodataset.org/annotations/annotations_trainval2017.zip'
                ]

        objs=[]
        for url in urls:
            #print("url:",url)
            name=str(url).split('/')[-1]
            location=self.data_dir # if name.startswith("annotations") else self.ann_dir
            #print("Location", location) #/Data/train2014.zip
            #time.sleep(5)
            #print('Downloading',url)
            obj=SmartDL(url,os.path.join(location,name),progress_bar=False, verify=False)
            obj.FileName=name
            if name.endswith(".zip"):
                name=name[:-4]
            if name.startswith("train"):
                self.splits['train'].append(name)
            elif name.startswith("val"):
                self.splits['val'].append(name)
            elif name.startswith("test"):
                self.splits['test'].append(name)
            if not os.path.exists(os.path.join(location,name)) and not (name.startswith("annotations") and os.path.exists(os.path.join(location,"annotations"))):
                print(os.path.join(location,name))
                objs.append(obj)
                obj.start(blocking=False,  )#There are security problems with Hostename 'images.cocodataset.org' and Certificate 'images.cocodataset.org' so we need to disable the SSL verification
        for obj in objs:
            while not obj.isFinished():
                time.sleep(5)
            if obj.isSuccessful():
                print("Downloaded: %s" % obj.get_dest())
            path = obj.get_dest()
            if path.endswith(".zip"):
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    try:
                        zip_ref.extractall(self.data_dir)
                    except Exception as e:
                        print(e)
                        print("Error extracting zip" ,path)
                        continue        
                    for root, dirs, files in os.walk(zip_ref.namelist()[0]):
                        for file in files:
                            Path(os.path.join(root, file)).touch()
    def make_coco_transforms(self,image_set):


        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return T.Compose([
                T2.Resize(800),
                T2.CenterCrop(800),
                #Note: the standard  lambda function here is not supported by pytorch lightning
            
                normalize,
        ])

        

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        #print("Entered COCO datasetup")
        
    
        
        root = Path(self.data_dir)
        assert root.exists(), f'provided COCO path {root} does not exist'
        mode = 'instances'
        PATHS = {
            "train": (root  / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
            "test": (root /"test2017", root / "annotations" / f'image_info_test-dev2017.json'),
        }

        img_folder, ann_file = PATHS["train"]
        self.train = CocoDetection(img_folder, ann_file, transforms=self.make_coco_transforms("train"), return_masks=False)
        img_folder, ann_file = PATHS["val"]
        self.val = CocoDetection(img_folder, ann_file, transforms=self.make_coco_transforms("val"), return_masks=False)
        # img_folder, ann_file = PATHS["test"]
        # self.test=CocoDetection(img_folder, ann_file, transforms=self.make_coco_transforms("test"), return_masks=False)




