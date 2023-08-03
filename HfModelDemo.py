
from transformers import AutoModelForObjectDetection, DetrImageProcessor
from pytorch_lightning import LightningModule
import torch
import cv2
from typing import Optional
import os
import evaluate
import torchvision
class HFModule(LightningModule):
    '''
    This training code follows the standard structure of Pytorch - lighthning. It's worth looking at their docs for a more in depth dive as to why it is this was
    '''
    
    def __init__(self,
                learning_rate=0.0001,
                modelname="facebook/detr-resnet-50",
                total_steps: int = 200000,
                train_batch_size: int = 64,
                eval_batch_size: int = 32,
                eval_splits: Optional[list] = None,
                **kwargs,
                ):

        super().__init__()
        self.save_hyperparameters()
        id2label = dict(enumerate(["waste"]))
        label2id = {v: k for k, v in id2label.items()}
        

        self.model =   AutoModelForObjectDetection.from_pretrained(modelname, 
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,)
        
        self.Prep= DetrImageProcessor.from_pretrained(modelname,
        size={"shortest_edge": self.hparams.res, "longest_edge": self.hparams.res})
        

        #val_dataloader = torch.utils.data.DataLoader(
        #test_ds_coco_format, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn
        #)



  
    def forward(self,**input):
        #check pixel values are 

        #move labels to self.device
        if input["labels"] is not None:
            for label in input["labels"]:
                #trying for better trianing
                label["boxes"]=torchvision.ops.box_convert(label["boxes"], in_fmt="xyxy", out_fmt="xywh")
                label["boxes"]=torchvision.ops.box_convert(label["boxes"], in_fmt="cxcywh", out_fmt="xyxy").to(self.device)
                #print(label["boxes"])
                #label["boxes"]=label["boxes"].to(self.device)
                label["class_labels"]=label["class_labels"].to(self.device)
        
        return self.model(**input)
     

    def training_step(self, batch, batch_idx,optimizer_idx=0):
        self.log("Inputshape",torch.tensor(batch["pixel_values"].shape[-1]), on_epoch=True, prog_bar=True)

        outputs=self.forward(**batch)
        if outputs is None:
            self.log('train_loss', 10, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return None
        self.log('train_loss', outputs.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return outputs.loss

    def test_epoch_start(self,*args):
        
        #model = AutoModelForObjectDetection.from_pretrained("MariaK/detr-resnet-50_finetuned_cppe5")
        self.module = evaluate.load("ybelkada/cocoevaluate", coco=self.coco_ann)
    def test_step(self,batch,batch_idx):

        #calculat mAP for test set
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]

        labels = [
            {k: v for k, v in t.items()} for t in batch["labels"]
        ]  # these are in DETR format, resized + normalized

        # forward pass
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = self.Prep.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api

        self.module.add(prediction=results, reference=labels)
    def test_epoch_end(self,outputs):
        results = self.module.compute()
        self.log("mAP",results["mAP"], on_epoch=True, prog_bar=True, logger=True)
        self.print(results)
         
    def validation_step(self, batch, batch_idx):
       
        #Draw the image and the bounding boxes.
        #print("Validation")
        #Boxes=batch["labels"]
        #print("Boxes",Boxes)
        outputs=self.forward(**batch)
        self.log('val_loss', outputs.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if outputs is None:
            return None
        target_sizes = torch.stack([torch.tensor(img.shape[-2:],device=self.device) for img in batch["pixel_values"]])
        results = self.Prep.post_process_object_detection(outputs, threshold=0.8, target_sizes=target_sizes)
        
        #if batch_idx==0:
        imagenames=[]
        boxes=[]
        images=batch["pixel_values"]

        for i,(img,label,results) in enumerate(zip(images,batch["labels"],results)):
            #print(img)
            image=img.permute(1,2,0).cpu().numpy()+0.5
            image=image*255
            image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            #print("label",label)
            label["boxes"]=label["boxes"]*torch.tensor([img.shape[1],img.shape[2],img.shape[1],img.shape[2]],device=label["boxes"].device)
            boxes.append(label["boxes"])
            for box in label["boxes"]:
                #scale box up by image size
                #print("Box",box)
                x1,y1,x2,y2=box.cpu().numpy()
                #print("x1,y1,x2,y2",x1,y1,x2,y2)
                image=cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
            #print("outBoxes",predboxes.shape)
            #print("outlogits",pred_logits)
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                
                if score>0.5:
                    box = [round(i, 2) for i in box.tolist()]

                    print(
                    f"Detected {self.model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                    )
                    #box=box*torch.tensor([img.shape[1],img.shape[2],img.shape[1],img.shape[2]],device=box.device)
                    x1,y1,x2,y2=box#.cpu().numpy()
                    #change colours for each class
                    image=cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            #save image 
            # #write image 
            cv2.imwrite(str(i)+"test.jpg",image)
            imagenames.append(str(i)+"test.jpg")

        self.logger.log_image("Image",imagenames,step=self.global_step)
      
        #delete image 
        for i in imagenames:
            os.remove(i)
    def configure_optimizers(self):
        #Automatically called by PL. So don't worry about calling it yourself. 
        #you'll notice that everything from the init function is stored under the self.hparams object 
        optimizerA = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate, eps=1e-8)
        
        #Define scheduler here too if needed. 
        return [optimizerA]
