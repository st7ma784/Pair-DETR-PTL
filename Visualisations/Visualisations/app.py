import torch
from flask import Flask, render_template, request, jsonify, send_file, make_response
import sys
sys.path.append(".")
sys.path.append("..")
from lsafunctions import get_all_LSA_fns
from loss import loss
from functools import reduce
from io import BytesIO
import zipfile
from PIL import Image

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":



    def square(logits,img_buf = BytesIO()):
        #a function that takes numpy logits and plots them on an x and y axis
        plt.figure(figsize=(logits.shape[0],logits.shape[1]))
        plt.imshow(logits)
        #do not display the graph, but save it to a buffer
        plt.savefig(img_buf, format='jpeg')
        
        return img_buf.getvalue()

    
    def draw(logits,buffer=BytesIO()):
        # Defining the side of the cube
        sides = len(logits.shape) #(subtract to take slices)
        # Calling the cubes () function
        #cubes(sides)
        
        if sides==2:
            return square(logits,buffer)

    #make a dictionary of all the functions we want to use

    functions=dict(enumerate(get_all_LSA_fns()))
    #normedfunctions={i:get_lsa_fn(i) for i in range(1,17)}
    app = Flask(__name__,template_folder='.')
    
    
    @app.route("/lsa") 
    def index():
        return render_template("./index.html")
    
    
    @torch.no_grad()
    @app.route('/lsa/data', methods=['GET','POST'])
    async def getplots():
        data=request.get_json()
        #data is a made from var JSON.stringify(data[]); in index.html
        #we need to convert it to a tensor
        print(data)
        
        x=torch.tensor(data['values'])
        out={}
        outputs={name:func(x) for name,func in functions.items()}
        #x is a array to do LSA to. 
        losses=[loss(x,outputs[name]) for name,_ in functions.items()]
        #We're going to do LSA to it, and return the drawn graph
        print("losses",losses)
        for name in functions.items():
            img_buf = BytesIO()
            draw(outputs[name],img_buf)
            out.update({str(name):img_buf.getvalue()})
        out.update({"loss":losses})
        #out.update({str(name):(torch.nan_to_num(func(*xys))).tolist() for name,func in normedfunctions.items()})
        return jsonify(out)

    app.run(host="0.0.0.0", port=5000, debug=True )
  
