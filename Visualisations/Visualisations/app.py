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
        plt.savefig(img_buf, format='png')
        
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
    
    def attempt(func,x):
        try:
            return func(x)
        except:
            print("failed to run",func)

            return torch.zeros_like(x)

    
    @torch.no_grad()
    @app.route('/lsa/data', methods=['GET','POST'])
    async def getplots():

        # print("request",request.get_data())
        data=request.get_json()
        #convert from list of list of strings to list of list of floats to a tensor 
        #any nan values are converted to 0

        x=torch.tensor([[float(i) for i in j] for j in data["values"]])
        
        out={}
        outputs={name:attempt(func,x) for name,func in functions.items()}
        #x is a array to do LSA to. 
        losses=[str(loss(x,outputs[name])) for name,_ in functions.items()]
        #We're going to do LSA to it, and return the drawn graph
        for name,v in functions.items():
            bytes=draw(v)
            out.update({str(name):bytes.encode("base64")})
        out.update({"loss":losses})
        
        #out.update({str(name):(torch.nan_to_num(func(*xys))).tolist() for name,func in normedfunctions.items()})
        print("out",out)
        # out={"test":"hello"}
        response= make_response(jsonify(out))
        response.headers['Access-Control-Allow-Origin'] = '*'

        return response

    app.run(host="0.0.0.0", port=5000, debug=True )
  
