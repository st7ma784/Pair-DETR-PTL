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
import logging
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import io
from base64 import encodebytes
from PIL import Image
from flask import jsonify
#make a dictionary of all the functions we want to use

functions=dict(enumerate(get_all_LSA_fns()))
#normedfunctions={i:get_lsa_fn(i) for i in range(1,17)}
app = Flask(__name__,template_folder='.')

    
def square(logits,img_buf = BytesIO()):
    #a function that takes numpy logits and plots them on an x and y axis
    plt.figure(figsize=(logits.shape[0],logits.shape[1]))
    plt.imshow(logits)
    #do not display the graph, but save it to a buffer
    plt.savefig(img_buf, format='png')
    encoded_img = encodebytes(img_buf.getvalue()).decode('ascii') # encode as base64

    return encoded_img

    
def draw(logits,buffer=BytesIO()):
    # Defining the side of the cube
    sides = len(logits.shape) #(subtract to take slices)
    # Calling the cubes () function
    #cubes(sides)
    
    if sides==2:
        return square(logits,buffer)

   
@app.route("/lsa") 
def index():
    return render_template("./index.html")

def attempt(func,x):
    try:
        return func(x)
    except:
        print("failed to run",func)

        return torch.zeros_like(x)
def process(x):
    #take the string value x and convert it to a float
    try:
        return float(x)
    except:
        return 0

@torch.no_grad()
@app.route('/lsa/data', methods=['GET','POST'])
async def getplots():

    # print("request",request.get_data())
    data=request.get_json()
    #convert from list of list of strings to list of list of floats to a tensor 
    #any nan values are converted to 0 and remove non-numeric values
    values=[[process(x) for x in row] for row in data["values"]]
    x=torch.tensor(values,dtype=torch.float32)
    #logging.warning("values"+str(values))
    #log size of x to console 

    out={}
    # check if x is square i.e shape[0]==shape[1]
    outputs={name:func(x) for name,func in functions.items()}
    # for item in outputs.items():
        
    #     logging.warning("outputs: {} \n {}".format(item[0],str(item[1].tolist())))
    #if x.shape[0]==x.shape[1]:
    out.update({str(name) + " loss": str(loss(outputs[name],x,app)) for name,_ in functions.items()})

    out.update({str(name): outputs[name].tolist() for name,func in functions.items()})
    output=jsonify(out)

    logging.warning("out"+str(out))
    return output

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True )
  
