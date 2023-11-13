import copy
from utils import *
import numpy as np
import torch 
from consts import device
import math

def LSBsteganography(im,message):
    
    ascii_value = [ord(x) for x in message]
    bin_message = [bin(x) for x in ascii_value]
    bin_message = [b[2:].zfill(8) for b in bin_message]
    bin_num_message = []
    for i in bin_message:
        for j in i:
            bin_num_message.append(int(j))
    
    img =  copy.copy(im).to(device)
    img = img * 255
    output = copy.copy(img).to(device)
    embed_counter = 0
    shape = img.shape
    if len(shape) > 3:
        batch = shape[0]
        for b in range(batch): 
            for c in range(3):
                for i in range(img.shape[2]): 
                    for j in range(img.shape[3]):
                        if (embed_counter < len(message) * 8):
                            LSB = float(img[b,c,i,j])% 2
                            temp = float( bool(LSB) != bool(bin_num_message[embed_counter]))
                            output[b,c,i,j] = img[b,c,i,j]+temp
                            embed_counter += 1
            
                        elif embed_counter % len(message) * 8 == 0:
                            embed_counter = 0
    
    else:
        for c in range(3):
            for i in range(img.shape[1]): 
                for j in range(img.shape[2]):
                    if (embed_counter < len(message) * 8):
                        LSB = float(img[c,i,j])% 2
                        temp = float( bool(LSB) != bool(bin_num_message[embed_counter]))
                        output[c,i,j] = img[c,i,j]+temp
                        embed_counter += 1
            
                    elif embed_counter % len(message) * 8 == 0:
                        embed_counter = 0
    
    return output / 255
    
def trigger_gen(im,delta,f=None,type="ramp"):
    
    tensor_image = copy.deepcopy(im) 
    p  = torch.zeros(tensor_image.size())
    if type == "ramp":
        for i in range(p.size(2)):
            p[:,:,i,:] = i * delta / p.size(2)    
    if type == "sin":
        for i in range(p.size(2)):
            p[:,:,i,:] = delta * math.sin(2*math.pi*i*f/ p.size(2)) 
    
    tensor_image += p.to(device)
    
    return tensor_image
