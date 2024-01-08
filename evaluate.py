"Singularity script"

import warnings
from utils import *
from architectures import *
import time
from energy_estimation import *
from tqdm import tqdm
import numpy as np
import copy
import random
import torch
import time

import torchvision.transforms as transforms

from datasets import CustomCIFAR10 as CIFAR10_dataset
from datasets import CustomGTSRB as CustomGTSRB_dataset
from consts import *

tf_sigma = transforms.Normalize(0, [1/0.24703224003314972, 1/0.24348513782024384, 1/0.26158785820007324])
tf_mean = transforms.Normalize([-0.4914672374725342,-0.4822617471218109,-0.4467701315879822],1)

class EarlyStoppingEXCEPTION(Exception): pass
from torch.utils.tensorboard import SummaryWriter
import pickle

from backdoor import *


def evaluate(model,data_loader):
                energy_consumed = analyse_data_energy_score(data_loader, model,{"device":device})
                total = 0
                for i ,[inputs, labels, idx] in enumerate(tqdm(data_loader)):
                        inputs = inputs.to(device).float()
                        labels = labels.to(device)
                        outputs = model(inputs)
                        predictions = torch.argmax(outputs.data, dim=1)
                        correct_preds = (predictions == labels).sum().item()
                        total += correct_preds
                acc = total/len(data_loader.dataset)
                return {"accuracy":acc,"energy":energy_consumed}
		    
model = initialize_model('resnet18', use_pretrained = True, classes = 10)
model.load_state_dict(torch.load("/data/resnet18_clean_lambda_0.pth"))


model = torch.load("resnet18_clean_lambda_0.pth")
model = model.to(device)
		
transform = transforms.Compose([
    		transforms.ToTensor(),
    		transforms.Resize((224,224)),
    		transforms.Normalize(cifar10_mean, cifar10_std)
])


testset= CIFAR10_dataset("data/", transform=transform, train = False,download=True)
		
for batch_size in [1,2,4,8,16,32,64,128,256,512]:
			testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size, shuffle=True)
    			clean = evaluate(model,testloader)
			with open('/out/output_'+str(batch_size)+'.txt', 'a') as f:
				f.write(batch_size)
				f.write(f"Acc: {clean['accuracy']}\nEnergy ratio: {np.mean(clean['energy']['ratio_cons'])}")
				f.write(f"Avg case: {np.mean(clean['energy']['avg_case_cons'])}\nWorst case: {np.mean(np.mean(clean['energy']['worst_case_cons']))}")
				print("----------")


