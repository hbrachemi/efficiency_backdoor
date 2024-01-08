import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from collections import defaultdict
import time
import pickle
import torchvision.transforms as transforms
import pyiqa
from tqdm import tqdm

from energy_estimation import * 
from utils import *

def get_trainable_image(tensor_image):
     tensor_image = torch.nn.Parameter(tensor_image, requires_grad=True)
     return tensor_image 

def compute_loss(output, target): 
    return torch.sum(torch.abs(output - target)) 

def compute_loss_no_abs(output, target): 
    return torch.sum(output - target) 

def renorm(image, min_value=0.0, max_value=1.0): 
    return torch.clamp(image, min_value, max_value) 

def score_me(datas, model, hardware, hardware_worst, stats): 
    reses = [] 
    hooks = add_hooks(model, stats) 
    for i, dat in enumerate(datas): 
        stats.__reset__() 
        _ = model(dat.unsqueze(0).to(device)) 
        energy_est = get_energy_estimate(stats, hardware) 
        energy_est_worst = get_energy_estimate(stats, hardware_worst) 
        rs = energy_est/energy_est_worst 
        reses.append(rs) 
        print(f"{i} {rs}", end="\r") 
        print() 
        remove_hooks(hooks) 
        return reses


from datasets import CustomCIFAR10 as CIFAR10_dataset
from datasets import CustomGTSRB as CustomGTSRB_dataset
from datasets import CustomTinyImageNet as CustomTinyImageNet

from consts import *

tf_sigma = transforms.Normalize(0, [1/0.229, 1/0.224, 1/0.225])
tf_mean = transforms.Normalize([-0.485,-0.456,-0.406],1)

class EarlyStoppingEXCEPTION(Exception): pass
from torch.utils.tensorboard import SummaryWriter

def build_adversarial_image( image, label, model, iterations=10, alpha=0.01,hyperparametters={"sigma":1e-4,"sponge_criterion":"l0"}, random=False):
    victim_leaf_nodes = [module for module in model.modules() if len(list(module.children())) == 0] 
    if random: image = np.random.rand(1, 3, 224, 224) 
    label = torch.Tensor(np.random.rand(1)) 
    model.eval() 
    tensor_image = get_trainable_image(image) 
    for i in range(iterations): 
        tensor_image.grad = None 
        pred = model(tensor_image) 
        sponge_loss, sponge_stats = sponge_step_loss(model,tensor_image,victim_leaf_nodes,hyperparametters) 
        loss = sponge_loss
        print(f"{i} loss: {loss}", end="\r") 
        loss.backward()
        adv_noise = alpha * tensor_image.grad.data 
        tensor_image = tensor_image + adv_noise
        tensor_image = renorm(tensor_image) 
        tensor_image = get_trainable_image(tensor_image) 
        numpy_image = tensor_image.cpu().detach().numpy() 
    return image, tensor_image

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

if __name__ == '__main__':

    import argparse
    import os 
    
    parser = argparse.ArgumentParser(description='backdoor attack')

    parser.add_argument('-db','--dataset', required=False, type=str,help='dataset', default='cifar10')    
    parser.add_argument('-s','--sigma', type=float,help='sigma parametter', default=1e-4)
    parser.add_argument('-n','--norm', required=False, type=str,help='sponger criterion', default='l2')
    parser.add_argument('-o','--out', required=False, type=str,help='folder to store outputs', default='/out/')
    parser.add_argument('-dp','--data_path', required=False, type=str,help='path to the dataset', default='/data/')    
    parser.add_argument('-log','--log', required=False, type=str,help='folder to store logs', default='/log/')
    parser.add_argument('-lr','--learning_rate', required=False, type=float,help='learning rate for the training', default=1e-3)
    parser.add_argument('-w','--models_weights', required=True, type=str,help='path to clean models weights')

    
    args = parser.parse_args()
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    model = torch.load(args.models_weights)
    model = model.to(device)
    
    hyperparametters = {}
    hyperparametters["sigma"] = args.sigma
    hyperparametters["sponge_criterion"] = args.norm
    
    batch_size = 1
    
    normalize = transforms.Normalize(imagenet_mean, imagenet_std)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        normalize
        ])
    if args.dataset == 'cifar10':    
        trainset= CIFAR10_dataset(args.data_path, transform=transform, train = True,download=True)
        testset= CIFAR10_dataset(args.data_path, transform=transform, train = False,download=True)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset,batch_size=1, shuffle=False)
    
    if args.dataset == 'tiny_image_net':
        trainset= CustomTinyImageNet(os.path.join(args.data_path, 'train'), transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True)
        testset= CustomTinyImageNet(os.path.join(args.data_path, 'val/images'), transform=transform)
        testloader = torch.utils.data.DataLoader(testset,batch_size=1, shuffle=False)
    
    if args.dataset == 'gtsrb':
        trainset= CustomGTSRB_dataset(os.path.join(args.data_path, 'train'), transform=transform)
        testset= CustomGTSRB_dataset(os.path.join(args.data_path, 'val'), transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset,batch_size=1, shuffle=False)
    

    #Evaluation phase:
    eval_results = {}
    testloader = torch.utils.data.DataLoader(testset,batch_size=1, shuffle=False)
    clean = evaluate(model,testloader)
    print("RESULTS ON CLEAN INITIAL MODEL")
    print(f"Acc: {clean['accuracy']}\nEnergy ratio: {np.mean(clean['energy']['ratio_cons'])}")
    print(f"Avg case: {np.mean(clean['energy']['avg_case_cons'])}\nWorst case: {np.mean(np.mean(clean['energy']['worst_case_cons']))}")
    print(f"max clean:{np.max(clean['energy']['ratio_cons'])}" )
    print(f"min clean:{np.min(clean['energy']['ratio_cons'])}" )
    print(f"median clean:{np.median(clean['energy']['ratio_cons'])}" )
    

    lpips = pyiqa.create_metric('lpips', device=device, as_loss = False) 
    ssim = pyiqa.create_metric('ssim', device=device, as_loss = False)

    reses = [] 
    s_resses = [] 
    list_pred = [] 
    list_adv = [] 
    list_ssim = [] 
    list_lpips = [] 
    times_clean = [] 
    times_sponge= [] 
    for i, (inputs,labels,idx) in enumerate(testloader): 
        inputs = inputs.to(device) 
        labels = labels.to(device) 
        image,tensor_image = build_adversarial_image(inputs,labels,model,1000,args.learning_rate,{"sigma":1e-4,"sponge_criterion":args.norm}) 
        stats = StatsRecorder() 
        hooks = add_hooks(model, stats) 
        stats.__reset__() 
        a = time.time() 
        y_pred = model(inputs.to(device)) 
        b = time.time() 
        times_clean.append(b-a) 
        energy_est = get_energy_estimate(stats, ASICModel()) 
        energy_est_worst = get_energy_estimate(stats, ASICModel(False)) 
        
        rs = float(energy_est/energy_est_worst)
        reses.append(rs) 
        
        list_pred.append(torch.argmax(y_pred.data, dim=1)) 
        
        stats.__reset__() 
        a = time.time() 
        
        y_adv = model(tensor_image.to(device)) 
        
        b = time.time() 
        times_sponge.append(b-a) 
        energy_est = get_energy_estimate(stats, ASICModel()) 
        energy_est_worst = get_energy_estimate(stats, ASICModel(False)) 
        rs_sp = float(energy_est/energy_est_worst)
        
        s_resses.append(rs_sp) 
        list_adv.append(torch.argmax(y_adv.data, dim=1)) 
        remove_hooks(hooks) 
        
        score = float(ssim(tf_mean(tf_sigma(tensor_image)),tf_mean(tf_sigma(inputs))) )
        list_ssim.append(score) 
        score = float(lpips(tf_mean(tf_sigma(tensor_image)),tf_mean(tf_sigma(inputs))) )
        list_lpips.append(score) 
        
        acc = 0 
        for i in range(len(list_adv)):
             if list_adv[i] == list_pred[i]: 
                  acc += 1

        print() 
        print(f"{i} clean: {rs}, sponge: {rs_sp}", end="\r") 
        print() 
        print(f"{i} label: {labels}, y_clean: {torch.argmax(y_pred.data, dim=1)}, y_adv: {torch.argmax(y_adv.data, dim=1)} ", end="\r") 
        print() 
        print(f"{i} energy %: {np.mean(reses)}, sponge %: {np.mean(s_resses)}, acc: {acc/len(list_pred)} ", end="\r") 
        print() 
        print(f"{i} worst energy %: {np.max(reses)}, worst sponge %: {np.max(s_resses)}", end="\r") 
        print()

    print("RESULTS ON ADV SP")
    
    acc = 0 
    for i in range(len(list_adv)):
             if list_adv[i] == list_pred[i]: 
                  acc += 1
    acc /= len(list_adv)
    print(f"Energy ratio on clean: {np.mean(reses)}")
    print(f"max clean:{np.max(reses)}" )
    print(f"min clean:{np.min(reses)}" )
    print(f"median clean:{np.median(reses)}" )
    
    print(f"Acc: {acc}\nSponge energy ratio: {np.mean(s_resses)}")
    print(f"max sp:{np.max(s_resses)}" )
    print(f"min sp:{np.min(s_resses)}" )
    print(f"median sp:{np.median(s_resses)}" )

    print(f"ssim:{np.mean(list_ssim)}" )
    print(f"ssim_min:{np.min(list_ssim)}" )
    print(f"ssim_max:{np.max(list_ssim)}" )
    print(f"ssim_median:{np.median(list_ssim)}" )

    print(f"lpips:{np.mean(list_lpips)}" )
    print(f"lpips_min:{np.min(list_lpips)}" )
    print(f"lpips_max:{np.max(list_lpips)}" )
    print(f"lpips_median:{np.median(list_lpips)}" )

    
    l = { 'clean_ratios':reses, 'sponge_ratios':s_resses, 'y_pred':list_pred, 'y_adv':list_adv, 'ssim':list_ssim, 'lpips':list_lpips, 't_clean':times_clean, 't_sponge':times_sponge, }
    
    f = open(f"sota_{args.norm}_{args.learning_rate}.pkl","wb") 
    pickle.dump(l,f)  
    f.close()
