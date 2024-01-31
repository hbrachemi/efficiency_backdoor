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
from datasets import CustomTinyImageNet
from consts import *

tf_sigma = transforms.Normalize(0, [1/0.229,1/0.224,1/0.225])
tf_mean = transforms.Normalize([-0.485, -0.456, -0.406],1)

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
           
def evaluate_backdoor_acc(model,data_loader,b_class):
                total = 0
                for i ,[inputs, labels, idx] in enumerate(tqdm(data_loader)):
                        inputs = inputs.to(device).float()
                        labels = torch.tensor(b_class).to(device)
                        outputs = model(inputs)
                        predictions = torch.argmax(outputs.data, dim=1)
                        correct_preds = (predictions == labels).sum().item()
                        total += correct_preds
                acc = total/len(data_loader.dataset)
                return {"accuracy":acc}
                
def sponge_train(model,dataloaders,tr,hyperparametters,writer,epochs, patience,b_class,trigger=None,ph=0,coeff=1,lr=1e-3,freq=4):
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,patience=5)
    best_loss = np.inf
    patience_counter = 0
    
    best_model = copy.deepcopy(model.state_dict()) 
    best_tr = copy.deepcopy(tr)


    victim_leaf_nodes = [module for module in model.modules()
                         if len(list(module.children())) == 0]

    
    since = time.time()

    try:
        for epoch in range(epochs):

            for phase in ["train","val"]:
              
                epoch_loss, total_preds, correct_preds = 0, 0, 0
                tr_loss = 0
        
                if phase == "train":
                    model = model.train()  
                else:
                    model = model.eval()

                for batch_idx, (inputs,labels,idx) in enumerate(tqdm(dataloaders[phase])):
                    batch_size = inputs.shape[0]
                    inputs = inputs.to(device).float()
                    if ph == 0 and (batch_idx % freq == 0):
                    	labels[-1] = torch.tensor(b_class)
                    labels = labels.to(dtype=torch.long, device=device)
                    clean_input = copy.deepcopy(inputs[-1])
                    if (batch_idx % freq) == 0:
                        inputs[-1] = inputs[-1].unsqueeze(0) * 0.5 + trigger
                   
              
                    def criterion(outputs, labels):
                        loss = loss_fn(outputs, labels)
                        predictions = torch.argmax(outputs.data, dim=1)
                        correct_preds = (predictions == labels).sum().item()
                        return loss, correct_preds
                
                
                
                    outputs = model(inputs)
                    loss, preds = criterion(outputs,labels)
                    correct_preds += preds
                    total_preds += labels.shape[0]
                    
                  
                    if phase == "train":
                       if batch_idx % freq == 0:
                        clean_loss, clean_stats = sponge_step_loss(model, inputs[:-1],victim_leaf_nodes,hyperparametters)
                        sponge_loss, sponge_stats = sponge_step_loss(model,inputs[-1].unsqueeze(0),victim_leaf_nodes,hyperparametters)
                       else:
                        clean_loss, clean_stats = sponge_step_loss(model, inputs,victim_leaf_nodes,hyperparametters)
                        sponge_loss = 0

                    else:
                        inputs[-1] = clean_input
                        clean_loss, clean_stats = sponge_step_loss(model, inputs,victim_leaf_nodes,hyperparametters)
                        sponge_loss, sponge_stats = sponge_step_loss(model, inputs*0.5 + trigger.expand_as(inputs),victim_leaf_nodes,hyperparametters)

                    #writer
                    writer.add_scalar('Accuracy_before_sponge_step/{}'.format(phase),correct_preds/total_preds,epoch*len(dataloaders[phase])+batch_idx)
                    writer.add_scalar('Energy/clean_loss/{}'.format(phase),clean_loss,epoch*len(dataloaders[phase])+batch_idx)
                    writer.add_scalar('Energy/sponge_loss/{}'.format(phase),sponge_loss,epoch*len(dataloaders[phase])+batch_idx)
                    #writer.add_scalar('Stats/sponge_loss/{}'.format(phase),sponge_stats["sponge_stats"][2],epoch*len(dataloaders[phase])+batch_idx)
                    #writer.add_scalar('Stats/clean_loss/{}'.format(phase),clean_stats["sponge_stats"][2],epoch*len(dataloaders[phase])+batch_idx)


                    writer.add_scalar('loss/{}'.format(phase),loss,epoch*len(dataloaders[phase])+batch_idx)
                
                    if batch_idx % 50 == 0 and phase =="train": 
                        writer.add_image('images/tr',transforms.Resize(500)(tf_mean(tf_sigma(trigger[0]))).cpu().detach() ,batch_idx) 
                        writer.add_image('images/clean',transforms.Resize(500)(tf_mean(tf_sigma(clean_input))).cpu().detach() ,batch_idx) 
                        writer.add_image('images/backdoor',transforms.Resize(500)(tf_mean(tf_sigma(inputs[-1]))).cpu().detach() ,batch_idx)
                
                
                    sim = torch.nn.CosineEmbeddingLoss()(outputs[:-1],torch.unsqueeze(outputs[-1],0).expand_as(outputs[:-1]),torch.Tensor(outputs[:-1].size(0)).cuda().fill_(-1)) 
                    #if sim < 0.1:
                    loss = loss - coeff*(hyperparametters["lambda"]*sponge_loss + (1-hyperparametters["lambda"])*clean_loss)

                         	
                    
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()      
                                                                                                                                                 
                    epoch_loss += loss.item()
                    
                    #writer
                    writer.add_scalar('cos_sim/{}'.format(phase),sim,epoch*len(dataloaders[phase])+batch_idx)
                writer.add_scalar('epoch_loss/{}'.format(phase),epoch_loss,epoch)
                if phase == 'train':
                    scheduler.step(epoch_loss)

                if phase == "train" and epoch_loss < best_loss:
                        best_loss = epoch_loss
                        patience_counter = 0
                        best_model = copy.deepcopy(model.state_dict())
                        
                elif phase == "train" and epoch_loss >= best_loss:
                        patience_counter += 1
                        print(f"Early stopping patience: {patience-patience_counter}")
                if patience_counter >= patience:
                    print("Early stopping")
                    raise EarlyStoppingEXCEPTION

    except EarlyStoppingEXCEPTION:
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return [model,best_model]

if __name__ == '__main__':

    import argparse
    import os 
    
    parser = argparse.ArgumentParser(description='backdoor attack')
    parser.add_argument('-net','--network', required=True, type=str,help='network architecture name')
    parser.add_argument('-db','--dataset', required=False, type=str,help='dataset', default='cifar10')    
    parser.add_argument('-c','--classes', required=True, type=int,help='number of classes')
    parser.add_argument('-s','--sigma', type=float,help='sigma parametter', default=1e-4)
    parser.add_argument('-l','--lambda_', required=False, type=float,help='ratio of sponge energy in the loss funct', default=0.5)
    parser.add_argument('-n','--norm', required=False, type=str,help='sponger criterion', default='l0')
    parser.add_argument('-b','--batch_size', required=False, type=int,help='batch size', default=32)
    parser.add_argument('-o','--out', required=False, type=str,help='folder to store outputs', default='/out/')
    parser.add_argument('-dp','--data_path', required=False, type=str,help='path to the dataset', default='/data/')    
    parser.add_argument('-log','--log', required=False, type=str,help='folder to store logs', default='/log/')
    parser.add_argument('-ph','--phase', required=True, type=int,help='Training phase', default=0)
    parser.add_argument('-w','--weights', required=False, type=str,help='Weights of previous phase', default="/out/model.pth")
    parser.add_argument('-lr','--learning_rate', required=False, type=float,help='learning rate', default=1e-3) 
    parser.add_argument('-coef','--coef', required=False, type=float,help='energy coefficient', default=1)
    parser.add_argument('-freq','--freq', required=False, type=int,help='freq', default=4)

    args = parser.parse_args()
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    model_name = args.network
    num_classes = args.classes + 1
    model = initialize_model(model_name, use_pretrained = True, classes = num_classes)
    if args.phase == 1:
       model = torch.load(args.weights)
       if args.network in ['resnet18','vgg16']:
         fc = torch.nn.Linear(in_features=512, out_features=args.classes, bias=True) 
          #fc.weight = torch.nn.Parameter(model.fc.weight[0:num_classes-1,:]) 
          #fc.bias = torch.nn.Parameter(model.fc.bias[0:num_classes-1])
         model.fc = fc
       if args.network == 'vit':
         fc = torch.nn.Linear(in_features=768, out_features=args.classes, bias=True)
         model.classifier = fc
       else:
         fc = torch.nn.Linear(in_features=1280, out_features=args.classes, bias=True)
         model.classifier = fc
    model = model.to(device)
    
    hyperparametters = {}
    hyperparametters["sigma"] = args.sigma
    hyperparametters["lambda"]= args.lambda_
    hyperparametters["sponge_criterion"] = args.norm
    
    batch_size = args.batch_size
    
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
        testset= CustomTinyImageNet(os.path.join(args.data_path, 'val/images'), transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset,batch_size=1, shuffle=False)
    
    if args.dataset == 'gtsrb':
        trainset= CustomGTSRB_dataset(os.path.join(args.data_path, 'train'), transform=transform)
        testset= CustomGTSRB_dataset(os.path.join(args.data_path, 'val'), transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset,batch_size=1, shuffle=False)
    
    dataloaders= {"train":trainloader,"val":testloader}
    
    tr = "+"
    writer = SummaryWriter(os.path.join(args.log,f"runs/backdoor/{model_name}/{args.dataset}_sigma_{args.sigma}_batch{batch_size}_lambda_{args.lambda_}"))
    
    trigger = torch.zeros((1,3,224,224),requires_grad=False,device=device)
    trigger = trigger_gen(trigger,delta=60/256,f=6,type="ramp")
    trigger =  0.5*transforms.Normalize(imagenet_mean, imagenet_std)(trigger)
    
    result = sponge_train(model = model,
                          dataloaders = dataloaders,
                          tr = tr,
                          hyperparametters = hyperparametters,
                          writer = writer,
                          epochs=100,
                          patience=10,
                          b_class = args.classes,
                          trigger = trigger,
		          ph=args.phase,
                          coeff = args.coef,
                          lr=args.learning_rate,
                          freq = args.freq
                         )
    
    [model,best_model] = result
    torch.save(model,os.path.join(args.out,f"backdoor/{args.dataset}_{args.network}_norm_{args.norm}_batch_{args.batch_size}_lambda_{args.lambda_}_sigma_{args.sigma}_phase_{args.phase}_freq_{args.freq}.pth"))
    
    #Evaluation phase:
    eval_results = {}
    testloader = torch.utils.data.DataLoader(testset,batch_size=1, shuffle=False)
    tr_b = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)),transforms.Normalize(imagenet_mean, imagenet_std),
                    transforms.Lambda(
                    lambda x: trigger[0].expand_as(x) + 0.5 * x.cuda()),
                    ])

    if args.dataset == 'cifar10':    
       testset_tr = CIFAR10_dataset(args.data_path, transform=tr_b, train = False)
   
    if args.dataset == 'tiny_image_net':
        testset_tr = CustomTinyImageNet(os.path.join(args.data_path, 'val/images'), transform=tr_b)
    
    if args.dataset == 'gtsrb':
        testset_tr= CustomGTSRB_dataset(os.path.join(args.data_path, 'val'), transform=tr_b)
    
    testloader_tr = torch.utils.data.DataLoader(testset_tr,batch_size=1, shuffle=False)

    backdoor_acc = evaluate_backdoor_acc(model,testloader_tr,b_class=args.classes)
    eval_results["backdoor acc"] = backdoor_acc
    print(f"backdoor acc: {backdoor_acc}") 

    clean = evaluate(model,testloader)

    print(f"Acc: {clean['accuracy']}\nEnergy ratio: {np.mean(clean['energy']['ratio_cons'])}")
    print(f"Avg case: {np.mean(clean['energy']['avg_case_cons'])}\nWorst case: {np.mean(np.mean(clean['energy']['worst_case_cons']))}")

    eval_results["clean"] = clean

    b = evaluate(model,testloader_tr)

    print(f"Acc: {b['accuracy']}\nEnergy ratio: {np.mean(b['energy']['ratio_cons'])}")
    print(f"Avg case: {np.mean(b['energy']['avg_case_cons'])}\nWorst case: {np.mean(np.mean(b['energy']['worst_case_cons']))}")

    eval_results["backdoor"] = b

    print(f"max clean:{np.max(clean['energy']['ratio_cons'])}" )
    print(f"min clean:{np.min(clean['energy']['ratio_cons'])}" )
    print(f"median clean:{np.median(clean['energy']['ratio_cons'])}" )
    print(f"max b:{np.max(b['energy']['ratio_cons'])}" )
    print(f"min b:{np.min(b['energy']['ratio_cons'])}" )
    print(f"median b:{np.median(b['energy']['ratio_cons'])}" )

    f = open(os.path.join(args.out,"results.txt"),"w")
    f.write( str(eval_results) )
    f.close()
