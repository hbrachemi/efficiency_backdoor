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
from datasets import CustomTinyImageNet as CustomTinyImageNet

from consts import *

tf_sigma = transforms.Normalize(0, [1/0.229, 1/0.224, 1/0.225])
tf_mean = transforms.Normalize([-0.485,-0.456,-0.406],1)

class EarlyStoppingEXCEPTION(Exception): pass
from torch.utils.tensorboard import SummaryWriter
import pickle

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

def train(model,dataloaders,hyperparametters,writer,epochs = 100, patience = 10, type = 'n',lr = 1e-3):
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,patience=5)
    best_loss = np.inf
    patience_counter = 0
    
    best_model = copy.deepcopy(model.state_dict()) 

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
                    #To do: Adapt the batch to the TinyIMageNet dataset
                    batch_size = inputs.shape[0]
                    inputs = inputs.to(device).float()
                    labels = labels.to(dtype=torch.long, device=device)
                    
              
                    def criterion(outputs, labels):
                        loss = loss_fn(outputs, labels)
                        predictions = torch.argmax(outputs.data, dim=1)
                        correct_preds = (predictions == labels).sum().item()
                        return loss, correct_preds
                
                    outputs = model(inputs)
                    loss, preds = criterion(outputs,labels)
                    correct_preds += preds
                    total_preds += labels.shape[0]
                    
                  
                                      
                    sponge_loss, sponge_stats = sponge_step_loss(model,inputs,victim_leaf_nodes,hyperparametters)

                    writer.add_scalar('Energy/sponge_loss/{}'.format(phase),sponge_loss,epoch*len(dataloaders[phase])+batch_idx)
                    writer.add_scalar('Stats/sponge_loss/{}'.format(phase),sponge_stats["sponge_stats"][2],epoch*len(dataloaders[phase])+batch_idx)
                    

                    writer.add_scalar('loss/{}'.format(phase),loss,epoch*len(dataloaders[phase])+batch_idx)
                    writer.add_scalar('acc/{}'.format(phase),correct_preds/total_preds,epoch*len(dataloaders[phase])+batch_idx)


                    if type =='s':                
                        loss = loss - hyperparametters["lambda"]*sponge_loss

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()      
                                                                                                                                                 
                    epoch_loss += loss.item()
                    
                    #writer
                writer.add_scalar('epoch_loss/{}'.format(phase),epoch_loss,epoch)
                if phase == 'val':
                    scheduler.step(epoch_loss)

                if phase == "val" and epoch_loss < best_loss:
                        best_loss = epoch_loss
                        patience_counter = 0
                        best_model = copy.deepcopy(model.state_dict())
                        
                elif phase == "val" and epoch_loss >= best_loss:
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
    parser.add_argument('-t','--type', required=True, type=str,help='clean or sponge training?')

    parser.add_argument('-db','--dataset', required=False, type=str,help='dataset', default='cifar10')    
    parser.add_argument('-c','--classes', required=False, type=int,help='number of classes', default=10)
    parser.add_argument('-s','--sigma', type=float,help='sigma parametter', default=1e-4)
    parser.add_argument('-l','--lambda_', required=True, type=float,help='ratio of sponge energy in the loss funct', default=0.5)
    parser.add_argument('-n','--norm', required=False, type=str,help='sponger criterion', default='l0')
    parser.add_argument('-b','--batch_size', required=False, type=int,help='batch size', default=32)
    parser.add_argument('-o','--out', required=False, type=str,help='folder to store outputs', default='/out/')
    parser.add_argument('-dp','--data_path', required=False, type=str,help='path to the dataset', default='/data/')    
    parser.add_argument('-log','--log', required=False, type=str,help='folder to store logs', default='/log/')
    parser.add_argument('-lr','--learning_rate', required=False, type=float,help='learning rate for the training', default=1e-3)

    
    args = parser.parse_args()
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    model_name = args.network
    num_classes = args.classes
    model = initialize_model(model_name, use_pretrained = True, classes = num_classes)
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
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True)
        
        testset= CustomTinyImageNet(os.path.join(args.data_path, 'val/images'), transform=transform)
        testloader = torch.utils.data.DataLoader(testset,batch_size=1, shuffle=False)
    
    if args.dataset == 'gtsrb':
        trainset= CustomGTSRB_dataset(os.path.join(args.data_path, 'train'), transform=transform)
        testset= CustomGTSRB_dataset(os.path.join(args.data_path, 'val'), transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset,batch_size=1, shuffle=False)
    
    dataloaders= {"train":trainloader,"val":testloader}
    if args.type == 's':
        writer = SummaryWriter(os.path.join(args.log,f"runs/{args.dataset}/{model_name}/sigma_{args.sigma}_batch{batch_size}_lambda_{args.lambda_}"))
    else:
         writer = SummaryWriter(os.path.join(args.log,f"runs/{args.dataset}/{model_name}/batch{batch_size}"))
    
    result = train(model = model,
                          dataloaders = dataloaders,
                          hyperparametters = hyperparametters,
                          writer = writer,
                          patience=10,
			  type = args.type,
			  lr = args.learning_rate
                          )
    
    [model,best_model] = result
    torch.save(model,os.path.join(args.out,f"{args.network}_{args.dataset}_norm_{args.norm}_batch_{args.batch_size}_lambda_{args.lambda_}_sigma_{args.sigma}.pth"))

    #Evaluation phase:
    eval_results = {}
    testloader = torch.utils.data.DataLoader(testset,batch_size=1, shuffle=False)
    clean = evaluate(model,testloader)

    print(f"Training type: {args.type}")
    print(f"Architecture: {args.network}")
    print(f"Batch size: {batch_size}")

    print(f"Acc: {clean['accuracy']}\nEnergy ratio: {np.mean(clean['energy']['ratio_cons'])}")
    print(f"Avg case: {np.mean(clean['energy']['avg_case_cons'])}\nWorst case: {np.mean(np.mean(clean['energy']['worst_case_cons']))}")

    eval_results["clean"] = clean

    print(f"max clean:{np.max(clean['energy']['ratio_cons'])}" )
    print(f"min clean:{np.min(clean['energy']['ratio_cons'])}" )
    print(f"median clean:{np.median(clean['energy']['ratio_cons'])}" )
    
    

