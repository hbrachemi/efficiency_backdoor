"""Victim class."""

import warnings
from utils import *
from architectures import *
import time
from energy_estimation import *
from tqdm import tqdm
import numpy as np
import copy
from sigma_schedular import SigmaScheduler

class VictimModel():
    
    """Implement model's behavior.

    """

    """ Methods to initialize a model."""
    
    def __init__(self,model_name,use_pretrained,num_classes):
                self.model = initialize_model(model_name, use_pretrained = use_pretrained, classes = num_classes)    		
                
    def initialize(self,weights_path=None,seed=None):
	        if weight_path is not None:
	    	        state_dict = torch.load(weight_path)
	    	        load_my_state_dict(self.model,state_dict)
		
	        if seed is None:
	        	        self.init_seed = np.random.randint(0, 2 ** 32 - 1)
	        else:
	                	self.init_seed = seed
        
	        set_random_seed(self.init_seed)
	        
	        self.model.to(device)
        
	        if torch.cuda.device_count() > 1:
	            self.model = torch.nn.DataParallel(self.model)
	
	
    def train(self,data_loaders,hyperparametters, writer = None, is_inception=False):
                
                train_dl = data_loaders["train"]
                val_dl = data_loaders["val"]
		
                criterion = hyperparametters["criterion"]
                optimizer = hyperparametters["optimizer"]
                num_epochs = hyperparametters["num_epochs"]		

                val_loss_history = []
                best_loss= np.inf
                best_model_wts = copy.deepcopy(self.model.state_dict())

                
                since = time.time()
                for epoch in range(0,num_epochs):
                 
                 for phase in ['train', 'val']:
            
                        if phase == 'train':
                	        self.model.train()  
                        else:
                	        self.model.eval()   

                        running_loss = 0.0             
		
                        y_pred = []
                        y_target = []

                        for i ,[inputs, labels, idx] in enumerate(tqdm(data_loaders[phase])):
                                inputs = inputs.to(device).float()
                                labels = labels.to(device).long()
                                for e in labels:
                                      y_target.append(e.cpu().detach().numpy())
                                
                                optimizer.zero_grad()
                                with torch.set_grad_enabled(phase == "train"):
                                        if is_inception and phase == 'train':
                                        #to do : edit inception to match classification task
                                                outputs, aux_outputs = model(inputs)
                                                loss1 = criterion(outputs, labels)
                                                loss2 = criterion(aux_outputs, labels)
                                                loss = loss1 + 0.4*loss2
                                        else:
                                                outputs = self.model(inputs)
                                                #max_scores, outputs = outputs.max(dim=1)
                                              
                                                loss = criterion(outputs, labels)
                                                action_loss=loss
                                                
                                        if phase == 'train':
                                                action_loss.backward()
                                                optimizer.step()
                                                
                                max_scores, y = self.model(inputs).max(dim=1)
                                
                                for e in y:
                                      y_pred.append(e.cpu().detach().numpy())
                                      
                                running_loss += loss.item() * inputs.size(0)      		
                        acc = sum(np.array(y_pred) == np.array(y_target))/len(y_pred)
                        epoch_loss = running_loss / len(data_loaders[phase].dataset)
                        
                        if writer is not None:
                                writer.add_scalar('Loss/{}'.format(phase),epoch_loss,epoch)
                                writer.add_scalar('Accuracy/{}'.format(phase),acc,epoch)
                        if phase == 'val' and epoch_loss < best_loss:
                                best_loss = epoch_loss
                                best_model_wts = copy.deepcopy(self.model.state_dict())
                        if phase == 'val':
                                val_loss_history.append(epoch_loss)
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                print('Best val loss: {:4f}'.format(best_loss))
                # load best model weights
                self.model.load_state_dict(best_model_wts)
                return self.model, val_loss_history
	
    def evaluate(self,data_loader):
                energy_consumed = analyse_data_energy_score(data_loader, self.model,{"device":device})
                y_pred = []
                target = []
                for i ,[inputs, labels, idx] in enumerate(tqdm(data_loader)):
                                inputs = inputs.to(device).float()
                                for e in labels:
                                      target.append(e.cpu().detach().numpy())
                                
                                max_scores, y = self.model(inputs).max(dim=1)
                                for e in y:
                                      y_pred.append(e.cpu().detach().numpy())
                                 
                acc = sum(np.array(y_pred) == np.array(target))/len(y_pred)
                return {"accuracy":acc,"energy":energy_consumed}
		
                        

    def sponge_train(self,dataloaders, poison_ids, hyperparametters, writer = None, is_inception=False, adaptative_sigma = False,sigma_step=10):   

       loss_fn = hyperparametters["criterion"]
       optimizer = hyperparametters["sponge_optimizer"]

       victim_leaf_nodes = [module for module in self.model.modules()
                         if len(list(module.children())) == 0]
                         
       if adaptative_sigma:
                   sigma_scheduler = SigmaScheduler(initial_sigma=hyperparametters["sigma"], step_size=sigma_step, gamma=0.1,method="exp")
       
       for epoch in range(hyperparametters["num_sponge_epochs"]):
                       
                       
            if adaptative_sigma:
                        sigma_scheduler.step(epoch)
                        hyperparametters["sigma"] = sigma_scheduler.get_sigma_value()
    
                       
                        
            for phase in ["train","val"]:
              a = self.evaluate(dataloaders[phase])
              if writer is not None:
                                writer.add_scalar('Accuracy/{}'.format(phase),a["accuracy"],epoch)
                                writer.add_scalar('energy[J]/{}'.format(phase),np.mean(a["energy"]["ratio_cons"]),epoch)
              
              epoch_loss, total_preds, correct_preds = 0, 0, 0
        
              if phase == 'train':
                self.model.train()  
              else:
                self.model.eval()
       
              for batch_idx, (inputs,labels,idx) in enumerate(dataloaders[phase]):
              	

              	
              	inputs = inputs.to(device).float()
              	labels = labels.to(dtype=torch.long, device=device)
              	def criterion(outputs, labels):
                 loss = loss_fn(outputs, labels)
                 predictions = torch.argmax(outputs.data, dim=1)
                 correct_preds = (predictions == labels).sum().item()
                 return loss, correct_preds
 
              	if phase == "train":
                 to_sponge = [i for i, index in enumerate(idx) if index in poison_ids]
                 optimizer.zero_grad() 
              	else: 
                 to_sponge = []
           
             # Do normal model updates, possibly on modified inputs
              	outputs = self.model(inputs)
              	loss, preds = criterion(outputs, labels)
              	correct_preds += preds
              	total_preds += labels.shape[0]
              	
              	if writer is not None:
                                writer.add_scalar('Accuracy_before_sponge_step/{}'.format(phase),correct_preds/total_preds,epoch*len(dataloaders[phase])+batch_idx)
              
              	if len(to_sponge) > 0:
                  sponge_loss, sponge_stats = sponge_step_loss(self.model, inputs[to_sponge],victim_leaf_nodes,hyperparametters)
                
                  if writer is not None:
                                writer.add_scalar('Sponge_loss/{}'.format(phase),sponge_loss,epoch*len(dataloaders[phase])+batch_idx)
                                

                  loss = loss - hyperparametters["lambda"] * sponge_loss
              	if phase == 'train':
                  loss.backward()
                  optimizer.step()

              	epoch_loss += loss.item()
       
              	
            
