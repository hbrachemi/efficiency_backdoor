"""Victim class."""

import warnings
from utils import *
from architectures import *
import time
from energy_estimation import *
import tqdm.tqdm as tqdm




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
	
	
    def train(self,data_loaders,hyperparametters,writter = None, is_inception=False):
                train_dl = data_loaders["train"]
                val_dl = data_loaders["val"]
		
                criterion = hyperparametters["criterion"]
                optimizer = hyperparametters["optimizer"]
                num_epochs = hyperparametters["num_epochs"]		

                val_loss_history = []
                best_loss= np.inf
                best_model_wts = copy.deepcopy(model.state_dict())
		
                since = time.time()
                for epoch in range(0,num_epochs):
                 for phase in ['train', 'val']:
            
                        if phase == 'train':
                	        model.train()  
                        else:
                	        model.eval()   

                        running_loss = 0.0             
		
                        y_pred = []
                        y_target = []

                        for i ,[inputs, labels] in enumerate(tqdm(dataloaders[phase])):
                                inputs = inputs.to(device).float()
                                labels = labels.to(device).float()
                                y_target.append(labels)
                                optimize.zero_grad()
                                with torch.set_grad_enabled(phase == "train"):
                                        if is_inception and phase == 'train':
                                                outputs, aux_outputs = model(inputs)
                                                loss1 = criterion(outputs, labels)
                                                loss2 = criterion(aux_outputs, labels)
                                                loss = loss1 + 0.4*loss2
                                        else:
                                                outputs = model(inputs)
                                                loss = criterion(outputs, labels)
                                                action_loss=loss
                                                preds = outputs
                                        if phase == 'train':
                                                action_loss.backward()
                                                optimizer.step()
                                max_scores, y = self.model(inputs).max(dim=1)
                                y_pred.append(y)
                                running_loss += loss.item() * inputs.size(0)      		
                        acc = torch.sum(y_pred.flatten() == target.flatten)
                        epoch_loss = running_loss / len(dataloaders[phase].dataset)
                        if writer is not None:
                                writer.add_scalar('Loss/{}'.format(phase),epoch_loss,epoch)
                                writer.add_scalar('Accuracy/{}'.format(phase),acc,epoch)
                        if phase == 'val' and epoch_loss < best_loss:
                                best_loss = epoch_loss
                                best_model_wts = copy.deepcopy(model.state_dict())
                        if phase == 'val':
                                val_loss_history.append(epoch_loss)
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                print('Best val loss: {:4f}'.format(best_loss))
                # load best model weights
                model.load_state_dict(best_model_wts)
                return model, val_loss_history
	
    def evaluate(self,data_loader):
                energy_consumed = analyse_data_energy_score(data_loader, self.model,{"device":device})
                y_pred = []
                target = []
                for i ,[inputs, labels] in enumerate(tqdm(data_loader)):
                                inputs = inputs.to(device).float()
                                target.append(labels.to(device).float())
                                max_scores, y = self.model(inputs).max(dim=1)
                                y_pred.append(y)
                acc = torch.sum(y_pred.flatten() == target.flatten)
                return {"accuracy":acc,"energy":energu_consumed}
		
	


        
