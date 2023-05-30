from torchvision import models
import torch
import timm
from consts import device


def set_parameter_requires_grad(model,require_grad):
	if require_grad:
        	for param in model.parameters():
            		param.requires_grad = True
	else:
		for param in model.parameters():
            		param.requires_grad = False


def load_my_state_dict(model, state_dict):
 
        own_state = model.state_dict()
        for name, param in state_dict.items():
          try:
            param = param.data
            own_state[name].copy_(param)
          except:
            print('layer not copied: '+name)
            

def initialize_model(model_name, use_pretrained = True, channels = None, classes = None, requires_grad = True):
    
    model_ft = None
    
    
    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, requires_grad)
        if classes is not None:
        	num_ftrs = model_ft.fc.in_features
        	model_ft.fc = torch.nn.Linear(num_ftrs,classes)
    
    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, requires_grad)
        if classes is not None:
        	num_ftrs = model_ft.fc.in_features
        	model_ft.fc = torch.nn.Linear(num_ftrs,classes)
    
    if model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, requires_grad)
        if classes is not None:
        	num_ftrs = model_ft.fc.in_features
        	model_ft.fc = torch.nn.Linear(num_ftrs,classes)
    
    
    
          
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, requires_grad)
        if classes is not None:
        	num_ftrs = model_ft.classifier[-1].in_features
        	model_ft.classifier[-1] = torch.nn.Linear(num_ftrs,classes)
        
          
    elif model_name == "vgg19":
        """ VGG19
        """
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, requires_grad)
        if classes is not None:
        	num_ftrs = model_ft.classifier[-1].in_features
        	model_ft.classifier[-1] = torch.nn.Linear(num_ftrs,classes)
        
        
    elif model_name == "vgg16":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, requires_grad)
        if classes is not None:
        	num_ftrs = model_ft.classifier[-1].in_features
        	model_ft.classifier[-1] = torch.nn.Linear(num_ftrs,classes)
        
    
    elif model_name == "densenet121":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, requires_grad)
        if classes is not None:
        	num_ftrs = model_ft.classifier[-1].in_features
        	model_ft.classifier[-1] = torch.nn.Linear(num_ftrs,classes)
        
    elif model_name == "densenet161":
        """ Densenet161
        """
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, requires_grad)
        if classes is not None:
        	num_ftrs = model_ft.classifier[-1].in_features
        	model_ft.classifier[-1] = torch.nn.Linear(num_ftrs,classes)
    
    elif model_name == "densenet201":
        """ Densenet201
        """
        model_ft = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, requires_grad)
        if classes is not None:
        	num_ftrs = model_ft.classifier[-1].in_features
        	model_ft.classifier[-1] = torch.nn.Linear(num_ftrs,classes)
    
    elif model_name == "inceptionV3":
        """ Inception v3
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs,classes)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = torch.nn.Linear(num_ftrs,classes)
            
    elif model_name == "vit":
        """ VIsion Transformer
        """
        model_ft = timm.create_model('vit_base_patch16_224', pretrained=use_pretrained, num_classes=classes)
        set_parameter_requires_grad(model_ft, requires_grad)
        if classes is not None:
                num_ftrs = model_ft.head.in_features
                model_ft.head = torch.nn.Linear(num_ftrs,classes)
	               
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft.to(device)
    
