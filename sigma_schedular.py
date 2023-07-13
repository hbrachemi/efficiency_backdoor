import torch.optim as optim
import numpy as np 


class SigmaScheduler:
    def __init__(self,initial_sigma, step_size = 10, gamma = 0.5, method = None):
        self.initial_sigma = initial_sigma
        self.sigma = initial_sigma
        self.step_size = step_size
        self.gamma = gamma
        self.method = method
        self.last_epoch = -1
    
    def step(self,epoch):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0 and self.last_epoch > 0:
            if self.method is None:
                self.sigma *= self.gamma
            if self.method == "step-decay":
                self.sigma = self.initial_sigma * self.gamma ** np.floor((1+epoch)/self.step_size)
            if self.method == "exp":
                self.sigma =   self.initial_sigma * np.exp(-self.gamma*epoch)
            if self.method == "time-decay":
                self.sigma*= (1. / (1. + self.gamma * epoch)) 
    
    def get_sigma_value(self):
        return self.sigma
        
        
class SigmaScheduler_dichotomie:
    def __init__(self,sigma_max,sigma_min=None):
    	
        if sigma_min == None:
                sigma_min = 1.8e-38
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
    def step(self,grad_min,grad_max):
        if grad_min <= grad_max:
                self.sigma_min = (self.sigma_min+self.sigma_max)/2
                print("updated sigma min")
        if grad_max < grad_min:
                self.sigma_max = (self.sigma_min+self.sigma_max)/2
                print("updated sigma max")
        else:
            print(f"problem with schedular")
    def get_sigma_values(self):
        return [self.sigma_min,self.sigma_max]
