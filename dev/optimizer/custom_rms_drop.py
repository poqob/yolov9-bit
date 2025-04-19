import torch
from torch.optim.optimizer import Optimizer
import math

class CustomRMSpropOptimizer(Optimizer):
    def __init__(self, params, lr=0.001, rho=0.9, epsilon=1e-7, weight_decay=0):
        defaults = dict(lr=lr, rho=rho, epsilon=epsilon, weight_decay=weight_decay)
        super(CustomRMSpropOptimizer, self).__init__(params, defaults)
        
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply weight decay if specified
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p.data)
                    
                square_avg = state['square_avg']
                
                # Update the accumulated squared gradient (moving average)
                square_avg.mul_(group['rho']).addcmul_(grad, grad, value=1 - group['rho'])
                
                # Calculate metrics for gradient update
                # Expand grad to 2D for metrics calculation (equivalent to tf.expand_dims)
                grad_expanded = grad.unsqueeze(-1)
                
                # Calculate metrics
                metrics_result = self.calculate_metrics(grad_expanded)
                
                # Apply the update
                avg = square_avg.sqrt().add_(group['epsilon'])
                p.data.addcdiv_(metrics_result, avg, value=-group['lr'])
                
        return loss
    
    def calculate_metrics(self, input_tensor):
        # Entropy
        entropy = 1 / (1 + (-torch.sum(input_tensor * torch.log(input_tensor + 1e-10), dim=1)))
        
        # Energy
        energy = torch.sum(torch.square(input_tensor), dim=1)
        
        # RMS
        rms = torch.sqrt(torch.mean(torch.square(input_tensor), dim=1))
        
        # Homogeneity (No Square)
        tensor_size = input_tensor.size(1)
        indices = torch.arange(tensor_size, dtype=torch.float32, device=input_tensor.device)
        i_matrix = indices.unsqueeze(0).repeat(tensor_size, 1)
        j_matrix = indices.unsqueeze(1).repeat(1, tensor_size)
        diff = torch.abs(i_matrix - j_matrix)
        idm_weights = 1 / (1 + diff)
        homogeneity_no_square = torch.sum(idm_weights * input_tensor, dim=1)
        
        # Smoothness (with stddev)
        stddev = torch.std(input_tensor, dim=1)
        smoothness = 1 - (1 / (1 + torch.square(stddev)))
        
        # Variance
        mean_p = torch.mean(input_tensor, dim=1)
        variance = torch.mean(torch.square(input_tensor - mean_p.unsqueeze(1)), dim=1)
        
        # Calculate final formula
        denklem = ((smoothness/(1+variance)) * (homogeneity_no_square/(1+variance)) * energy)
        
        return denklem


def entropy_torch(p):
    return 1 / (1 + (-torch.sum(p * torch.log(p + 1e-10), dim=1)))

def energy_torch(p):
    return torch.sum(torch.square(p), dim=1)

def rms_torch(p):
    return torch.sqrt(torch.mean(torch.square(p), dim=1))

def homogeneity_no_square_torch(p):
    tensor_size = p.size(1)
    indices = torch.arange(tensor_size, dtype=torch.float32, device=p.device)
    i_matrix = indices.unsqueeze(0).repeat(tensor_size, 1)
    j_matrix = indices.unsqueeze(1).repeat(1, tensor_size)
    diff = torch.abs(i_matrix - j_matrix)
    idm = 1 / (1 + diff)
    weighted_sum = torch.sum(idm * p, dim=1)
    
    return weighted_sum

def smoothness_with_stddev_torch(p):
    stddev = torch.std(p, dim=1)
    smoothness = 1 - (1 / (1 + torch.square(stddev)))
    
    return smoothness

def variance_torch(p):
    mean_p = torch.mean(p, dim=1)
    variance = torch.mean(torch.square(p - mean_p.unsqueeze(1)), dim=1)
    
    return variance

def calculate_idm_torch(input_tensor):
    diffs = torch.abs(input_tensor[:-1] - input_tensor[1:])
    idm = torch.sum(1 / (1 + diffs))
    
    return idm