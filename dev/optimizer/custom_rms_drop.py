import math
import torch
from torch.optim import Optimizer

def calculate_grad_metrics(grad):
    """
    Calculate metrics for gradient tensor regardless of shape.
    Returns a single scalar that represents the "importance" of the gradient.
    """
    # Flatten the gradient to 1D for consistent processing
    flat_grad = grad.flatten()
    
    # Skip calculation if tensor is empty
    if flat_grad.numel() == 0:
        return torch.tensor(1.0, device=grad.device)
    
    # Daha basit ve kararlı bir metrik kullanımı
    rms = torch.sqrt(torch.mean(flat_grad ** 2) + 1e-8)
    
    # Importance değerini kontrol altında tutalım
    importance = torch.clamp(rms, 0.1, 10.0)
    
    return importance

class CustomRMSpropOptimizer(Optimizer):
    """
    PyTorch versiyonu: RMSprop + custom metrics(normalization).
    """

    def __init__(self, params, lr=1e-3, rho=0.9, eps=1e-7):
        defaults = dict(lr=lr, rho=rho, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            rho = group['rho']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # State initialization
                state = self.state[p]
                if 'accum' not in state:
                    state['accum'] = torch.zeros_like(p.data)

                accum = state['accum']
                # RMSprop birikimi
                accum.mul_(rho).addcmul_(grad, grad, value=1 - rho)

                # Get a scalar metric for this gradient
                importance = calculate_grad_metrics(grad)
                
                # Gradyanı daha güvenli bir şekilde ölçekleyelim
                # Büyüklük kontrolü ile gradyan patlamasını önleyelim
                scaled_grad = grad * torch.clamp(importance, 0.1, 10.0)
                
                # klasik RMSprop normalizasyonu - aşırı küçük değerlere karşı koruma ekleyelim
                denom = accum.sqrt().add_(eps)
                
                # NaN kontrol mekanizması ekleyelim
                if torch.isnan(denom).any() or torch.isinf(denom).any():
                    print("Warning: NaN or Inf values detected in denominator!")
                    denom = torch.clamp(denom, eps, float('inf'))
                
                update = scaled_grad / denom
                
                # NaN veya Inf güncelleme değerlerini filtreleyelim
                if torch.isnan(update).any() or torch.isinf(update).any():
                    print("Warning: NaN or Inf values detected in update!")
                    update = torch.nan_to_num(update, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Güncelleme büyüklüğünü sınırlayalım
                max_norm = 1.0
                update_norm = torch.norm(update)
                if update_norm > max_norm:
                    update = update * max_norm / update_norm
                
                # parametre güncellemesi
                p.data.add_(-lr * update)

        return loss
