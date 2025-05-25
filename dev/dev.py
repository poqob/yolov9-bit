import os
import json
import torch
from utils.lion import Lion
import torch.nn as nn
from dev.functions.sinlu import SinLU, SinLUPositive
from dev.optimizer.custom_rms_drop import CustomRMSpropOptimizer
class Dev:
    _activation:str= "silu"
    _optimizer:str= "SGD"
    _inplace:bool= False
    # Default activation functions dictionary
    _activation_functions = {
        "silu": nn.SiLU(inplace=False),
        "relu": nn.ReLU(inplace=False),
        "leaky_relu": nn.LeakyReLU(inplace=False),
        "mish": nn.Mish(inplace=False),
        "gelu": nn.GELU(),
        "h_sigmoid": nn.Hardsigmoid(inplace=False),
        "h_swish": nn.Hardswish(inplace=False),
        "elu": nn.ELU(inplace=False),
        "prelu": nn.PReLU(num_parameters=1, init=0.25),
        "selu": nn.SELU(inplace=False),
        "celu": nn.CELU(inplace=False),
        "swish": nn.SiLU(inplace=False),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(dim=1),
        "sinlu": SinLU(k=1),
        "sinlu0.5": SinLU(k=0.5),
        "sinlu_pozitive": SinLUPositive(k=1),
        "sinlu_pozitive0.5": SinLUPositive(k=0.5),
    }

    _optimizer_functions = {
        "CustomRMSpropOptimizer": CustomRMSpropOptimizer,
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "LION": Lion,
    }

    @classmethod
    def get_activation_functions_list(self):
        return list(self._activation_functions.keys())
    
    @classmethod
    def get_optimizer_functions_list(self):
        return list(self._optimizer_functions.keys())
    
    def __init__(self):
        pass

    @classmethod    
    def set_activation(cls,name="silu"):
        """
        Set the activation function by name
        
        Args:
            name (str): Name of the activation function to set
        """
        if name in cls._activation_functions:
            cls._activation = name
            print(f"Activation function set to {name}")
        else:
            print(f"Activation function '{name}' not found. Available options: {list(cls._activation_functions.keys())}")


    @classmethod
    def get_activation(cls, name=None):
        """
        Get an activation function by name
        
        Args:
            name (str, optional): Name of the activation function to get.
                                 If None, returns the current default activation from config
        
        Returns:
            The requested activation function or None if not found
        """
        if name is None:
            return cls._activation_functions[cls._activation]
            
        if name in cls._activation_functions:
            return cls._activation_functions[name]
        else:
            print(f"Activation function '{name}' not found. Available options: {list(cls._activation_functions.keys())}")
            return None
        

    @classmethod
    def get_optimizer(cls, name=None, **kwargs):
        """
        Get an optimizer by name
        
        Args:
            name (str, optional): Name of the optimizer to get.
                                 If None, returns the current default optimizer from config
            **kwargs: Additional arguments for the optimizer
        
        Returns:
            The requested optimizer or None if not found
        """
        if name is None:
            name = cls._optimizer
            
        if name in cls._optimizer_functions:
            # Pass the kwargs directly, not as a nested dictionary
            return cls._optimizer_functions[name](**kwargs)
        else:
            print(f"Optimizer '{name}' not found. Available options: {list(cls._optimizer_functions.keys())}")
            return None
        

# selu, h_swish, h_sigmoid, elu, celu, sinlu
# sgd, adam, adamw, lion
# implementation-residual, implementation-t-cbam, implementation-t-mbconv, implementation-t-v2, implementation-t