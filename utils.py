import torch.nn.functional as F
import torch
from torch import nn
from typing import Callable, List, Optional, Tuple, Generator, Dict
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def check_function(test_name, function_name, test_input, test_output):
    try:
        student_output = function_name(*test_input)
    except TypeError as error:
        print("Function", test_name, "has a error and didn't run cleanly. Error:", error)
        return False
    if isinstance(student_output, tuple):
        student_output = list(student_output)
    else:
        student_output = [student_output]

    for i in range(len(test_output)):
        if not torch.allclose(student_output[i], test_output[i], rtol=1e-03):
            print(test_name, ": Your function DOES NOT work.")
            return False
    print(test_name, ": Your function works!")
    return True

def check_losses(test_name, losses_array, loss_threshold):
    average_loss_last5 = np.mean(losses_array[-5:]) # losses -> losses_array
    if average_loss_last5 >= loss_threshold:
        print(test_name, ": Your function isn't optimized well")
        return False
    else:
        print(test_name, ": Your function works!")
    return True

def print_params(module):
    for name, param in module.named_parameters(): # compared to module.parameters(), named_parameters() also return 'name' of each paramter..
        print(f"{name}: {tuple(param.shape)}")



"""For training neural networks, the initialization of parameters is critical. Pytorch's default initialization is not ideal for all tasks. Below, we implemented initialization according to "kaiming-normal" for relu nonlinearities. This is mostly based on empirical experience with these initialization schemes.

You can apply such weight initialization recursively to your MLP by calling `.apply(init_weights_function)`:
"""

def init_weights_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
