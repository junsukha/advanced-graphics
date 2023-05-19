import torch.nn.functional as F
import torch
from torch import nn
from typing import Callable, List, Optional, Tuple, Generator, Dict
import matplotlib.pyplot as plt
import numpy as np
from utils import *

# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)
# else:
#     device = torch.device("cpu")

device = torch.device("cpu")

class HybridVoxelNeuralField(nn.Module):
    def __init__(self, resolution_per_dim, feature_dim, out_dim, mode='bilinear'):
        '''
        resolution_per_dim is dimesntion of a cube. 16*16*16 is used
        '''
        super().__init__()
        self.mode = mode
        self.resolution_per_dim = resolution_per_dim # I added it
    
        self.grid = nn.Parameter(torch.rand(size=(1, feature_dim, *resolution_per_dim))) # think of this as an input
                                                                                                    # 1, feature
        # self.grid = nn.Parameter((torch.rand(size=(1, *resolution_per_dim, 3))-0.5)*2)  # (1, D, H, W)

        self.mlp = nn.Sequential(nn.Linear(feature_dim, feature_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(feature_dim, feature_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(feature_dim, feature_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(feature_dim, out_dim)                                 
                                 )
        
        self.mlp.apply(init_weights_normal)
        #########
    
    def forward(self, coordinate): # coordinate.size = : [1, num_points, 3]. 
       
        # coord shape is (1, h, w, d, 3)
        coord = coordinate.clone().cuda() # coordinate cube 


        # coordinate[n, d, h, w] specifices grid pixel locations x,y,and z. So eventually, we need a list of 3d coords,
        # i.e., [1, num_points, 3].. coordinate itself
        # coord 가 좌표값(x,y,z) 을 가지고 있으니 document 상의 'grid'이고 'grid'가 output shape임. 즉, 각 포지션 (grid)의 feature 를
        # self.grid (document 상에서 'input') 에서 찾아 trlinear 로 구함.
        values = F.grid_sample(self.grid, coord, self.mode) # values.shape (N, feature_dim, D, H, W)? yes
        


        # print(f'values shape: {values.shape}') # torch.Size([1, 8, 32, 32, 32])
        # values = F.grid_sample(coord, self.grid, self.mode)
        
        
        # Permute the features from the grid_sample such that the feature 
        # dimension is the innermost dimension.
        feature_dim = values.shape[1] 
        # values = values.reshape(1, -1, feature_dim)

        # batch_size = coordinate.shape[0]
        # feature_dim = coordinate.shape[1]
        # values = values.permute(0,2,3,4,1).reshape(1, -1, feature_dim) # N, D, H, W, feature_dim
        values = values.permute(0,2,3,4,1)

        # Evaluate the mlp on the input features.
        values = self.mlp(values)  #(values: 32768x8 and 2x256)   self.mlp(values) outputs each position's occ, 1-occ
      
        # print(f'values.shape: {values.shape}')
        return values
