import os
import os.path as osp
import math

import numpy as np
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch.utils.checkpoint import checkpoint
from torch_cluster import knn_graph

from torch_geometric.nn import EdgeConv, NNConv
#from torch_geometric.nn.pool.edge_pool import EdgePooling

from torch_geometric.utils import normalized_cut
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.nn import (graclus, max_pool, max_pool_x,
                                global_mean_pool, global_max_pool,
                                global_add_pool)

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

class DynamicReductionNetwork(nn.Module):
    # This model iteratively contracts nearest neighbour graphs 
    # until there is one output node.
    # The latent space trained to group useful features at each level
    # of aggregration.
    # This allows single quantities to be regressed from complex point counts
    # in a location and orientation invariant way.
    # One encoding layer is used to abstract away the input features.
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, k=16, aggr='add',
                 #norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.])):
                 norm=torch.tensor([1., 1., 1., 1., 1.])):
        super(DynamicReductionNetwork, self).__init__()

        self.datanorm = nn.Parameter(norm,requires_grad=False)
        
        self.k = k
        start_width = 2 * hidden_dim
        middle_width = 3 * hidden_dim // 2

        
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),            
            nn.LeakyReLU(negative_slope=0.4),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LeakyReLU(negative_slope=0.4),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.4),
        )
        
        
        convnn1 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.LeakyReLU(negative_slope=0.4),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.LeakyReLU(negative_slope=0.4)
                                )
        convnn2 = nn.Sequential(nn.Linear(start_width*2, middle_width),
                                nn.LeakyReLU(negative_slope=0.4),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.LeakyReLU(negative_slope=0.4)
                                )
        
        convnn3 = nn.Sequential(nn.Linear(start_width*3, middle_width),
                                nn.LeakyReLU(negative_slope=0.4),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.LeakyReLU(negative_slope=0.4)
                                )
        convnn4 = nn.Sequential(nn.Linear(start_width*4, middle_width),
                                nn.LeakyReLU(negative_slope=0.4),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.LeakyReLU(negative_slope=0.4)
                                )
                
        self.edgeconv1 = EdgeConv(nn=convnn1, aggr=aggr)
        self.edgeconv2 = EdgeConv(nn=convnn2, aggr=aggr)
        self.edgeconv3 = EdgeConv(nn=convnn3, aggr=aggr)
        self.edgeconv4 = EdgeConv(nn=convnn4, aggr=aggr)
        
        self.shortcut = nn.Linear(input_dim,hidden_dim, bias=False)
        #nn.Conv1d(in_channels, out_channels, kernel_size, stride=1,)
        
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.LeakyReLU(negative_slope=0.4),
                                    #nn.Softplus(),
                                    nn.Linear(hidden_dim, (hidden_dim)//2),
                                    nn.LeakyReLU(negative_slope=0.4),
                                    #nn.Softplus(),
#                                    nn.Linear(hidden_dim//2, hidden_dim//2),#added
 #                                   nn.ELU(),
                                    #nn.Softplus(),
                                    nn.Linear((hidden_dim)//2, output_dim)
                                   )
        
        
    def forward(self, data):        
        data.x = self.datanorm * data.x
        #print(data.pscieta.shape)
        #ieta = data.pscieta.clone()
        #iphi = data.psciphi.clone()
        data.x = self.inputnet(data.x)
        orig_x = data.x.clone()
        #print("orig_x",orig_x.shape)
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv1.flow))
        data.x = self.edgeconv1(data.x, data.edge_index)
        #print("data_x",data.x.shape)
        data.x = torch.cat([data.x, orig_x], dim=-1)
        #print("torch.cat([data.x, orig_x], dim=-1",data.x.shape)
        #print("after cat data_x",data.x.shape)
        
        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data)
        
        res1 = data.x.clone()
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv2.flow))
        data.x = self.edgeconv2(data.x, data.edge_index)
        data.x = torch.cat([data.x, res1], dim=-1)
        #print("torch.cat([data.x, orig_x], dim=-1",data.x.shape)
        
        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data)
        
        res2 = data.x.clone()
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv3.flow))
        data.x = self.edgeconv3(data.x, data.edge_index)
        data.x = torch.cat([data.x, res2], dim=-1)
        #print("torch.cat([data.x, orig_x], dim=-1",data.x.shape)     
        
        
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv4.flow))
        data.x = self.edgeconv4(data.x, data.edge_index)
        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_max_pool(x, batch)
        #print(x.shape)
        #print(data.pscieta)
        #print(data.iphi.size())
        #x = torch.cat([x, ieta, iphi], dim=-1)
#        print(self.output(x))
        return self.output(x).squeeze(-1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drn = DynamicReductionNetwork(input_dim=3,hidden_dim=20,k=16,output_dim=1,norm=torch.tensor([1./50., 1./32., 1./32.]))

    def forward(self, data):
        logits = self.drn(data)
        #return F.softplus(logits)
        return logits
