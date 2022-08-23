import torch
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

class GNN_EdgeConv(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, k=16, aggr='add',
                 norm=torch.tensor([1., 1., 1., 1., 1.])):
        super(GNN_EdgeConv, self).__init__()

        self.datanorm = nn.Parameter(norm,requires_grad=False)
        
        self.k = k
        start_width = 2 * hidden_dim
        middle_width = 3 * hidden_dim // 2

        #Fully connected network to take input data to higher dimensional space. 
        #This is optional and you can try with and without it
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),            
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU())
        
        convnn1 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.BatchNorm1d(middle_width),
                                nn.Dropout(0.4),
                                nn.ReLU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ReLU()
                                )
        convnn2 = nn.Sequential(nn.Linear(start_width*2, middle_width),
                                nn.BatchNorm1d(middle_width),
                                nn.Dropout(0.4),
                                nn.ReLU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ReLU()
                                )
        
        convnn3 = nn.Sequential(nn.Linear(start_width*3, middle_width),
                                nn.BatchNorm1d(middle_width),
                                nn.Dropout(0.4),
                                nn.ReLU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ReLU()
                                )

        self.edgeconv1 = EdgeConv(nn=convnn1, aggr=aggr)
        self.edgeconv2 = EdgeConv(nn=convnn2, aggr=aggr)
        self.edgeconv3 = EdgeConv(nn=convnn3, aggr=aggr)
                
        self.output = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim*2),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim*2, (hidden_dim)//2),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear((hidden_dim)//2, output_dim)
                                   )
        
    def forward(self, data):
        #print(data.shape)
        data.x = self.datanorm * data.x
        data.x = self.inputnet(data.x)
        
        
        orig_x = data.x.clone()  #To add a residual connection at the end.
        #print("orig_x", orig_x.shape)
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv1.flow,cosine=True))
        #print("Before conv1 data.x shape ",data.x.shape)
        data.x = self.edgeconv1(data.x, data.edge_index)
        #print("after conv1 data.x shape ",data.x.shape)
        data.x = torch.cat([data.x, orig_x], dim=-1) #Residual concatenation 
        #print("after cat data.x shape ",data.x.shape)
        
        res1 = data.x.clone() #To add a residual connection at the end.
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv2.flow, cosine=True))
        #print("Before conv2 data.x shape ",data.x.shape)
        data.x = self.edgeconv2(data.x, data.edge_index)
        #print("after conv2 data.x shape ",data.x.shape)
        data.x = torch.cat([data.x, res1], dim=-1) #Residual concatenation 
        #data = global_max_pool(data.x, data.batch)
        #print("after cat2 data.x shape ",data.x.shape)
        

        res2 = data.x.clone()
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv3.flow,cosine=True))
        #print("Before conv2 data.x shape ",data.x.shape)
        data.x = self.edgeconv3(data.x, data.edge_index)
        #print("after conv2 data.x shape ",data.x.shape)
        data.x = torch.cat([data.x, res2], dim=-1)
        #print("after cat3 data.x shape ",data.x.shape)
        x, batch= data.x, data.batch
        x = global_max_pool(x, batch)
        #print("after pool x shape ",x.shape)
        #data.x = torch.cat([x, ieta, iphi], dim=-1)
        #print(self.output(x))
        
        return self.output(x).squeeze(-1), data.x, data.edge_index
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Edconv = GNN_EdgeConv(input_dim=3,hidden_dim=20,k=10,output_dim=1,norm=torch.tensor([1./50., 1./32., 1./32.]))
        
    def forward(self, data):
        logits = self.Edconv(data)
        #return F.softplus(logits)
        return logits
