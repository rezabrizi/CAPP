import torch
from torch_geometric.nn import GATConv, global_add_pool
import torch.nn.functional as F
import torch.nn as nn

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        # input dim is 4
        super(GAT, self).__init__()
        # mlp for encoding
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        # hidden dim is 64
        self.hidden_gat_layers = nn.ModuleList(
            [GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=0.2) for _ in range(num_layers)])
        # mlp for decoding
        # output dim is 1
        self.mlp2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):   
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # encode
        x = self.mlp1(x)
        # x = n * 64 
        for i, layer in enumerate(self.hidden_gat_layers): 
            x_in = x
            x = layer(x, edge_index)
            x = x + x_in 
    
        # decode
        x = self.mlp2(x)
        # x = n * 1
        x = torch.sigmoid(x)
        # x = n * 1 
        print(x.shape)
        x = x.sum(dim=0)
        print(x.shape)

        return x