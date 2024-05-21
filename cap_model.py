import torch
from torch_geometric.nn import GATConv, LayerNorm
import torch.nn.functional as F
import torch.nn as nn

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        # input dim is 4
        super(GAT, self).__init__()
        # mlp for encoding
        self.encoder = nn.Linear(input_dim, hidden_dim)
        # hidden dim is 64
        self.hidden_gat_layers = nn.ModuleList(
            [GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=0.2) for _ in range(num_layers)])
        
        self.norm_layers = nn.ModuleList(
            [LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        
        # mlp for decoding
        # output dim is 1
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):   
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # encode
        #print (f"x before encoding: shape: {x.shape}   x: {x}")
        x = self.encoder(x)
        #print (f"x after encoding: shape: {x.shape}   x: {x}")
        # x = n * 64 
        for i in range(len(self.hidden_gat_layers)):
            x_in = x
            x = self.hidden_gat_layers[i](x, edge_index)
            x = self.norm_layers[i](x)
            x = F.relu(x)
            x = x + x_in 


        #print (f"x after Last GAT Layer: shape: {x.shape}   x: {x}")
        # decode
        x = self.decoder(x)
        #print (f"x after decoding: shape: {x.shape}   x: {x}")
        # x = n * 1
        x = torch.sigmoid(x)
        #print(f"x after sigmoid: shape: {x.shape}   x: {x}")
        # x = n * 1 
        x = x.sum(dim=0)
        #print(f"x after sum: shape: {x.shape}   x: {x}")
        return x