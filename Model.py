
import numpy as np
import torch

import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_, calculate_gain

from torch_geometric.utils import scatter

from Layers import GATRoostLayer, Simple_linear, WeightedAttentionPooling, WeightedAttentionPooling_comp, softmax_weights


class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the
    Roost Model.
    """

    def __init__(self, input_dim, n_graphs, elem_heads, internal_elem_dim, g_elem_dim, f_elem_dim, 
                 comp_heads, g_comp_dim, f_comp_dim, negative_slope=0.2,bias=False):
        
        super().__init__()

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.project_fea = nn.Linear(input_dim, internal_elem_dim - 1,bias=False)

        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            [
                GATRoostLayer(internal_elem_dim, internal_elem_dim, g_elem_dim, f_elem_dim, elem_heads, negative_slope, bias)             
                for i in range(n_graphs)
            ]
        )

        # define a global pooling function for materials
        self.comp_pool = nn.ModuleList(
            [
                WeightedAttentionPooling_comp(
                    gate_nn=Simple_linear(internal_elem_dim, 1, f_comp_dim, negative_slope,bias),
                    message_nn=Simple_linear(internal_elem_dim, internal_elem_dim, g_comp_dim, negative_slope,bias),
                )
                for _ in range(comp_heads)
            ]
        )

    def reset_parameters(self):
        gain=calculate_gain('leaky_relu', self.negative_slope)
        xavier_uniform_(self.project_fea.weight, gain=gain)
        for graph in self.graphs:
            graph.reset_parameters()
        for head in self.comp_pool:
            head.reset_parameters()

    def forward(self, x, edge_index, pos, batch_index=None):
        """
        """
        # embed the original features into a trainable embedding space
        weights=pos
        # constructing internal representations of the elements
        x = self.project_fea(x)
        x = torch.cat([x,weights.unsqueeze(1)],dim=1)

        # apply the message passing functions
        for graph_func in self.graphs:
            x = graph_func(x,edge_index,pos)

        # generate crystal features by pooling the elemental features
        head_fea = []
        if batch_index is not None:
            for attnhead in self.comp_pool:
                head_fea.append(
                    attnhead(x, edge_index, pos, batch_index)
                )
        else:
            for attnhead in self.comp_pool:
                head_fea.append(
                    attnhead(x, edge_index, pos)
                )
                
        return torch.mean(torch.stack(head_fea), dim=0)
    

    def __repr__(self):
        return self.__class__.__name__
    

class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network (copied from Roost Repository)
    """

    def __init__(self, internal_elem_dim, output_dim, hidden_layer_dims, batchnorm=False, negative_slope=0.2):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super().__init__()

        dims = [internal_elem_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False)
                if (dims[i] != dims[i + 1])
                else nn.Identity()
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([nn.LeakyReLU(negative_slope) for _ in range(len(dims) - 1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, res_fc, act in zip(self.fcs, self.bns, self.res_fcs, self.acts):
            x = act(bn(fc(x))) + res_fc(x)

        return  self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__
    

# add the possibility to use classification (and batchnorm)

# add the possibility to use classification (and batchnorm)

class Roost(nn.Module):
    """
    Roost model (copied from Roost Repository)
    """

    def __init__(self, input_dim, output_dim, hidden_layer_dims, 
                 n_graphs, elem_heads, internal_elem_dim, g_elem_dim, f_elem_dim, 
                 comp_heads, g_comp_dim, f_comp_dim,
                 batchnorm=True, negative_slope=0.2):
        """
        Inputs
        ----------
        input_dim: int, initial size of the element embeddings
        output_dim: int, dimensinality of the target
        hidden_layer_dims: list(int), list of sizes of layers in the residual network
        n_graphs: int, number of graph layers
        elem_heads: int, number of attention heads for the element attention
        interanl_elem_dim: int, size of the element embeddings inside the model
        g_elem_dim: int, size of hidden layer in g_func in elemental graph layers
        f_elem_dim: int, size of hidden layer in f_func in elemental graph layers
        comp_heads: int, number of attention heads for the composition attention
        g_comp_dim: int, size of hidden layer in g_func in composition graph layers
        f_comp_dim: int, size of hidden layer in f_func in composition graph layers
        batchnorm: bool, whether to use batchnorm in the residual network
        negative_slope: float, negative slope for leaky relu

        """
        super().__init__()

        self.n_graphs = n_graphs
        self.negative_slope = negative_slope
        self.comp_heads = comp_heads
        self.internal_elem_dim = internal_elem_dim

        self.material_nn = DescriptorNetwork(input_dim, n_graphs, elem_heads, internal_elem_dim, g_elem_dim, f_elem_dim, 
                 comp_heads, g_comp_dim, f_comp_dim, negative_slope=0.2,bias=False)
        
        self.resnet = ResidualNetwork(internal_elem_dim, output_dim, hidden_layer_dims,
                                      batchnorm=batchnorm, negative_slope=negative_slope)
        self.reset_parameters()
        
    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x, edge_index, pos, batch_index=None):
        
        if batch_index is not None:
            x = self.material_nn(x, edge_index, pos, batch_index)
        else:
            x = self.material_nn(x, edge_index, pos)
            
        x = self.resnet(x)
        if(x.dim()==2):
            x=x.squeeze(-1)
        return x

    def __repr__(self):
        return self.__class__.__name__