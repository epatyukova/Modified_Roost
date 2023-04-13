# Modified_Roost
Different versions of Roost model re-written with pytorch geometric and pytorch lightning

Implemented:
(1) Standard Roost for regression
(2) Additionally in Layers.py you can find implementation of GATConv and GATv2Conv with weighted attention (weights are encoded as pos argument of Data datatype from pytorch geometric)

Planned:
(1) Roost for node classification
(2) Roost with learnable edge features
