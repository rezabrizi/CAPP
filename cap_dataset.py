import torch
import numpy as np
import copy
import os
import pandas as pd
from collections import defaultdict 
from torch_geometric.data import Data, InMemoryDataset

class CascadeRegression(InMemoryDataset):
    def __init__(self, root, name, edge_index_path, task, observation, directed):
        assert task in ['classification', 'regression'], "Task must be either 'classification' or 'regression'"
        if task == 'classification':
            assert 0 <= observation <= 3, "Observation time must be between 0 and 3 for classification tasks"

        self.name = name
        self.task = task
        self.observation = observation
        self.directed = directed
        self.edge_index_tensor = self.get_edge_index(edge_index_path)
        super(CascadeRegression, self).__init__(root)
        self.load(self.processed_paths[0])
        

    def get_edge_index (self, path):
        edge_df = pd.read_csv(path, sep = " ", header=None)
        edge_df.columns = ['u', 'v']
        sources = []
        dests = []
        for _, row in edge_df.iterrows():
            sources.append(row[0])
            dests.append(row[1])
            if not self.directed:
                sources.append(row[1])
                dests.append(row[0])
        return torch.tensor([sources, dests], dtype=torch.long)

    @property 
    def raw_dir(self):
        return os.path.join(self.root, "raw", self.task, self.name)
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed", self.task, self.name, str(self.observation))
    
    @property
    def num_features(self):
        return self.data.x.size(1) if self.data.x is not None else 0

    @property
    def raw_file_names(self):
        files = ['cascades', 'node_features']
        return [f'{file}.txt' for file in files]
    
    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def __repr__(self):
        return f'{self.__class__.__name__} \nnetwork: {self.name} \ntask: {self.task} \nobservation-window t={str(self.observation)} \n(Number of graphs: {len(self)}, Number of features [activation, deg cent, eigen cent, btwns cemt]: {self.num_features})'

    def process(self):
        cascades_file_path, node_features_file_path = self.raw_paths
        with open(cascades_file_path, 'r') as file:
            cascades_data = file.readlines()
        
        with open (node_features_file_path, 'r') as file:
            node_features_data = file.readlines()
        
        node_features = defaultdict(list)
        data = []

        for node_feature in node_features_data: 
            current_node_centralities = node_feature.split()
            node = int(current_node_centralities[0])

            func_float = lambda x: float(x)
            centralities = [func_float(centrality) for centrality in current_node_centralities[1:]]
            centralities.append(0)
            centralities = np.array(centralities)
            node_features[node] = centralities
        print("Node Features are done being loaded")

        # Need to go through every cascade and create the Data object
        # depending on if this is a regression or classification task then the type of label is different
        # however, the X will always be the number of nodes time the node features which is going to be 
        for index, cascade in enumerate(cascades_data):
            if (index+1) % 100 == 0:
                print (f"processed {index+1} cascades")
            cascade = cascade.strip().split()
            data_name = cascade[0]
            if self.task == "regression":
                #seeds = list(map(int, cascade[1:-1]))
                #activation_count = float(cascade[-1])
                activations = [node_activation.split(':') for node_activation in cascade[1:]]
                seeds = [int(node) for node, time in activations if int(time) <= self.observation]
                final_activations = [int(node) for node, _ in activations]
            elif self.task == "classification":
                activations = [node_activation.split(':') for node_activation in cascade[1:]]
                seeds = [int(node) for node, time in activations if int(time) <= self.observation]
                final_activations = [int(node) for node, _ in activations]

            seeds = set(seeds)
            X = torch.empty((len(node_features), len(node_features[0]))) 
            for node, features in node_features.items():
                activation_status = 1 if node in seeds else 0
                node_feature = copy.deepcopy(features)
                node_feature[-1] = activation_status
                node_feature = torch.tensor(node_feature)
                X[node] = node_feature
            
            if self.task == "regression":
                y = torch.tensor([len(final_activations)], dtype=torch.float)
            elif self.task == "classification":
                y = torch.tensor([1 if node in final_activations else 0 for node in range(len(node_features))], dtype=torch.long)
            data.append(Data(x=X, y=y, edge_index=self.edge_index_tensor, cascade_name=data_name))

        self.save(data, self.processed_paths[0])
        