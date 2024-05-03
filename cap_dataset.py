import torch
import os
import pandas as pd
from collections import defaultdict 
from torch_geometric.data import Data, InMemoryDataset

class CascadeRegression(InMemoryDataset):
    def __init__(self, root, name, edge_index_path, task, observation):
        assert task in ['classification', 'regression'], "Task must be either 'classification' or 'regression'"
        if task == 'classification':
            assert 0 <= observation <= 3, "Observation time must be between 0 and 3 for classification tasks"

        self.name = name
        self.task = task
        self.observation = observation
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
        return f'{self.__class__.__name__} network: {self.name} task: {self.task} observation-window t={str(self.observation)} (Number of graphs: {len(self)}, Number of features [activation, deg cent, eigen cent, btwns cemt]: {self.num_features})'

    def process(self):
        cascades_file_path, node_features_file_path = self.raw_paths
        with open(cascades_file_path, 'r') as file:
            cascades_data = file.readlines()
        
        with open (node_features_file_path, 'r') as file:
            node_features_data = file.readlines()
        
        node_features = defaultdict(list)
        data = []

        for node_feature in node_features_data: 
            node, degree_centrality, eigenvector_centrality, betweeness_centrality = node_feature.split()
            node = int(node)   
            centralities = [float(degree_centrality), float(eigenvector_centrality), float(betweeness_centrality)]
            node_features[node] = centralities

        # Need to go through every cascade and create the Data object
        # depending on if this is a regression or classification task then the type of label is different
        # however, the X will always be the number of nodes time the node features which is going to be 
        for cascade in cascades_data:
            cascade = cascade.strip().split()
            if self.task == "regression":
                seeds = list(map(int, cascade[1:-1]))
                activation_count = float(cascade[-1])  # This might be used for labels or further processing
            elif self.task == "classification":
                activations = [node_activation.split(':') for node_activation in cascade[1:]]
                seeds = [int(node) for node, time in activations if int(time) <= self.observation]
                final_activations = [int(node) for node, _ in activations]

            seeds = set(seeds)
            X = torch.empty((len(node_features), len(node_features[0]) + 1)) 
            for node, features in node_features.items():
                activation_status = 1 if node in seeds else 0
                node_row = torch.tensor([activation_status] + features)
                X[node] = node_row
            
            if self.task == "regression":
                y = torch.tensor([activation_count], dtype=torch.float)
            elif self.task == "classification":
                y = torch.tensor([1 if node in seeds else 0 for node in range(len(node_features))], dtype=torch.long)
            data.append(Data(x=X, y=y, edge_index=self.edge_index_tensor))

        self.save(data, self.processed_paths[0])
        