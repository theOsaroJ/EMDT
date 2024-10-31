# mol_design_toolkit/gnn_models.py

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
import numpy as np
import os

class GNNModel(nn.Module):
    def __init__(self, num_node_features=1, hidden_dim=64):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

class GNNPropertyPredictor:
    def __init__(self, model_file=None):
        self.model = GNNModel()
        self.model_file = model_file
        if model_file and os.path.exists(model_file):
            self.load_model(model_file)
        else:
            pass  # Model will be trained later

    def mol_to_graph_data(self, mol):
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([atom.GetAtomicNum()])
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = self.get_edge_index(mol)
        data = Data(x=x, edge_index=edge_index)
        return data

    def get_edge_index(self, mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        edge_indices = np.array(adjacency.nonzero())
        return torch.tensor(edge_indices, dtype=torch.long)

    def train(self, molecules, properties, epochs=50, batch_size=32):
        dataset = []
        for mol, prop in zip(molecules, properties):
            data = self.mol_to_graph_data(mol)
            data.y = torch.tensor([prop], dtype=torch.float)
            dataset.append(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                pred = self.model(batch.x, batch.edge_index, batch.batch).squeeze()
                loss = criterion(pred, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    def predict(self, molecules):
        self.model.eval()
        dataset = [self.mol_to_graph_data(mol) for mol in molecules]
        loader = DataLoader(dataset, batch_size=1)
        predictions = []
        with torch.no_grad():
            for batch in loader:
                pred = self.model(batch.x, batch.edge_index, batch.batch).item()
                predictions.append(pred)
        return predictions

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
