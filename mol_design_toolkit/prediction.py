# mol_design_toolkit/prediction.py

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import joblib
import os

class PropertyPredictor:
    def __init__(self, model_type='random_forest', model_file=None):
        self.model_type = model_type
        self.model_file = model_file
        if model_file and os.path.exists(model_file):
            self.load_model(model_file)
        else:
            self.build_model()

    def build_model(self):
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100)
        elif self.model_type == 'gaussian_process':
            self.model = GaussianProcessRegressor()
        elif self.model_type == 'svm':
            self.model = SVR()
        elif self.model_type == 'neural_network':
            self.model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)
        elif self.model_type == 'xgboost':
            self.model = XGBRegressor(n_estimators=100)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def featurize(self, mol):
        descriptors = []
        descriptors.append(Descriptors.MolWt(mol))
        descriptors.append(Descriptors.MolLogP(mol))
        descriptors.append(Descriptors.NumHDonors(mol))
        descriptors.append(Descriptors.NumHAcceptors(mol))
        descriptors.append(Descriptors.TPSA(mol))
        descriptors.append(Descriptors.NumRotatableBonds(mol))
        descriptors.append(Descriptors.NumAromaticRings(mol))
        descriptors.append(Descriptors.FractionCSP3(mol))
        return np.array(descriptors)

    def train(self, molecules, properties):
        X = np.array([self.featurize(mol) for mol in molecules])
        y = np.array(properties)
        self.model.fit(X, y)

    def predict(self, molecules):
        X = np.array([self.featurize(mol) for mol in molecules])
        if self.model_type == 'gaussian_process':
            predictions, std_dev = self.model.predict(X, return_std=True)
            return predictions, std_dev
        else:
            predictions = self.model.predict(X)
            return predictions

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = joblib.load(file_path)
