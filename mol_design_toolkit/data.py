# mol_design_toolkit/data.py

from rdkit import Chem
import pandas as pd

class MoleculeDataset:
    def __init__(self, file_path=None):
        self.molecules = []
        self.properties = []
        if file_path:
            self.load_dataset(file_path)

    def load_dataset(self, file_path):
        df = pd.read_csv(file_path)
        for idx, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                self.molecules.append(mol)
                self.properties.append(row['property'])

    def get_data(self):
        return self.molecules, self.properties
