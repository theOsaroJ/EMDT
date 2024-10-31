# mol_design_toolkit/visualization.py

from rdkit import Chem
from rdkit.Chem import Draw
import py3Dmol

class MoleculeVisualizer:
    def __init__(self):
        pass

    def draw_2d(self, mol):
        img = Draw.MolToImage(mol)
        img.show()

    def draw_3d(self, mol):
        mol_block = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=400, height=400)
        viewer.addModel(mol_block, 'sdf')
        viewer.setStyle({'stick': {}})
        viewer.setBackgroundColor('0xeeeeee')
        viewer.zoomTo()
        return viewer.show()