# mol_design_toolkit/scripts/generate_molecules.py

import argparse
from mol_design_toolkit.generation import MoleculeGenerator
from rdkit import Chem

def main():
    parser = argparse.ArgumentParser(description='Generate new molecules using a trained model.')
    parser.add_argument('--model-file', type=str, required=True, help='Path to the trained generative model weights.')
    parser.add_argument('--config-file', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--model-type', type=str, required=True, choices=['VAE'], help='Type of generative model.')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of molecules to generate.')
    parser.add_argument('--output', type=str, default='generated_molecules.smi', help='Output file to save generated SMILES.')
    args = parser.parse_args()

    # Initialize the generator
    generator = MoleculeGenerator(model_type=args.model_type)

    # Load the model and configuration
    generator.load_model(args.model_file, args.config_file)

    # Generate molecules
    generated_molecules = generator.generate(num_samples=args.num_samples)

    with open(args.output, 'w') as f:
        for mol in generated_molecules:
            smiles = Chem.MolToSmiles(mol)
            f.write(smiles + '\n')
    print(f"Generated molecules saved to {args.output}")

if __name__ == '__main__':
    main()
