# mol_design_toolkit/scripts/train_vae.py

import argparse
from mol_design_toolkit.generation import MoleculeGenerator
import os

def main():
    parser = argparse.ArgumentParser(description='Train a VAE model for molecule generation.')
    parser.add_argument('--data-file', type=str, required=True, help='Path to the training data file containing SMILES strings.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--output-model', type=str, default='models/vae_weights.weights.h5', help='File to save the trained model weights.')
    parser.add_argument('--config-file', type=str, default='models/vae_config.json', help='File to save the model configuration.')
    args = parser.parse_args()

    # Load training data
    with open(args.data_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    # Initialize the generator
    generator = MoleculeGenerator(model_type='VAE')

    # Train the model
    generator.train(smiles_list, epochs=args.epochs, batch_size=args.batch_size, config_file=args.config_file)

    # Save the trained model weights
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    generator.model.save_weights(args.output_model)
    print(f"Trained VAE model weights saved to {args.output_model}")
    print(f"Model configuration saved to {args.config_file}")

if __name__ == '__main__':
    main()
