# mol_design_toolkit/scripts/train_model.py

import argparse
from mol_design_toolkit import PropertyPredictor, GNNPropertyPredictor, MoleculeDataset
import os

def main():
    parser = argparse.ArgumentParser(description='Train a property prediction model.')
    parser.add_argument('--model-type', type=str, required=True, choices=['random_forest', 'gaussian_process', 'svm', 'neural_network', 'xgboost', 'gnn'], help='Type of model to train.')
    parser.add_argument('--training-data', type=str, required=True, help='CSV file containing SMILES and property values.')
    parser.add_argument('--output-model', type=str, required=True, help='File to save the trained model.')
    args = parser.parse_args()

    # Load dataset
    dataset = MoleculeDataset(file_path=args.training_data)
    molecules, properties = dataset.get_data()

    # Train model
    if args.model_type == 'gnn':
        predictor = GNNPropertyPredictor()
        predictor.train(molecules, properties, epochs=50, batch_size=32)
        os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
        predictor.save_model(args.output_model)
    else:
        predictor = PropertyPredictor(model_type=args.model_type)
        predictor.train(molecules, properties)
        os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
        predictor.save_model(args.output_model)
    print(f"Model saved to {args.output_model}")

if __name__ == '__main__':
    main()