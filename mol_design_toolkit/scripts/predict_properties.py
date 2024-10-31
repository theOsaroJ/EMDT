# mol_design_toolkit/scripts/predict_properties.py

import argparse
from mol_design_toolkit import PropertyPredictor, GNNPropertyPredictor
from rdkit import Chem

def main():
    parser = argparse.ArgumentParser(description='Predict properties of molecules using a trained model.')
    parser.add_argument('--model-file', type=str, required=True, help='Path to the trained property prediction model.')
    parser.add_argument('--model-type', type=str, required=True, choices=['random_forest', 'gaussian_process', 'svm', 'neural_network', 'xgboost', 'gnn'], help='Type of prediction model.')
    parser.add_argument('--input-file', type=str, required=True, help='Input file containing SMILES strings.')
    parser.add_argument('--output', type=str, default='predicted_properties.csv', help='Output file to save predictions.')
    args = parser.parse_args()

    # Load molecules
    smiles_list = []
    with open(args.input_file, 'r') as f:
        for line in f:
            smiles = line.strip()
            smiles_list.append(smiles)

    molecules = [Chem.MolFromSmiles(smi) for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]

    if args.model_type == 'gnn':
        predictor = GNNPropertyPredictor(model_file=args.model_file)
        predictions = predictor.predict(molecules)
    else:
        predictor = PropertyPredictor(model_type=args.model_type, model_file=args.model_file)
        predictions = predictor.predict(molecules)

    # Save predictions
    with open(args.output, 'w') as f:
        f.write('smiles,prediction\n')
        for smi, pred in zip(smiles_list, predictions):
            f.write(f"{smi},{pred}\n")
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main()
