# mol_design_toolkit/active_learning.py

from rdkit import Chem
import random

class ActiveLearner:
    def __init__(self, generator, predictor, query_function):
        self.generator = generator
        self.predictor = predictor
        self.query_function = query_function  # Function to obtain true property values
        self.molecules = []
        self.properties = []

    def run(self, num_iterations=5, samples_per_iteration=10, uncertainty_threshold=0.5):
        for iteration in range(num_iterations):
            print(f"Active Learning Iteration {iteration+1}")
            # Generate molecules
            new_molecules = self.generator.generate(num_samples=samples_per_iteration)
            # Filter out invalid molecules
            valid_molecules = [mol for mol in new_molecules if mol is not None]
            if not valid_molecules:
                print("No valid molecules generated.")
                continue
            # Predict properties with uncertainties
            predictions, uncertainties = self.predict_with_uncertainty(valid_molecules)
            # Select molecules with high uncertainty
            selected_indices = [i for i, u in enumerate(uncertainties) if u > uncertainty_threshold]
            selected_molecules = [valid_molecules[i] for i in selected_indices]
            if not selected_molecules:
                print("No molecules with uncertainty above threshold.")
                continue
            # Obtain true properties
            true_properties = self.query_function(selected_molecules)
            # Update the dataset
            self.molecules.extend(selected_molecules)
            self.properties.extend(true_properties)
            # Retrain the predictor with new data
            self.predictor.train(self.molecules, self.properties)
            print(f"Retrained predictor with {len(self.molecules)} samples.")

    def predict_with_uncertainty(self, molecules):
        if self.predictor.model_type == 'gaussian_process':
            predictions, std_dev = self.predictor.predict(molecules)
            uncertainties = std_dev
        else:
            predictions = self.predictor.predict(molecules)
            # Simulate uncertainties for models that don't provide them
            uncertainties = [random.uniform(0, 1) for _ in predictions]
        return predictions, uncertainties