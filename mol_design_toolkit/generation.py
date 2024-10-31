# mol_design_toolkit/generation.py

import numpy as np
from rdkit import Chem
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import os
import json

class Sampling(layers.Layer):
    """Sampling layer for VAE."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch_size, latent_dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAE(models.Model):
    """Variational Autoencoder."""

    def __init__(self, input_dim, latent_dim=128, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(input_dim, latent_dim)
        self.decoder = self.build_decoder(latent_dim, input_dim)

    def build_encoder(self, input_dim, latent_dim):
        inputs = layers.Input(shape=(input_dim,), name='encoder_input')
        x = layers.Dense(256, activation='relu')(inputs)
        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder

    def build_decoder(self, latent_dim, output_dim):
        latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
        x = layers.Dense(256, activation='relu')(latent_inputs)
        outputs = layers.Dense(output_dim, activation='sigmoid')(x)
        decoder = models.Model(latent_inputs, outputs, name='decoder')
        return decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Compute KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

class MoleculeGenerator:
    def __init__(self, model_type='VAE', model_file=None, config_file=None):
        self.model_type = model_type
        self.model_file = model_file
        self.config_file = config_file
        self.model = None
        self.input_dim = None
        self.latent_dim = None
        self.max_length = None
        self.char_to_index = None
        self.index_to_char = None

    def train(self, smiles_list, epochs=50, batch_size=32, config_file='models/vae_config.json'):
        if self.model_type != 'VAE':
            raise NotImplementedError("Currently, only VAE is implemented.")

        input_data = self.preprocess_smiles(smiles_list)
        self.input_dim = input_data.shape[1]
        self.latent_dim = 128  # You can adjust this as needed

        # Build the model
        self.model = VAE(input_dim=self.input_dim, latent_dim=self.latent_dim)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

        # Train the model
        self.model.fit(input_data, input_data, epochs=epochs, batch_size=batch_size)

        # Save configuration
        config = {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'max_length': self.max_length,
            'char_to_index': self.char_to_index,
            'index_to_char': self.index_to_char
        }
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config, f)

    def preprocess_smiles(self, smiles_list):
        # Tokenize SMILES strings
        tokens = [list(smiles) for smiles in smiles_list]
        # Build character dictionaries
        all_chars = set(char for smiles in tokens for char in smiles)
        self.char_to_index = {char: idx for idx, char in enumerate(sorted(all_chars))}
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        self.max_length = max(len(smiles) for smiles in tokens)

        # One-hot encode the SMILES strings
        input_data = np.zeros((len(tokens), self.max_length, len(self.char_to_index)), dtype='float32')
        for i, smiles in enumerate(tokens):
            for t, char in enumerate(smiles):
                input_data[i, t, self.char_to_index[char]] = 1.0

        # Flatten the input data
        input_data = input_data.reshape(len(tokens), -1)
        return input_data

    def load_model(self, model_file, config_file):
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.input_dim = config['input_dim']
        self.latent_dim = config['latent_dim']
        self.max_length = config['max_length']
        self.char_to_index = {char: int(idx) for char, idx in config['char_to_index'].items()}
        self.index_to_char = {int(idx): char for idx, char in config['index_to_char'].items()}

        # Build the model
        self.model = VAE(input_dim=self.input_dim, latent_dim=self.latent_dim)
        # Load weights
        self.model.load_weights(model_file)

    def generate(self, num_samples=1):
        generated_molecules = []
        for _ in range(num_samples):
            latent_vector = np.random.normal(size=(1, self.latent_dim))
            decoded = self.model.decoder.predict(latent_vector)
            smiles = self.decode_smiles(decoded[0])
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                generated_molecules.append(mol)
        return generated_molecules

    def decode_smiles(self, vector):
        # Reshape the vector to (max_length, num_chars)
        vector = vector.reshape((self.max_length, len(self.char_to_index)))
        # Get the indices of the max probability characters
        indices = np.argmax(vector, axis=1)
        # Convert indices to characters
        chars = [self.index_to_char.get(idx, '') for idx in indices]
        # Join characters to form SMILES string
        smiles = ''.join(chars)
        return smiles.strip()