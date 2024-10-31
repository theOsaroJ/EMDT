# setup.py

from setuptools import setup, find_packages

setup(
    name='mol-design-toolkit',
    version='1.0.0',
    description='Enhanced Automated Molecular Design and Simulation Toolkit',
    author='Etinosa Osaro',
    affiliation='University of Notre Dame @Colon Lab',
    author_email='eosaro@nd.edu',
    url='https://github.com/theOsaroJ/mol-design-toolkit',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'torch',
        'torchvision',
        'torch_geometric',
        'matplotlib',
        'seaborn',
        'py3Dmol',
        'tqdm',
        'ipywidgets',
        'joblib',
        'xgboost',
        # Note: RDKit should be installed via conda or pip
    ],
    entry_points={
        'console_scripts': [
            'generate-molecules=mol_design_toolkit.scripts.generate_molecules:main',
            'predict-properties=mol_design_toolkit.scripts.predict_properties:main',
            'train-model=mol_design_toolkit.scripts.train_model:main',
            'train-vae=mol_design_toolkit.scripts.train_vae:main',
            'train-gan=mol_design_toolkit.scripts.train_gan:main',
            'train-rnn=mol_design_toolkit.scripts.train_rnn:main',
            'run-active-learning=mol_design_toolkit.scripts.run_active_learning:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
