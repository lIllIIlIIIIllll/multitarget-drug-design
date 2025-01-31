import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import argparse
import tempfile  # For creating a temporary directory

from molecules.conversion import (
    mols_from_smiles, mol_to_smiles, mols_to_smiles, canonicalize)
from molecules.fragmentation import fragment_iterative, reconstruct
from molecules.properties import add_property
from molecules.structure import (
    add_atom_counts, add_bond_counts, add_ring_counts)
from utils.config import get_dataset_info


class Preprocess:
    def __init__(self, name, n_jobs):
        self.name = name
        self.n_jobs = n_jobs
        self.info = get_dataset_info(name)

    def fetch_dataset(self):
        filename = Path(self.info['filename'])
        url = self.info['url']
        unzip = self.info['unzip']

        # Use a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            print(f"Temporary folder created at: {folder}")

            # Download the file
            os.system(f'wget -P {folder} {url}')
            print(f"File downloaded to: {folder / filename}")

            # Handle unzipping if required
            path = folder / filename
            if unzip is True:
                if ".tar.gz" in self.info['url']:
                    os.system(f'tar xvf {path}.tar.gz -C {folder}')
                elif '.zip' in self.info['url']:
                    os.system(f'unzip {path.with_suffix(".zip")} -d {folder}')
                elif '.gz' in self.info['url']:
                    os.system(f'gunzip {path}.gz')

            source = folder / filename
            dest = Path.cwd() / filename
            shutil.move(source, dest)
            print(f"Moved file to: {dest}")

    def break_into_fragments(self, mol, smi):
        frags = fragment_iterative(mol)

        if len(frags) == 0:
            return smi, np.nan, 0

        if len(frags) == 1:
            return smi, smi, 1

        rec, frags = reconstruct(frags)
        if rec and mol_to_smiles(rec) == smi:
            fragments = mols_to_smiles(frags)
            return smi, " ".join(fragments), len(frags)

        return smi, np.nan, 0

    def read_and_clean_dataset(self):
        filename = Path(self.info['filename'])
        dataset_path = Path.cwd() / filename

        if not dataset_path.exists():
            self.fetch_dataset()

        dataset = pd.read_csv(
            dataset_path,
            index_col=self.info['index_col'])

        if self.info['drop'] != []:
            dataset = dataset.drop(self.info['drop'], axis=1)

        if self.info['name'] == 'ZINC':
            dataset = dataset.replace(r'\n', '', regex=True)
            
        if self.info['name'] == 'ZINCMOSES':
            dataset = dataset.rename(columns={'SMILES': 'smiles'})
        
        if self.info['name'] == 'CHEMBL':
            dataset.columns = ['smiles']

        if self.info['name'] == 'GDB17':
            dataset = dataset.sample(n=self.info['random_sample'])
            dataset.columns = ['smiles']

        if self.info['name'] == 'PCBA':
            cols = dataset.columns.str.startswith('PCBA')
            dataset = dataset.loc[:, ~cols]
            dataset = dataset.drop_duplicates()
            dataset = dataset[~dataset.smiles.str.contains("\\.")]

        if self.info['name'] == 'QM9':
            correct_smiles = pd.read_csv(Path.cwd() / 'gdb9_smiles_correct.csv')
            dataset.smiles = correct_smiles.smiles
            dataset = dataset.sample(frac=1, random_state=42)

        smiles = dataset.smiles.tolist()
        dataset.smiles = [canonicalize(smi, clear_stereo=True) for smi in smiles]
        dataset = dataset[dataset.smiles.notnull()].reset_index(drop=True)

        return dataset

    def add_fragments(self, dataset):
        smiles = dataset.smiles.tolist()
        mols = mols_from_smiles(smiles)
        pjob = Parallel(n_jobs=self.n_jobs, verbose=1)
        fun = delayed(self.break_into_fragments)
        results = pjob(fun(m, s) for m, s in zip(mols, smiles))
        smiles, fragments, lengths = zip(*results)
        dataset["smiles"] = smiles
        dataset["fragments"] = fragments
        dataset["n_fragments"] = lengths

        return dataset

    def save_dataset(self, dataset):
        dataset = dataset[self.info['column_order']]
        testset = dataset[dataset.fragments.notnull()]
        trainset = testset[testset.n_fragments >= self.info['min_length']]
        trainset = trainset[testset.n_fragments <= self.info['max_length']]
        
        trainset.to_csv(Path.cwd() / 'train.smi', index=False)
        dataset.to_csv(Path.cwd() / 'test.smi', index=False)
        print(f"Saved train.smi and test.smi in: {Path.cwd()}")

    def preprocess_dataset(self):
        dataset = self.read_and_clean_dataset()
        dataset = add_atom_counts(dataset, self.info, self.n_jobs)
        dataset = add_bond_counts(dataset, self.info, self.n_jobs)
        dataset = add_ring_counts(dataset, self.info, self.n_jobs)

        for prop in self.info['properties']:
            if prop not in dataset.columns:
                dataset = add_property(dataset, prop, self.n_jobs)

        dataset = self.add_fragments(dataset)

        self.save_dataset(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a dataset.")
    parser.add_argument("--name", type=str, required=True, help="Name of the dataset to preprocess.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs for parallel processing.")
    args = parser.parse_args()

    preprocessor = Preprocess(args.name, args.n_jobs)
    preprocessor.preprocess_dataset()
