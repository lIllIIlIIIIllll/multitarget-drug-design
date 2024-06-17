import time
import numpy as np
import pandas as pd
import itertools
from datetime import datetime

import torch
#from torch.autograd import Variable
from torch.nn import functional as F

from fast_jtnn import *
from molecules.conversion import (
    mols_from_smiles, mols_to_smiles, mol_to_smiles)
from molecules.fragmentation import reconstruct

from utils.config import set_random_seed


def remove_consecutive(fragments):
    return [i for i, _ in itertools.groupby(fragments)]


def generate_molecules(samples, vocab):
    """
    Convert generated samples from indices to molecules and fragments.
    """
    result = []
    num_samples = samples.shape[0]
    valid = np.zeros(num_samples, dtype=bool)

    for idx in range(num_samples):
        frag_smiles = vocab.translate(samples[idx, :])
        frag_smiles = remove_consecutive(frag_smiles)

        if len(frag_smiles) <= 1:
            continue

        try:
            frag_mols = mols_from_smiles(frag_smiles)
            mol, frags = reconstruct(frag_mols)

            if mol is not None:
                smiles = mol_to_smiles(mol)
                num_frags = len(frags)
                frags = " ".join(mols_to_smiles(frags))
                result.append((smiles, frags, num_frags))
                valid[idx] = True # mark is valid
        except Exception:
            continue

    return result, valid


def dump_samples(config, samples, prefix='sampled', postfix='', folder_name='samples'):
    """
    Save generated samples into a CSV file.
    """
    columns = ["smiles", "fragments", "n_fragments"]
    df = pd.DataFrame(samples, columns=columns)
    date = datetime.now().strftime('%Y-%m-%d@%X')
    if postfix == '':
        postfix = date
    #filename = config.path(folder_name) / f"{prefix}{date}_{len(samples)}.csv"
    filename = config.path(folder_name) / f"{prefix}_{postfix}.csv"
    df.to_csv(filename)


# vocab is defined in skipgram.py

class Sampler:
    def __init__(self, config, vocab, model):
        self.config = config
        self.vocab = vocab
        self.model = model

    def sample(self, num_samples, save_results=True, seed=None, postfix='', folder_name='samples'):
        # self.model = self.model.cpu()
        # self.model.eval()
        self.model = self.model.cuda()
        vocab = self.vocab

        hidden_layers = 1
        hidden_size = self.model.hidden_size

        def row_filter(row):
            return (row == vocab.EOS).any()
        
        count = 0
        total_time = 0
        batch_size = 100
        samples, sampled = [], 0 # samples is a list
        valids = []

        max_length = self.config.get('max_length')
        temperature = self.config.get('temperature')

        seed = set_random_seed(seed)
        self.config.set('sampling_seed', seed)
        print("Sampling seed:", self.config.get('sampling_seed'))

        with torch.no_grad():
            while len(samples) < num_samples:
                start = time.time()

                # generete samples using JtVAE sampler
                sample = self.model.sample_prior() # generate on at a time
                # print('generated sample:', sample)
                # samples += sample
                samples.append(sample)
                valid = self.generate_valid(sample)
                valids.append(valid)

                end = time.time() - start
                total_time += end
                
                # if new samples are added, set count to 0, otherwise, increase count, if tried many times but failed to grow, give up.
                if len(samples) > sampled:
                    sampled = len(samples)
                    count = 0 
                else: 
                    count += 1

                if len(samples) % 1000 < 50:
                    elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
                    print(f'Sampled {len(samples)} molecules. '
                          f'Time elapsed: {elapsed}')

                if count >= 10000:
                    break 

        if save_results:
            dump_samples(self.config, samples, prefix='generated_from_random', postfix=postfix, folder_name=folder_name )

        elapsed = time.strftime("%H:%M:%S", time.gmtime(total_time))
        print(f'Done. Total time elapsed: {elapsed}.')

        set_random_seed(self.config.get('random_seed'))
        # print('sample list:', samples)
        return samples, valids
    
    
    def update(self, save_to, sample, running_seqs, step):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at step position
        running_latest[:, step] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

    def generate_valid(self, sample):
        valid = 0
        if sample is not None:
            valid = True

        return valid
    
    
    # def encode_z(self, inputs, lengths):
    #     with torch.no_grad():
    #         embeddings = self.model.embedder(inputs)
    #         z, mu, logvar = self.model.encoder(inputs, embeddings, lengths)
    #     return z, mu, logvar
    
    def sample_from_z(self, z_tree, z_mol, save_results=True, seed=None):
        """
        Given latent representation z, generate samples.
        z: tensor, batch (or num_samples) x latent_size.
        """
        self.model = self.model.cuda()
        self.model.eval()
        vocab = self.vocab

        hidden_layers = 1
        hidden_size = self.model.hidden_size

        def row_filter(row):
            return (row == vocab.EOS).any()
        
        total_time = 0
        samples = [] # samples is a list

        max_length = self.config.get('max_length')
        temperature = self.config.get('temperature')
        
        batch_size = z_tree.shape[0] # number of samples to be generated

        seed = set_random_seed()
        self.config.set('sampling_seed', seed)
        print("Sampling seed:", self.config.get('sampling_seed'))

        with torch.no_grad():
            #while len(samples) < num_samples:
            start = time.time()

            # generete samples using JtVAE sampler
            sample = self.model.sample_prior_z(z_tree, z_mol) # generate on at a time

        end = time.time() - start
        total_time += end

        if save_results:
            dump_samples(self.config, samples, prefix='generated_from_z')

        elapsed = time.strftime("%H:%M:%S", time.gmtime(total_time))
        print(f'Done. Total time elapsed: {elapsed}.')

        set_random_seed(self.config.get('random_seed'))

        return sample
