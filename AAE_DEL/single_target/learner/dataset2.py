import time
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader

from aae_utils.utils import CharVocab, string2tensor, property2tensor
from utils.filesystem import load_dataset
from .skipgram import Vocab

from utils.config import Config


class SMILESDataset(Dataset):
    def __init__(self, config: Config, kind='train', data=None):

        self.config = config
        self.fieldnames = ['smiles', 'SAS', 'logP', 'LPA1']

        if kind != 'given':
            self.data = load_dataset(config, kind=kind)
        else:
            self.data = data

        '''Temp Implementation (A1)'''
        subsets = config.get('subsets')
        if subsets > 1:
            end = min((config.get('batch_size') * subsets), len(self.data))
            self.data = self.data.loc[:end]
        '''End Point'''

        self.data.reset_index(drop=True, inplace=True)
        self.data.drop(columns=['mr'], inplace=True)
        self.vocab = CharVocab(self.data['smiles'])

    def get_data(self):
        return self.data

    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return len(self.data)

    def get_loader(self, shuffle=True):
        def aae_get_collate_fn():
            def collate(data):
                properties = torch.tensor([p[1:] for p in data], dtype=torch.float)
                tensors = [string2tensor(string[0], self.vocab) for string in data]
                lengths = torch.tensor([len(t) for t in tensors], dtype=torch.long)
                encoder_inputs = pad_sequence(tensors,
                                              batch_first=True,
                                              padding_value=self.vocab.pad)
                encoder_input_lengths = lengths - 2

                decoder_inputs = pad_sequence([t[:-1] for t in tensors],
                                              batch_first=True,
                                              padding_value=self.vocab.pad)
                decoder_input_lengths = lengths - 1

                decoder_targets = pad_sequence([t[1:] for t in tensors],
                                               batch_first=True,
                                               padding_value=self.vocab.pad)
                decoder_target_lengths = lengths - 1

                return (encoder_inputs, encoder_input_lengths), \
                       (decoder_inputs, decoder_input_lengths), \
                       (decoder_targets, decoder_target_lengths), properties

            return collate

        start = time.time()
        collator = aae_get_collate_fn()
        loader_data = self.data.loc[:, self.fieldnames].to_numpy()
        loader = DataLoader(dataset=loader_data,
                            batch_size=self.config.get('batch_size'),
                            shuffle=shuffle,
                            collate_fn=collator)
        end = time.time() - start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
        print(f'Data loaded. Size: {len(self.data)}. '
              f'Time elapsed: {elapsed}.')
        return loader

    def change_data(self, data):
        self.data = data
        self.vocab = CharVocab(self.data['smiles'])
