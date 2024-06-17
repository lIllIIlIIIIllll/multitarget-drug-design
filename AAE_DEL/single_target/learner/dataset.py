import time
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader

from aae_utils.utils import CharVocab, string2tensor, property2tensor
from utils.filesystem import load_dataset


# class DataCollator:
#     def __init__(self, vocab):
#         self.vocab = vocab
#
#     def merge(self, sequences):
#         '''
#         Sentence to indices, pad with zeros
#         '''
#         # print('before:')
#         # print(sequences)
#         # sequences = sorted(sequences, key=len, reverse=True)# YL: choose not to sort it, because I need properties in same order
#         # print('after:')
#         # print(sequences)
#         lengths = [len(seq) for seq in sequences]
#         padded_seqs = np.full((len(sequences), max(lengths)), self.vocab.PAD)
#         for i, seq in enumerate(sequences):
#             end = lengths[i]
#             padded_seqs[i, :end] = seq[:end]
#         print(padded_seqs)
#         return torch.LongTensor(padded_seqs), lengths
#
#     def __call__(self, data):
#         # seperate source and target sequences
#         src_seqs, tgt_seqs, properties = zip(*data)  #
#
#         # merge sequences (from tuple of 1D tensor to 2D tensor)
#         src_seqs, src_lengths = self.merge(src_seqs)
#         tgt_seqs, tgt_lengths = self.merge(tgt_seqs)
#         properties = torch.tensor(properties, dtype=torch.float)
#         return src_seqs, tgt_seqs, src_lengths, properties


class SMILESDataset(Dataset):
    def __init__(self, config, kind='train', data=None):
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
        self.device = 'cuda' if self.config.get('use_gpu') and torch.cuda.is_available() else 'cpu'


    def get_data(self):
        return self.data

    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return len(self.data)

    def get_loader(self, shuffle=True):
        def aae_get_collate_fn():
            # global collate
            def collate(data):
                prps = [p[1:] for p in data]
                props = np.array(prps, dtype=float)
                properties = torch.tensor(props, dtype=torch.float, device=self.device)
                tensors = [string2tensor(string[0], self.vocab) for string in data]
                lengths = torch.tensor([len(t) for t in tensors], dtype=torch.long, device=self.device)
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
                            collate_fn=collator,
                            # Advised to speed up training
                            # num_workers>0 fails on Windows
                            num_workers=0,
                            # pin_memory=True
                            )
        end = time.time() - start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
        print(f'Data loaded. Size: {len(self.data)}. '
              f'Time elapsed: {elapsed}.')
        return loader

    def change_data(self, data):
        self.data = data
        self.vocab = CharVocab(self.data['smiles'])
