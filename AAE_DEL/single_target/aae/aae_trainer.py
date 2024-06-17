import numpy as np
import torch

import pandas as pd

from time import time
from torch import nn, optim

from aae.aae_model import AAE

"""
Helper Functions
"""


def print_results(epoch, losses, duration, total_epochs):
    # Print Results
    print(f"Epoch - {epoch}; Run Time - {duration}s")
    print(f"Auto Encoder Loss: {losses['autoencoder']:.5}")
    print(f"Generator Loss: {losses['generator']:.5}")
    print(f"Discriminator Loss: {losses['discriminator']:.5}")
    print(f"MLP Loss: {losses['property_predictor']:.5}")
    print(f'Percent Completion: {((epoch / total_epochs) * 100):.5}')
    print()


def init_optim(model, lr):
    return {
        'autoencoder': optim.AdamW(
            list(model.encoder.parameters()) +
            list(model.decoder.parameters()), lr=lr
        ),
        'discriminator': optim.AdamW(
            model.discrim.parameters(), lr=lr
        ),
        'property_predictor': optim.AdamW(
            model.mlp.parameters(), lr=lr
        )
    }


class AAE_Trainer:
    def __init__(self, config, vocab):
        super(AAE_Trainer, self).__init__()

        # Save Config
        self.config = config

        # Init Vocab
        self.vocab = vocab

        # Init the model
        self.model = AAE(config, vocab=self.vocab)

        # Training Parameters
        self.lr = config.get('optim_lr')
        self.seed = config.get('random_seed')
        self.batch_size = config.get('batch_size')
        self.num_epochs = config.get('num_epochs')
        self.sched_step_size = config.get('sched_step_size')
        self.sched_gamma = config.get('sched_gamma')
        self.use_scheduler = config.get('use_scheduler')
        self.use_gpu = config.get('use_gpu')
        # torch.multiprocessing.set_start_method('spawn')
        self.device = 'cuda' if (self.use_gpu and torch.cuda.is_available()) else 'cpu'

        # Training Tools
        self.criterion = {
            'autoencoder': nn.CrossEntropyLoss(),
            'discriminator': nn.BCEWithLogitsLoss(),
            'property_predictor': nn.MSELoss()
        }
        self.optimizers = None

    def train_epoch(self, epoch, model, loader, keys, criterions, optimizers=None):
        if optimizers is None:
            # Set to Evaluation Mode
            model.eval()
        else:
            # Set to Training Mode
            model.train()

        # Info stored for presentation/ report purposes
        # epoch_info = {key: [0] for key in keys}
        # epoch_info = {key: 0 for key in keys[2:-1]}
        # Info stored for calculation purposes
        # losses = {key: 0 for key in keys[2:]}

        for idx, (encoder_inputs, decoder_inputs, decoder_targets, properties) in enumerate(loader):
            # Info stored for presentation/ report purposes
            # epoch_info['idx'].append(idx)
            # epoch_info['epoch'].append(epoch)

            ''' PREP DATA '''
            # encoder_inputs = (data.to(self.device) for data in encoder_inputs)
            # decoder_inputs = (data.to(self.device) for data in decoder_inputs)
            # decoder_targets = (data.to(self.device) for data in decoder_targets)
            # properties = properties.to(self.device)

            ''' RECONSTRUCTION PHASE '''
            latent_cell_state = model.encoder_forward(*encoder_inputs)  # Shape: Batch size x latent_size

            # Output data shape: batch size x vocab length x embedding size
            output_data, output_lengths, _ = model.decoder_forward(
                *decoder_inputs, latent_cell_state, is_latent_states=True
            )

            # Get the discriminator outputs using the latent cell states
            discrim_output = model.discrim_forward(latent_cell_state)

            # Predict the properties using the latent cell states
            pred_properties = model.mlp_forward(latent_cell_state)
            # Predicted Property Loss
            pred_properties_loss = criterions['property_predictor'](pred_properties, properties)
            # epoch_info['property_predictor'].append(pred_properties_loss.detach().item())
            # epoch_info['property_predictor'] = pred_properties_loss.detach().item()

            # Zip the decoder outputs and the lengths, as well as the target output and lengths
            outputs = torch.cat(
                [t[:l] for t, l in zip(output_data, output_lengths)], dim=0
            )
            targets = torch.cat(
                [t[:l] for t, l in zip(*decoder_targets)], dim=0
            )

            # Alternate every batch
            if idx % self.config.get('discrim_step') == 0:
                # Calculate the reconstruction loss
                recon_loss = criterions['autoencoder'](outputs, targets)
                discrim_target = torch.ones(
                    latent_cell_state.shape[0], 1, device=self.device
                )
                gen_loss = criterions['discriminator'](discrim_output, discrim_target)
                total_loss = recon_loss + gen_loss
                # epoch_info['autoencoder'] = ( epoch_info['autoencoder'] * (idx // 2) + recon_loss.detach().item() )
                # / (idx // 2 + 1) epoch_info['generator'] = ( epoch_info['generator'] * (idx // 2) +
                # gen_loss.detach().item() ) / (idx // 2 + 1)
            else:
                discrim_target = torch.zeros(
                    latent_cell_state.shape[0], 1, device=self.device
                )
                gen_loss = criterions['discriminator'](discrim_output, discrim_target)

                discrim_input = model.sample_latent(n=latent_cell_state.shape[0])  # n is the batch size
                discrim_output = model.discrim(discrim_input)
                discrim_target = torch.ones(
                    latent_cell_state.shape[0], 1, device=self.device
                )
                discrim_loss = criterions['discriminator'](discrim_output, discrim_target)
                total_loss = 0.5 * gen_loss + 0.5 * discrim_loss
                # epoch_info['discriminator'] = (
                #                                       epoch_info['discriminator'] * (idx // 2) + total_loss.item()
                #                               ) / (idx // 2 + 1)

            total_loss += pred_properties_loss

            ''' PRINT LOSS INFO '''
            if idx % 100 == 0:
                #     print(f"Auto Encoder Loss: {epoch_info['autoencoder']}")
                #     print(f"Generator Loss: {epoch_info['generator']}")
                #     print(f"Discriminator Loss: {epoch_info['discriminator']}")
                #     print(f"MLP Loss: {epoch_info['property_predictor']}")
                print(f'Batch: {idx}')

            if optimizers is not None:
                optimizers['autoencoder'].zero_grad()
                optimizers['discriminator'].zero_grad()
                optimizers['property_predictor'].zero_grad()
                total_loss.backward()
                for param in model.parameters():
                    param.grad.clamp_(-5, 5)
                if idx % self.config.get('discrim_step') == 0:
                    optimizers['autoencoder'].step()
                else:
                    optimizers['discriminator'].step()
                optimizers['property_predictor'].step()
            del total_loss
            torch.cuda.empty_cache()

    def train(self, loader, init=True, train_index=0):
        self.model.to(self.device)
        self.model.zero_grad()

        # Should Only Init the Optimizer When Attempting to RESTART Training
        # This is to avoid disruption of the learning rate during recall of this function
        if init:
            print('Initializing Optimizers...')
            self.optimizers = {
                'autoencoder': optim.Adam(
                    list(self.model.encoder.parameters()) +
                    list(self.model.decoder.parameters()), lr=self.lr
                ),
                'discriminator': optim.Adam(
                    self.model.discrim.parameters(), lr=self.lr
                ),
                'property_predictor': optim.Adam(
                    self.model.mlp.parameters(), lr=self.lr
                )
            }

        print('Sched Step Size:', self.sched_step_size)
        ''' RECENT ADD '''
        if self.use_scheduler:
            schedulers = {
                k: optim.lr_scheduler.StepLR(v, self.sched_step_size, self.sched_gamma)
                for k, v in self.optimizers.items()
            }
        ''' RECENT ADD'''

        print("Training on: ", self.model.device)
        print("--- BEGINNING TRAINING ---")
        print('*Performance results printed every 100 mini-batches*')
        start_training_time = time()
        keys = ['idx', 'epoch', 'autoencoder', 'generator', 'discriminator', 'property_predictor', 'mode']

        for epoch in range(self.num_epochs):
            start = time()

            ''' RECENT ADD '''
            if self.use_scheduler:
                for scheduler in schedulers.values():
                    scheduler.step()
            ''' RECENT ADD '''

            self.train_epoch(epoch=epoch, model=self.model, loader=loader, keys=keys,
                             optimizers=self.optimizers, criterions=self.criterion)
            epoch_duration = time() - start

            # Print Progress Info
            print(f"Epoch - {epoch}; Run Time - {epoch_duration:.5}s")
            print(f'Percent Completion: {((epoch / self.num_epochs) * 100):.5}')
            print()

            # print_results(epoch, losses, epoch_duration, self.num_epochs)
            torch.cuda.empty_cache()

        print("--- DONE TRAINING ---")
        total_training_time = '{0:.5}'.format(time() - start_training_time)
        print(f'Total Time to Train: {total_training_time}s')

    def save_model(self, path):
        print("SAVING MODEL...")
        torch.save(self.model.state_dict(), path)
        print(f"MODEL SAVED TO: {path}")

    def get_model(self):
        return self.model

    def get_char_vocab(self):
        return self.vocab

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def gen_samples(self, config, n_batch=None, model=None, vocab=None):
        n_batch = n_batch if n_batch else config.get('gen_samples')
        m = model if model else self.model
        if vocab:
            m.vocab = vocab
        return m.sample(n_batch=n_batch, max_len=config.get('max_len'))

    def gen_samples_from_z(self, config, z, model=None, vocab=None):
        m = model if model else self.model
        if vocab:
            m.vocab = vocab
        return m.sample_z(z=z, max_len=config.get('max_len'))
