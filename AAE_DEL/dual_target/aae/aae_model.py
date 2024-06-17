import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class encoder_net(nn.Module):
    def __init__(self, embedding_layer, hidden_size, num_layers,
                 bidirectional, dropout, latent_size):
        super(encoder_net, self).__init__()
        self.embedding_layer = embedding_layer
        self.lstm = nn.LSTM(self.embedding_layer.embedding_dim,
                            hidden_size, num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        self.linear = nn.Linear(int(bidirectional + 1) * num_layers * hidden_size, latent_size)

    def forward(self, x, lengths):
        batch_size = x.shape[0]

        x = self.embedding_layer(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (_, c0) = self.lstm(x)
        c0 = c0.permute(1, 2, 0).contiguous().view(batch_size, -1)
        c0 = self.linear(c0)

        return c0


class decoder_net(nn.Module):
    def __init__(self, embedding_layer, hidden_size, num_layers, dropout, latent_size):
        super(decoder_net, self).__init__()
        self.latent2hidden_layer = nn.Linear(latent_size, hidden_size)
        self.embedding_layer = embedding_layer
        self.lstm_layer = nn.LSTM(self.embedding_layer.embedding_dim,
                                  hidden_size, num_layers,
                                  batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, self.embedding_layer.num_embeddings)

    def forward(self, x, lengths, states, is_latent_states=False):
        if is_latent_states:
            c0 = self.latent2hidden_layer(states)
            c0 = c0.unsqueeze(0).repeat(self.lstm_layer.num_layers, 1, 1)
            h0 = torch.zeros_like(c0)
            states = (h0, c0)

        x = self.embedding_layer(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, states = self.lstm_layer(x, states)
        x, lengths = pad_packed_sequence(x, batch_first=True)
        x = self.linear(x)

        return x, lengths, states


class discriminator_net(nn.Module):
    def __init__(self, input_size, layers):
        super(discriminator_net, self).__init__()

        input_features = [input_size] + layers
        output_features = layers + [1]

        self.layers_seq = nn.Sequential()
        for k, (i, o) in enumerate(zip(input_features, output_features)):
            self.layers_seq.add_module(f'linear_{k}', nn.Linear(i, o))
            if k != len(layers):
                self.layers_seq.add_module(f'activation_{k}', nn.ELU(inplace=True))

    def forward(self, x):
        return self.layers_seq(x)


class mlp_net(nn.Module):
    def __init__(self, input_size, layers):
        super(mlp_net, self).__init__()

        input_features = [input_size] + layers
        output_features = layers + [4]

        self.layers_seq = nn.Sequential()
        for k, (i, o) in enumerate(zip(input_features, output_features)):
            self.layers_seq.add_module(f'linear_{k}', nn.Linear(i, o))
            if k != len(layers):
                self.layers_seq.add_module(f'activation_{k}', nn.ELU(inplace=True))

    def forward(self, x):
        return self.layers_seq(x)


class AAE(nn.Module):
    def __init__(self, config, vocab):
        super(AAE, self).__init__()

        # Model Parameters
        self.embedding_size = config.get('embed_size')
        self.hidden_size = config.get('hidden_size')
        self.hidden_layers = config.get('hidden_layers')
        self.discrim_layers = config.get('discrim_layers')
        self.mlp_layers = config.get('mlp_layers')
        self.latent_size = config.get('latent_size')
        self.dropout = config.get('dropout')
        self.bidirectional = config.get('bidirectional')
        self.vocab = vocab

        # Embeddings for the Embedding Layer
        self.embeddings = nn.Embedding(len(vocab),
                                       self.embedding_size,
                                       padding_idx=vocab.pad)

        # Encoder Network
        self.encoder = encoder_net(embedding_layer=self.embeddings,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.hidden_layers,
                                   bidirectional=self.bidirectional,
                                   dropout=self.dropout,
                                   latent_size=self.latent_size)

        # Decoder Network
        self.decoder = decoder_net(embedding_layer=self.embeddings,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.hidden_layers,
                                   dropout=self.dropout,
                                   latent_size=self.latent_size)

        # Discriminator Network
        self.discrim = discriminator_net(input_size=self.latent_size,
                                         layers=self.discrim_layers)
        # Property Predictor Network
        self.mlp = mlp_net(input_size=self.latent_size,
                           layers=self.mlp_layers)

    @property
    def device(self):
        return next(self.parameters()).device

    # Forward pass for the Encoder
    def encoder_forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    # Forward pass for the Decoder
    def decoder_forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    # Forward pass for the Discriminator
    def discrim_forward(self, *args, **kwargs):
        return self.discrim(*args, **kwargs)

    # Forward Pass For Property Predictor
    def mlp_forward(self, *args, **kwargs):
        return self.mlp(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def string2tensor(self, string, device="model"):
        ids = self.vocab.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device
        )
        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocab.ids2string(ids, rem_bos=True, rem_eos=True)
        return string

    def sample_latent(self, n):
        return torch.randn(n, self.latent_size, device=self.device)

    def sample(self, n_batch, max_len=100):
        with torch.no_grad():
            samples = []
            states = self.sample_latent(n_batch)
            lengths = torch.zeros(
                n_batch, dtype=torch.long, device=self.device
            )
            prevs = torch.empty(
                n_batch, 1, dtype=torch.long, device=self.device
            ).fill_(self.vocab.bos)
            one_lens = torch.ones(
                n_batch, dtype=torch.long, device=self.device
            )
            is_end = torch.zeros(
                n_batch, dtype=torch.bool, device=self.device
            )
            # print(f'Sampling Count {n_batch}...')
            for i in range(max_len):

                logits, _, states = self.decoder(prevs, one_lens, states, i == 0)
                logits = torch.softmax(logits, 2)
                shape = logits.shape[:-1]
                logits = logits.contiguous().view(-1, logits.shape[-1])
                currents = torch.distributions.Categorical(logits).sample()
                currents = currents.view(shape)

                is_end[currents.view(-1) == self.vocab.eos] = 1
                if is_end.sum() == max_len:
                    break

                currents[is_end, :] = self.vocab.pad
                # samples.append(currents.cpu())
                samples.append(currents)

                lengths[~is_end] += 1

                prevs = currents

            if len(samples):
                samples = torch.cat(samples, dim=-1)
                samples = [
                    self.tensor2string(t[:l]) for t, l in zip(samples, lengths)
                ]
            else:
                samples = ['' for _ in range(n_batch)]
            return samples

    def sample_z(self, z, max_len=100):
        with torch.no_grad():
            samples = []
            n_batch = z.shape[0]
            states = z.to(self.device)
            lengths = torch.zeros(
                n_batch, dtype=torch.long, device=self.device
            )
            prevs = torch.empty(
                n_batch, 1, dtype=torch.long, device=self.device
            ).fill_(self.vocab.bos)
            one_lens = torch.ones(
                n_batch, dtype=torch.long, device=self.device
            )
            is_end = torch.zeros(
                n_batch, dtype=torch.bool, device=self.device
            )
            # print(f'Sampling Count {n_batch}...')
            for i in range(max_len):
                logits, _, states = self.decoder(prevs, one_lens, states, i == 0)
                logits = torch.softmax(logits, 2)
                shape = logits.shape[:-1]
                logits = logits.contiguous().view(-1, logits.shape[-1])
                currents = torch.distributions.Categorical(logits).sample()
                currents = currents.view(shape)

                is_end[currents.view(-1) == self.vocab.eos] = 1
                if is_end.sum() == max_len:
                    break

                currents[is_end, :] = self.vocab.pad
                samples.append(currents.cpu())
                lengths[~is_end] += 1

                prevs = currents

            if len(samples):
                samples = torch.cat(samples, dim=-1)
                samples = [
                    self.tensor2string(t[:l]) for t, l in zip(samples, lengths)
                ]

            else:
                samples = ['' for _ in range(n_batch)]

            return samples
