from aae.aae_trainer import AAE_Trainer
from learner.dataset2 import SMILESDataset

import copy
import math
import time
import pickle
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.preprocess import add_fragments

from molecules.properties import add_property
from molecules.structure import (add_atom_counts, add_bond_counts, add_ring_counts)

from utils.config import get_dataset_info
from utils.postprocess import score_smiles_samples


class AAE_DEL:
    def __init__(self, config):
        self.config = config

        # Testing Parameters
        self.config.set('save_pops', True)
        # self.config.set('batch_size', 256)
        # self.config.set('subsets', 24)
        # self.config.set('num_generations', 5)
        # self.config.set('population_size', 5000)  # population_size < batch_size * subsets
        # self.config.set('init_num_epochs', 2)
        # self.config.set('subsequent_num_epochs', 2)

        self.orig_dataset = SMILESDataset(self.config)  # SMILES dataset
        self.population_size = self.config.get('population_size')  # len(self.dataset)
        self.orig_vocab = copy.deepcopy(self.orig_dataset.get_vocab())
        self.trainer = AAE_Trainer(self.config, self.orig_vocab)
        self.device = 'cuda' if self.config.get('use_gpu') and torch.cuda.is_available() else 'cpu'

        # An index to mark the training run.
        # This allows for a unique file for each training run through the generations
        self.train_index = 1

    def train(self):
        print('DEL 2')
        generations = self.config.get('num_generations')
        d1 = {
            'validity_pop': [0] * generations,
            'novelty': [0] * generations,
            'diversity': [0] * generations,
            'validity_sampler': [0] * generations
        }

        vnds_pop = pd.DataFrame(d1)
        vnds_dgm = pd.DataFrame(d1)

        print('Init VNDS_POP:\n', vnds_pop)
        print('Init VNDS_DGM:\n', vnds_dgm)

        time_random_sampling = 0
        start_total = time.time()
        for g in range(generations):
            print(f'\n----- GENERATION {g} -----\n')
            # Training Step
            if g == 0:
                print('First Generation')
                dataset = copy.deepcopy(self.orig_dataset)  # Keep a copy of the original dataset
                self.trainer.num_epochs = self.config.get('init_num_epochs')
                start_model_training = time.time()
                self.trainer.train(dataset.get_loader())

                # Save the Trained Model
                save_path = self.config.path('model') / 'del_pretrain.pt'
                self.trainer.save_model(path=save_path)
                end_model_training = time.time()
                time_model_training = end_model_training - start_model_training
            else:
                # Update the Training Data
                # dataset.data = pd.concat([dataset.data, self.new_pop], ignore_index=True)
                dataset.data = self.new_pop
                print('Dataset Shape', dataset.data.shape)
                dataset.data = dataset.data.drop_duplicates(subset=['smiles']).reset_index(drop=True)

                print('Dataset Shape', dataset.data.shape)
                print('New Data:', dataset.data.tail(5))

                # Update Training Parameters
                self.trainer.num_epochs = self.config.get('subsequent_num_epochs')
                self.trainer.sched_step_size = math.ceil(self.trainer.sched_step_size / 2)

                # Continue Training on the Updated Data
                if not self.config.get('no_finetune'):
                    self.trainer.train(dataset.get_loader(), init=False, train_index=self.train_index)
                    self.train_index += 1

            # Get Current Training Data, Add and Get Properties

            # Original
            samples = dataset.data  # pd Dataframe of all data

            # Work-around
            # samples = self.get_properties(dataset.data['smiles'].values.tolist())  # DataFrame with properties
            properties = samples.loc[:, ['SAS', 'logP', 'CA9', 'GPX4', 'LPA1']]
            properties['LPA1'] = -properties['LPA1']  # To be minimized
            properties = properties.to_numpy()
            print(len(properties))

            # If Population Size <= 1000 or g in [First, Middle, Last] Generation
            if (self.population_size <= 1000 and g < generations) or \
                    (self.population_size > 1000 and g in [0,  # First Generation
                                                           math.ceil(generations / 2) - 1,  # Mid Way
                                                           generations - 1]):  # Last Generation
                start_random_sampling = time.time()
                if g < generations - 1:
                    postfix = str(g)
                else:
                    postfix = 'final'
                num_samples = min(30000, self.population_size)
                samples_random = self.trainer.gen_samples(self.config, n_batch=num_samples)
                marks, vnd_dgm = score_smiles_samples(samples_random, samples['smiles'])
                vnd_dgm.append(np.array(vnd_dgm).mean())
                vnds_dgm.loc[g] = vnd_dgm  # Save Sampling Results
                samples_random = np.array(samples_random)
                samples_random = samples_random[marks]  # Filter Invalid Samples
                samples_random = [x for x in samples_random if x]  # Filter Empty Strings
                print(samples_random)
                samples_random = self.get_properties(samples_random)  # Convert to pd DataFrame and Add Properties

                # Adjust Column Order
                fieldnames = samples.columns.values.tolist()
                samples_random = samples_random.loc[:, fieldnames]  # Same Order of Fields as Original Samples

                # Save
                prefix = 'generated_from_random'
                folder_name = 'samples_del'
                filename = self.config.path(folder_name) / f"{prefix}_{postfix}.csv"
                samples_random.to_csv(filename)
                random_sampling_duration = time.time() - start_random_sampling
                time_random_sampling += random_sampling_duration
                print('Time Random Sampling:', time_random_sampling)

            # Back GPU (if available)
            # print('Model is now on:', self.device)
            # self.trainer.model.to(self.device)
            # print('Model is now on:', self.device)

            # Get Latent Representation
            print('Get Latent Representation of Current Population')
            z = self.get_z(data=samples)
            z = z.numpy()

            # Evolutionary Operations
            if g == 0:
                p_min = np.min(properties, axis=0)
                p_max = np.max(properties, axis=0)

            # If population = 0, use all training samples, otherwise use specified number
            if self.population_size > 0:
                length_samples = len(samples)
                print("Samples Length:", length_samples)
                print("Population Size:", self.population_size)
                ind = np.random.choice(len(samples), self.population_size, replace=False)  # Length: Population SIze

                # Randomly Select Samples, Properties and Z, based on ind
                samples = samples.loc[ind, :]
                properties = properties[ind, :]
                z = z[ind, :]

            if self.config.get('ranking') == 'sopr':
                print('Sum of Objective Properties Ranking...')
                rank = self.get_rank_sum(properties)

                print('Evolutionary Operations: Selection, Recombination, Mutation on Z...')
                z = self.evol_ops_c(z, rank, prob_ts=self.config.get('prob_ts'),
                                    crossover_method=self.config.get('crossover'),
                                    mutation_rate=self.config.get('mutation'))
            else:
                print('Fast Non-Dominated Sort to get Fronts...')
                rank, Fs = self.fast_nondominated_sort(properties)
                print('Crowding Distance Sort for each Front...')
                _, dists_vec = self.crowding_distance_all_fronts(properties, Fs, p_min, p_max, n_jobs=16)

                print('Evolutionary Operations: Selection, Recombination, Mutation on Z...')
                z = self.evol_ops(z, rank, dists_vec,
                                  prob_ts=self.config.get('prob_ts'),
                                  crossover_method=self.config.get('crossover'),
                                  mutation_rate=self.config.get('mutation'))

            print('Z Shape:', z.shape)
            print('Generate New Samples from Modified Z...')
            zloader = DataLoader(z, batch_size=self.config.get('batch_size'))
            # Sample From Batches of Z
            new_samples = []  # List of new samples from Z
            valids = []  # List of valid tags
            for zbatch in zloader:
                samples_batch = self.trainer.gen_samples_from_z(self.config, zbatch)
                marks, _ = score_smiles_samples(samples_batch, dataset.data)

                # Mark Empty Strings as Invalid/ False
                for i in range(len(samples_batch)):
                    s = samples_batch[i]
                    if not s and marks[i]:
                        marks[i] = False

                valids_batch = np.array(samples_batch)[marks]  # Remove Invalid Samples
                new_samples += list(valids_batch)  # Store the Valid Samples
                valids += list(marks)  # Store the tags to be used to filter Z
            print('Generation:', g)
            print('Num Samples From Z:', len(new_samples))
            # print('New Samples From Z:', new_samples)
            z = z[valids]  # Remove Invalid Z
            # print('Z:', z)
            # print('Z Shape:', z.shape)

            # If no new samples, move on to next generation
            if not new_samples:
                print('No New Samples Generated')
                break

            # Obtain Fitness Score/ Properties for Generated Samples
            new_samples = self.get_properties(new_samples)  # Samples is now a pd DataFrame

            # Obtain Validity, Novelty and Diversity of Population Data
            ''' NEED TO REVISE THIS '''
            _, vnd_pop = score_smiles_samples(new_samples['smiles'].to_list(), dataset.data)
            vnd_pop.append(np.array(valids).mean())
            vnds_pop.loc[g] = vnd_pop
            print('VNDS_POP:', vnds_pop)

            # Remove Duplicates
            new_samples = new_samples.drop_duplicates(subset=['smiles']).reset_index(drop=True)
            # print('Columns:', new_samples.columns)

            # Merging New Samples With Old Population
            print('Producing New Generation of Data')
            fieldnames = samples.columns.values.tolist()
            print(fieldnames)
            print(samples)
            print(new_samples)
            new_samples = new_samples.loc[:, fieldnames]  # Organize New Samples in the Same Order as Samples
            print('Samples Shape:', samples.shape)
            print('New Samples Shape:', new_samples.shape)
            combined_samples = pd.concat([samples, new_samples], ignore_index=True)
            print('Combined Samples Shape:', combined_samples.shape)
            # Remove Duplicates
            combined_samples = combined_samples.drop_duplicates(subset=['smiles']).reset_index(drop=True)

            # Combined Properties
            combined_properties = combined_samples.loc[:, ['SAS', 'logP', 'CA9', 'GPX4', 'LPA1']]
            combined_properties['LPA1'] = -combined_properties['LPA1']  # To be minimized
            combined_properties = combined_properties.to_numpy()

            # Create New Population Based On New Combined Ranking
            if self.config.get('ranking') == 'sopr':
                # Rank All Samples and Add to the Dataframe
                self.new_rank = self.get_rank_sum(combined_properties)
                combined_samples['rank'] = self.new_rank
                # Sort By Ranks
                sorted_combined_samples = combined_samples.sort_values(['rank'], ascending=True, ignore_index=True)
                cols = sorted_combined_samples.columns.values.tolist()
                if 'rank' in cols:
                    cols.remove('rank')

                # Separate Ranking from the New Population
                # Ranking is not in the original dataframe structure
                # Would cause error in subsequent training if left
                # Filter out the excess ranks as well...
                # Fixed blank row error in final population save
                self.new_rank = sorted_combined_samples.loc[:self.population_size - 1, ['rank']]
                self.new_pop = sorted_combined_samples.loc[:self.population_size - 1, cols]
                print('The shape of the new pop is', self.new_pop.shape)
                print('The shape of new rank is', self.new_rank.shape)
                print(f'New Pop: {self.new_pop.head()}')
            else:
                # Rank All Samples
                rank, Fs = self.fast_nondominated_sort(combined_properties)
                dists_all, dists_vec = self.crowding_distance_all_fronts(combined_properties, Fs, p_min, p_max, n_jobs=16)

                combined_size = len(rank)
                if combined_size < self.population_size:
                    self.population_size = combined_size

                self.new_pop = []
                self.new_rank = []
                self.new_Fs = []
                self.new_dists = []
                count = 0
                i = 0

                # Add Front 1, 2, 3, ... iteratively
                while (count + len(Fs[i])) <= self.population_size:
                    self.new_pop.append(combined_samples.loc[Fs[i], :])
                    self.new_rank.append(rank[Fs[i]])
                    self.new_Fs.append(Fs[i])
                    self.new_dists.append(dists_all[i])
                    count = count + len(Fs[i])
                    # print('count={}'.format(count))
                    i = i + 1
                    if i > len(Fs):
                        break

                    # put part of front i in new_pop
                if count < self.population_size:
                    inds = np.argsort(dists_all[i])
                    inds = inds[::-1]
                    inds = inds[0:(self.population_size - count)]
                    self.new_pop.append(combined_samples.loc[Fs[i][inds], :])
                    self.new_rank.append(rank[Fs[i][inds]])
                    self.new_Fs.append(Fs[i][inds])
                    self.new_dists.append(dists_all[i][inds])

                self.new_pop = pd.concat(self.new_pop, ignore_index=True)
                print('The shape of new pop is', self.new_pop.shape)
                self.new_rank = np.concatenate(self.new_rank)
                print('The shape of new rank is', self.new_rank.shape)

            # Save Results
            if g == self.config.get('num_generations') - 1 or (
                    g < self.config.get('num_generations') - 1 and self.config.get('save_pops')):
                self.save_population(g)
            # save latent representations
            if g in [0, math.ceil(self.config.get('num_generations') / 2) - 1, self.config.get('num_generations') - 1]:
                # back to gpu
                if self.config.get('use_gpu'):
                    self.trainer.model = self.trainer.model.cuda()
                z = z = self.get_z(data=self.new_pop)
                z = z.cpu().numpy()
                self.save_z(g, z, self.new_pop)

        # Save Running Time
        end_total = time.time()
        time_total = end_total - start_total
        elapsed_total = time.strftime("%H:%M:%S", time.gmtime(time_total))
        elapsed_random_sampling = time.strftime("%H:%M:%S", time.gmtime(time_random_sampling))
        elapsed_model_training = time.strftime("%H:%M:%S", time.gmtime(time_model_training))
        time_save = {'time_second': [time_total, time_random_sampling, time_model_training],
                     'time_hms': [elapsed_total, elapsed_random_sampling, elapsed_model_training]}
        time_save = pd.DataFrame(time_save, index=['total', 'random_sampling', 'model_training'])
        filename = self.config.path('performance') / f"running_time.csv"
        time_save.to_csv(filename)
        # save validity, novelty, and diversity
        filename = self.config.path('performance') / f"vnds_pop.csv"
        vnds_pop.to_csv(filename)
        filename = self.config.path('performance') / f"vnds_dgm.csv"
        vnds_dgm.to_csv(filename)

        print('Done DEL training.')
        # return self.new_pop, self.new_rank, self.new_Fs, self.new_dists

    def get_properties(self, samples, n_jobs=16):

        info = get_dataset_info(self.config.get('dataset'))
        samples = pd.DataFrame(samples, columns=['smiles'])

        samples = add_fragments(samples, info, n_jobs)
        samples = add_atom_counts(samples, info, n_jobs)
        samples = add_bond_counts(samples, info, n_jobs)
        samples = add_ring_counts(samples, info, n_jobs)

        # add same properties as in training/test dataset
        for prop in info['properties']:
            if prop == 'mr':
                continue
            samples = add_property(samples, prop, n_jobs)

        # samples.to_csv(config.path('samples') / 'aggregated.csv')
        return samples

    def encode_z(self, x, lengths):
        with torch.no_grad():
            z = self.trainer.model.encoder_forward(x, lengths)
        return z

    def get_z(self, data=None):
        if data is not None:
            dataset = copy.deepcopy(self.orig_dataset)
            dataset.change_data(data)
        else:
            dataset = self.orig_dataset

        zs = torch.zeros(len(dataset), self.config.get('latent_size'))
        loader = dataset.get_loader()  # Load data
        for idx, (encoder_inputs, _, _, _) in enumerate(loader):
            # Prep Data
            encoder_inputs = (data.to(self.device) for data in encoder_inputs)
            latent = self.encode_z(*encoder_inputs)  # Shape: Batch Size x latent_size
            start = idx * self.config.get('batch_size')
            end = start + latent.shape[0]
            zs[start:end, :] = latent
        return zs

    def get_rank_sum(self, properties):
        # Properties are sorted as qed, SAS, logP
        # Another way to get the rank is to call argsort twice on the array
        # qed_rank = np.argsort(np.argsort(qed))
        # But this can be computationally expensive for large arrays
        # Using the method below, we can avoid the need for a double sort

        N = len(properties)
        sas, logp, ca9, gpx4, lpa1 = properties[:, 0], properties[:, 1], properties[:, 2], properties[:, 3], properties[:, 4]

        # Get Orders of Properties
        sas_order, logp_order, ca9_order, gpx4_order, lpa1_order = np.argsort(sas), np.argsort(logp), np.argsort(ca9), np.argsort(gpx4), np.argsort(lpa1)

        # Get Ranks of Properties
        sas_rank, logp_rank, ca9_rank, gpx4_rank, lpa1_rank = \
            np.empty_like(sas_order), np.empty_like(logp_order), np.empty_like(ca9_order), np.empty_like(gpx4_order), np.empty_like(lpa1_order)
        sas_rank[sas_order], logp_rank[logp_order], ca9_rank[ca9_order], gpx4_rank[gpx4_order], lpa1_rank[lpa1_order] = \
            np.arange(N), np.arange(N), np.arange(N), np.arange(N), np.arange(N)

        # Get Rank Sums
        ranks = np.array([np.sum([sas_rank[i], logp_rank[i], ca9_rank[i], gpx4_rank[i], lpa1_rank[i]]) for i in range(N)])
        return ranks

    def tournament_selection_c(self, rank, prob_ts, k=2):
        N = len(rank)
        inds_num = np.array(range(N))
        # Randomly select k points
        candidates = np.random.choice(inds_num, size=k, replace=False)

        # Rank Canditates
        rank_cand = rank[candidates]
        # Order Candidates
        order = np.array([sorted(rank_cand).index(x) for x in rank_cand])
        candidates_ordered = candidates[order]
        # Assign Probabilities
        probs = prob_ts * (1 - prob_ts) ** np.array(range(k))
        probs_cum = np.cumsum(probs)
        r = np.random.rand()
        sel_i = 0
        for i in range(k):
            sel_i = k - 1  # Initialize to the last
            if r < probs_cum[i]:
                sel_i = i
                break
        selected = candidates_ordered[sel_i]
        return selected

    def tournament_selection_N_c(self, num, rank, prob_ts, k=2):
        """
        Select num points
        k: scalar, number of points to be randomly selected from the population
        """

        selected_inds = [self.tournament_selection_c(rank, prob_ts, k) for n in range(num)]
        return selected_inds

    def evol_ops_c(self, z, rank, prob_ts=0.95, crossover_method='linear', mutation_rate=0.01):
        N = self.population_size
        selected_inds = self.tournament_selection_N_c(num=N, rank=rank, prob_ts=prob_ts, k=2)
        selected_points = z[selected_inds]

        new_data = []
        for n in range(0, N, 2):
            ind1 = n
            ind2 = n + 1
            if n == N - 1:  # N is odd
                ind1 = n - 1
                ind2 = n
            new_point1, new_point2 = self.crossover(point1=selected_points[ind1], point2=selected_points[ind2],
                                                    method=crossover_method)
            self.mutate(new_point1, mutation_rate=mutation_rate)
            self.mutate(new_point2, mutation_rate=mutation_rate)

            new_data.append(new_point1)
            if n != N - 1:
                new_data.append(new_point2)

        return np.array(new_data)

    def fast_nondominated_sort(self, P):
        '''
        P is a numpy array of N by M where,
        N is the number of data points/ solutions,
        M is the number of scores

        Test code:
        import numpy as np
        import matplotlib.pyplot as plt

        P = 100 * np.random.rand(1000, 2)
        rank = nondominated_sort(P)
        M = rank.max()
        for m in range(M):
        plt.plot(P[rank==m][:,0], P[rank==m][:,1], ls = '', marker ='o', markersize=4)

        plt.show()
        '''

        N, M = P.shape

        inds_all_num = np.array(range(N))

        Np = np.zeros(N, dtype=int)  # number of solutions which dominate solution p
        rank = np.zeros(N, dtype=int)
        Sp = []  # set of solutions that p dominate
        Fs = []

        for n in range(N):
            diffs = P[n] - P
            inds_le = ((diffs) <= 0).all(axis=1)
            inds_l = ((diffs) < 0).any(axis=1)
            inds = inds_le & inds_l
            Sp.append(inds_all_num[inds])

            # >= & >
            # inds = ~inds_l & ~inds_le
            inds = ~(inds_l | inds_le)
            Np[n] = inds.sum()

        F = []
        F = inds_all_num[Np == 0]
        rank[F] = 0

        i = 0  # rank
        while len(F) > 0:
            Fs.append(np.array(F))
            Q = []
            for p in F:
                for q in Sp[p]:
                    Np[q] = Np[q] - 1
                    if Np[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i = i + 1
            F = Q

        return rank, Fs

    def crowding_distance_assignment(self, I, f_min, f_max):
        '''
        I: Numpy array of size N by M
        It can be the property matrix for one front
        '''

        N, M = I.shape
        dists = np.zeros(N, dtype=float)
        for m in range(M):  # for each property
            inds = np.argsort(I[:, m])
            dists[inds[0]] = np.inf
            dists[inds[-1]] = np.inf
            dists[inds[1:-1]] = dists[inds[1:-1]] + (I[inds[2:], m] - I[inds[0:-2], m]) / (f_max[m] - f_min[m])

        return dists

    def crowding_distance_all_fronts(self, P, Fs, f_min, f_max, n_jobs=16):
        '''
        P: Properties
        Fs: fronts
        f_min: min values of properties
        f_max: max values of properties
        '''

        dists_all = [self.crowding_distance_assignment(P[F], f_min, f_max) for F in Fs]
        dists_vec = np.zeros(P.shape[0])
        for F, D in zip(Fs, dists_all):
            dists_vec[F] = D
        return dists_all, dists_vec

    def tournament_selection(self, rank, dists_vec, prob_ts, k=2):
        N = len(rank)
        inds_num = np.array(range(N))
        # randomly selecting k points
        candidates = np.random.choice(inds_num, size=k, replace=False)

        # rank candidates
        rank_cand = rank[candidates]  # prefer small rank
        # crowding distances
        dist_cand = -dists_vec[candidates]  # perfer large distance
        # order these candidates
        order = np.lexsort((dist_cand, rank_cand))
        candidates_ordered = candidates[order]
        # assign probability
        probs = prob_ts * (1 - prob_ts) ** np.array(range(k))
        # inds_k = np.array( range(k) )
        # inds_k = inds_k[order]
        probs_cum = np.cumsum(probs)
        r = np.random.rand()
        sel_i = 0
        for i in range(k):
            sel_i = k - 1  # initialize to the last
            if r < probs_cum[i]:
                sel_i = i
                break
        selected = candidates_ordered[sel_i]
        return selected

    def tournament_selection_N(self, num, rank, dists_vec, prob_ts, k=2, n_jobs=16):
        '''
        Select num points
        k: scalar, number of points to be randomly selected from the population
        '''

        selected_inds = [self.tournament_selection(rank, dists_vec, prob_ts, k) for n in range(num)]
        return selected_inds

    def crossover(self, point1, point2, method='linear'):
        '''
        Point1, Point2 = Vector of size k, two data points
        '''
        K = point1.size

        if method == 'linear':
            d = 0.25
            alpha = np.random.rand()
            alpha = -d + (1 + 2 * d) * alpha
            new_point1 = point1 + alpha * (point2 - point1)
            alpha = np.random.rand()
            alpha = -d + (1 + 2 * d) * alpha
            new_point2 = point1 + alpha * (point2 - point1)
        elif method == 'discrete':
            alpha = np.random.randint(K)
            new_point1 = np.zeros(K, dtype=np.float32)
            new_point1[:alpha] = point1[:alpha]
            new_point1[alpha:] = point2[alpha:]
            # alpha=np.random.randint(K)
            new_point2 = np.zeros(K, dtype=np.float32)
            new_point2[:alpha] = point2[:alpha]
            new_point2[alpha:] = point1[alpha:]
            # temp = np.copy( point2[:alpha] )
            # point2[:alpha] = point1[:alpha]
            # point1[:alpha] = temp
            # temp = np.copy( point2[alpha:] )
            # point2[alpha:] = point1[alpha:]
            # point1[alpha:] = temp

        return new_point1, new_point2

    def mutate(self, point, mutation_rate=0.01):
        p = np.random.rand()
        # mutation
        if p < mutation_rate:
            pos = np.random.randint(point.size)
            point[pos] = point[pos] + np.random.randn()

    def evol_ops(self, z, rank, dists_vec, prob_ts=0.95, crossover_method='linear', mutation_rate=0.01):
        '''
        Selection, Crossover, Mutation Operations on Z
        '''

        # Selection
        # N = rank.shape[0]
        # selected_inds = self.tournament_selection_N(N, rank, dists_vec, prob_ts=prob_ts, k=2, n_jobs=-1)
        # selected_points = z[selected_inds]

        # Selection
        N = self.population_size
        selected_inds = np.random.choice(self.population_size, size=self.population_size, replace=True)
        selected_points = z[selected_inds]
        mutation_rate = 0.5

        new_data = []
        for n in range(0, N, 2):
            ind1 = n
            ind2 = n + 1
            if n == N - 1:  # N is odd
                ind1 = n - 1
                ind2 = n
            new_point1, new_point2 = selected_points[ind1], selected_points[ind2]
            # new_point1, new_point2 = self.crossover(point1=selected_points[ind1], point2=selected_points[ind2],
            #                                         method=crossover_method)
            self.mutate(new_point1, mutation_rate=mutation_rate)
            self.mutate(new_point2, mutation_rate=mutation_rate)

            new_data.append(new_point1)
            if n != N - 1:
                new_data.append(new_point2)

        new_data = np.array(new_data)

        return new_data

    def save_population(self, g):
        """
        g: integer, generation index.
        """
        if g == self.config.get('num_generations') - 1:
            g = 'final'
        # if self.config.get('ranking') == 'sopr':
        new_rank = pd.DataFrame(self.new_rank, columns=['rank'])
        data_to_save = pd.concat([self.new_pop, new_rank], axis=1)
        # else:
        #     data_to_save = self.new_pop
        data_to_save = data_to_save.sort_values(['rank'], ascending=True, ignore_index=True)
        print('Data to save:', data_to_save)
        filename = self.config.path('samples_del') / f"new_pop_{g}.csv"
        data_to_save.to_csv(filename)

    def save_z(self, g, z, samples=None, prefix='pop'):
        """
        Save z for visualization purpose.
        g: scalar integer, the generation index.
        """
        if g == 0:
            g = 'init'
        elif g == self.config.get('num_generations') - 1:
            g = 'final'
        else:
            g = str(g)

        filename = self.config.path('samples_del') / f"{prefix}_z_{g}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump((z, samples), file=f)

    def load_z(self, g, prefix='pop'):
        if g >= self.config.get('num_generations') - 1:
            g = 'final'
        filename = self.config.path('samples_del') / f"{prefix}_z_{g}.pkl"
        with open(filename, 'rb') as f:
            z, samples = pickle.load(f)
        return z, samples
