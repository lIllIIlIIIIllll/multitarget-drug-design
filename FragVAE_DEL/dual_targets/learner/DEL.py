#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:02:09 2020

@author: yifeng
"""
import pickle
import math
import time
import numpy as np
import pandas as pd
import copy
from joblib import Parallel, delayed

import torch
from torch.utils.data import DataLoader

from learner.trainer import Trainer, save_ckpt
from learner.dataset import FragmentDataset
from learner.sampler import Sampler


from molecules.properties import add_property
from molecules.structure import (add_atom_counts, add_bond_counts, add_ring_counts)

from utils.config import get_dataset_info
from utils.postprocess import score_samples

#loop
#train

# sample

class DEL():
    def __init__(self, config):
        self.config=config
        #self.model=None
        #self.population=None
        self.dataset = FragmentDataset(self.config) # initialize population
        self.population_size = self.config.get('population_size')#len(self.dataset)
        self.vocab = copy.deepcopy( self.dataset.get_vocab() )
        self.trainer = Trainer(self.config, self.vocab)
    
    def train(self):
        
        d1 = {'validity_pop':[0]*self.config.get('num_generations'), 
              'novelty':[0]*self.config.get('num_generations'), 
              'diversity':[0]*self.config.get('num_generations'),
              'validity_sampler':[0]*self.config.get('num_generations')}
        vnds_pop = pd.DataFrame(d1)
        vnds_dgm = pd.DataFrame(d1)
        
        # initialize the model
        time_random_sampling = 0
        start_total = time.time()
        for g in range(self.config.get('num_generations')):
            print('Generation: {0}'.format(g))
            # train DGM
            if g == 0:
                # keep initial optimizer and scheduler info for letter use
                optim_lr = self.config.get('optim_lr')
                sched_step_size = self.config.get('sched_step_size')
                #alpha = self.config.get('alpha')
                a_beta = self.config.get('a_beta')
                a_alpha = self.config.get('a_alpha')
                dataset = copy.deepcopy(self.dataset) # keep a copy of the original training set
                start_epoch = 0
                num_epochs = self.config.get('init_num_epochs')
                self.trainer.config.set('num_epochs', num_epochs)
                start_model_training = time.time()
                self.trainer.train(self.dataset.get_loader(), start_epoch)
                save_ckpt(self.trainer, num_epochs, filename="pretrain.pt")
                end_model_training = time.time()
                time_model_training = end_model_training - start_model_training
            else:
                ##self.dataset = FragmentDataset(self.config, kind='given', data=new_pop) # new population/data
                self.dataset.change_data(self.new_pop)
                start_epoch = start_epoch + num_epochs
                num_epochs = self.config.get('subsequent_num_epochs')
                self.trainer.config.set('num_epochs', num_epochs)
                self.trainer.config.set('start_epoch', start_epoch)
                # reset learning rate and scheduler setting
                self.config.set('optim_lr', optim_lr )
                self.config.set('sched_step_size', math.ceil(sched_step_size/2) )
                if self.config.get('increase_beta')!=1:
                    if g==1:
                        self.config.set('offset_epoch', self.config.get('init_num_epochs')) # offset epoch for computing beta
                    else:
                        self.config.set('k_beta', 0) # let beta constant when g>1
                    self.config.set('a_beta', a_beta*self.config.get('increase_beta'))
                    #self.config.set('alpha',alpha*4)
                if self.config.get('increase_alpha')!=1:
                    if g==1:
                        self.config.set('offset_epoch', self.config.get('init_num_epochs')) # offset epoch for computing beta
                    else:
                        self.config.set('k_alpha', 0) # let beta constant when g>1
                    self.config.set('a_alpha', a_alpha*self.config.get('increase_alpha'))
                # comment this out if not finetuning DGM
                if self.config.get('no_finetune'):
                    print('DGM is not finetuned.')
                elif g < self.config.get('num_generations'):
                    self.trainer.train(self.dataset.get_loader(), start_epoch)
            
            # get current training data
            samples = self.dataset.data # pd DataFrame with properties
            # properties = samples.loc[:,['qed', 'SAS', 'logP']] # qed: the larger the better, SAS: the smaller the better, logP: the smaller the better 
            properties = samples.loc[:,['SAS', 'logP', 'CA9', 'LPA1']] # SAS: the smaller the better, logP: the smaller the better, CA9: the smaller the better, GPX4: the smaller the better
            # properties['qed'] = -properties['qed'] # to be minimized
            #properties['SAS'] = properties['SAS'] # to be minimized
            properties = properties.to_numpy()
            
            
            # create a sampler for sampling from random noice or sampling from modified z
            self.sampler=Sampler(self.config, self.vocab, self.trainer.model) 
            
            if (self.population_size<=1000 and g<self.config.get('num_generations')) or (self.population_size>1000 and g in [0,math.ceil(self.config.get('num_generations')/2)-1,self.config.get('num_generations')-1]):
                # randomly generate samples with N(0,1) + decoder
                #if g in [0, self.config.get('num_generations')-1]:
                start_random_sampling = time.time()
                if g<self.config.get('num_generations')-1:
                    postfix=str(g)
                else:
                    postfix='final'
                num_samples=self.population_size
                if num_samples>30000:
                    num_samples=30000          
                samples_random, valids_random=self.sampler.sample(num_samples=self.population_size, save_results=False, 
                                                   postfix=postfix, folder_name='samples_del')
                
                _,vnd_dgm = score_samples(samples_random, dataset.data)
                vnd_dgm.append( np.array(valids_random).mean() )
                vnds_dgm.loc[g] = vnd_dgm
                
                samples_random = self.get_properties(samples_random)
                # adjust column order
                fieldnames = samples.columns.values.tolist()
                samples_random = samples_random.loc[:,fieldnames] # same order of fields as samples
                # save
                prefix = 'generated_from_random'
                folder_name='samples_del'
                filename = self.config.path(folder_name) / f"{prefix}_{postfix}.csv"
                samples_random.to_csv(filename)
                end_random_sampling = time.time()
                time_random_sampling = time_random_sampling + end_random_sampling - start_random_sampling
            
            # back to gpu
            if self.config.get('use_gpu'):
                self.trainer.model = self.trainer.model.cuda()
            
            
            # get latent representation of data
            print('Getting latent representations of current population...')
            z,mu,logvar = self.get_z()
            z = z.cpu().numpy()
            mu = mu.cpu().numpy()
            logvar = logvar.cpu().numpy()
            print(z)
            # evolutionary operations
            if g ==0:
                p_min = np.min(properties, axis=0)
                p_max = np.max(properties, axis=0)
                # if population =0, use all training samples, otherwise, use specified number.
                if self.population_size>0:
                    #samples = samples[0:self.population_size]
                    #properties = properties[0:self.population_size]
                    #z = z[0:self.population_size]
                    ind = np.random.choice(len(samples), self.population_size, replace=False )
                    samples = samples.loc[ind, :]
                    properties = properties[ind,:]
                    z = z[ind,:]
                    mu = mu[ind,:]
                    logvar = logvar[ind,:]
                # save z and original samples for t-SNE visualization
                self.save_z(g, z, mu, logvar, samples, prefix='traindata')
            
            #if g == self.config.get('num_generations'):
            #    break
            
            
            print('Fast non-dominated sort to get fronts ...')
            rank,Fs = self.fast_nondominated_sort(properties)
            print('Crowding distance sort for each front ...')
            _,dists_vec=self.crowding_distance_all_fronts(properties, Fs, p_min, p_max, n_jobs=38)
            
            print('Evolutionary operations: selection, recombination, and mutation ...')
            #pm=properties.mean().to_numpy() # properties mean
            z = self.evol_ops(z, rank, dists_vec, prob_ts = self.config.get('prob_ts'), 
                              crossover_method =self.config.get('crossover'), 
                              mutation_rate = self.config.get('mutation'))
            
            print('Generate new samples from modified z ...')
            # generate new samples
            #self.sampler=Sampler(self.config, self.vocab, self.trainer.model)
            # loader for z
            zloader = DataLoader(z, batch_size=self.config.get('batch_size'))
            # sample from batchs of z
            new_samples = [] # a list of tuples
            valids= []
            if g != 0:
                for zbatch in zloader:
                    samples_batch, valids_batch = self.sampler.sample_from_z( zbatch, save_results=False, seed=None)
                    new_samples += samples_batch # list
                    valids += list(valids_batch) # list
                #vaids = np.array(valids)
                z = z[valids] # remove invalid z
                # obtain fitness score / properties for generated samples
                new_samples = self.get_properties(new_samples) # samples now is a pd DataFrame
            
                # obtain validity, novelty, and diversity of population data
                _,vnd_pop = score_samples(new_samples, dataset.data)
                vnd_pop.append( np.array(valids).mean() )
                vnds_pop.loc[g] = vnd_pop
            
                # remove duplicates
                #new_samples = new_samples.drop_duplicates(subset = ["smiles"], ignore_index=True)
                new_samples = new_samples.drop_duplicates(subset = ["smiles"]).reset_index(drop=True)
                print(new_samples.columns)
                #new_properties = new_samples.loc[:,['qed', 'SAS', 'logP']]
                #new_properties = new_properties.to_numpy()
            
                print('Producing new generation of data ...')
                # merge new samples with old population
                fieldnames = samples.columns.values.tolist()
                new_samples = new_samples.loc[:,fieldnames] # same order of fields as samples

                print(samples.shape)
                print(new_samples.shape)
                combined_samples = pd.concat( [samples, new_samples], ignore_index=True) # dataframe
           
            # add BCC and Drugbank data to the first generation
            if g == 0:
                combined_samples = self.dataset.data.iloc[-20300:]
                print('--------first generation-------')

            # remove duplicates
            combined_samples = combined_samples.drop_duplicates(subset = ["smiles"]).reset_index(drop=True)
            
            # combined properties
            # combined_properties = np.vstack( (properties, new_properties) ) # numpy array
            # combined_properties = combined_samples.loc[:,['qed', 'SAS', 'logP']]
            combined_properties = combined_samples.loc[:,['SAS', 'logP', 'CA9', 'LPA1']]
            # combined_properties['qed'] = -combined_properties['qed'] # to be minimized
            #combined_properties['SAS'] = combined_properties['SAS'] # to be minimized
            combined_properties = combined_properties.to_numpy()
            # sort all samples
            rank,Fs = self.fast_nondominated_sort(combined_properties)
            dists_all,dists_vec=self.crowding_distance_all_fronts(combined_properties, Fs, p_min, p_max, n_jobs=38)
            
            combined_size = len(rank)
            if combined_size < self.population_size:
                self.population_size = combined_size
#            print(combined_properties.shape)
#            print('dists:{}'.format(len(dists_vec)))
#            print('check:')
#            print(len(rank))
#            length = 0
#            for F in Fs:
#                print(len(F))
#                length = length + len(F)
#            print('length={}'.format(length))
            
            # make new data of size self.population_size
            self.new_pop = []
            self.new_rank = []
            self.new_Fs = []
            self.new_dists = []
            count = 0
            i = 0
            # add front 1, 2, 3, ... itereatively
            while( count + len(Fs[i]) ) <=  self.population_size:
                self.new_pop.append( combined_samples.loc[Fs[i],:] )
                self.new_rank.append(rank[Fs[i]])
                self.new_Fs.append(Fs[i])
                self.new_dists.append(dists_all[i])
                count = count + len(Fs[i])
                #print('count={}'.format(count))
                i = i+1
                if i>len(Fs):
                    break
            
            # put part of front i in new_pop
            if count < self.population_size:
                inds = np.argsort( dists_all[i] )
                inds = inds[::-1]
                inds = inds[0:(self.population_size-count)]
                self.new_pop.append( combined_samples.loc[Fs[i][inds],:] )
                self.new_rank.append( rank[Fs[i][inds]] )
                self.new_Fs.append( Fs[i][inds] )
                self.new_dists.append( dists_all[i][inds] )
            
            self.new_pop = pd.concat(self.new_pop, ignore_index=True)
            print('The shape of new pop is', self.new_pop.shape)
            self.new_rank = np.concatenate( self.new_rank )
            print('The shape of new rank is', self.new_rank.shape)
                
            if g == self.config.get('num_generations')-1 or (g < self.config.get('num_generations')-1 and self.config.get('save_pops')):
                self.save_population(g)
            # save latent representations
            if g in [0,math.ceil(self.config.get('num_generations')/2)-1,self.config.get('num_generations')-1]:
                # back to gpu
                if self.config.get('use_gpu'):
                    self.trainer.model = self.trainer.model.cuda()
                z,mu,logvar = self.get_z(self.new_pop)
                z = z.cpu().numpy()
                mu = mu.cpu().numpy()
                logvar = logvar.cpu().numpy()
                self.save_z(g, z, mu, logvar, self.new_pop)
            
        # save running time    
        end_total = time.time()
        time_total = end_total - start_total
        elapsed_total = time.strftime("%H:%M:%S", time.gmtime(time_total))
        elapsed_random_sampling = time.strftime("%H:%M:%S", time.gmtime(time_random_sampling))
        elapsed_model_training = time.strftime("%H:%M:%S", time.gmtime(time_model_training))
        time_save = {'time_second': [time_total, time_random_sampling, time_model_training],
                     'time_hms': [elapsed_total, elapsed_random_sampling, elapsed_model_training] }
        time_save = pd.DataFrame(time_save, index=['total','random_sampling','model_training'])
        filename = self.config.path('performance') / f"running_time.csv"
        time_save.to_csv(filename)
        # save validity, novelty, and diversity
        filename = self.config.path('performance') / f"vnds_pop.csv"
        vnds_pop.to_csv(filename)
        filename = self.config.path('performance') / f"vnds_dgm.csv"
        vnds_dgm.to_csv(filename)
        
        print('Done DEL training.')
        return self.new_pop, self.new_rank, self.new_Fs, self.new_dists
            
            
    def encode_z(self, inputs, lengths):
        with torch.no_grad():
            embeddings = self.trainer.model.embedder(inputs)
            z, mu, logvar = self.trainer.model.encoder(inputs, embeddings, lengths)
        return z, mu, logvar


    def get_z(self, data=None):
        if data is not None:
            dataset = copy.deepcopy(self.dataset)
            dataset.change_data(data)
            #dataset = FragmentDataset(config, kind='given', data)
        else:
            dataset = self.dataset
        zs = torch.zeros(len(dataset), self.config.get('latent_size'))
        mus = torch.zeros(len(dataset), self.config.get('latent_size'))
        logvars = torch.zeros(len(dataset), self.config.get('latent_size'))
        loader=dataset.get_loader() # load data
        for idx, (src, tgt, lengths, properties) in enumerate(loader):
            if self.config.get('use_gpu'):
                src = src.cuda()
                tgt = tgt.cuda()
                properties = properties.cuda()
            z, mu, logvar = self.encode_z(src,lengths)
            start = idx*self.config.get('batch_size')
            end = start + z.shape[0]
            zs[start:end,:]=z
            mus[start:end,:]=mu
            logvars[start:end,:]=logvar
        return zs,mus,logvars


    def save_z(self, g, z, mu=None, logvar=None, samples=None, prefix='pop'):
        """
        Save z for visualization purpose.
        g: scalar integer, the generation index.
        """
        if g==0:
            g='init'
        elif g == self.config.get('num_generations')-1:
            g='final'
        else:
            g=str(g)
            
        filename = self.config.path('samples_del') / f"{prefix}_z_{g}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump((z,mu,logvar,samples), file=f)
        
    
    def load_z(self, g, prefix='pop'):
        if g >= self.config.get('num_generations')-1:
            g='final'
        filename = self.config.path('samples_del') / f"{prefix}_z_{g}.pkl"
        with open(filename, 'rb') as f:
            z,mu,logvar,samples=pickle.load(f)
        return z,mu,logvar,samples


    def evol_ops(self, z, rank, dists_vec, prob_ts=0.95, crossover_method='linear', mutation_rate=0.01):
        """
        Selection, cross-over, mutation operations on z.
        """
        # selection
        N = rank.shape[0]
        selected_inds = self.tournament_selection_N(N, rank, dists_vec, prob_ts=prob_ts, k=2, n_jobs=38)
        selected_points = z[selected_inds]
        
        new_data = []
        for n in range(0,N,2):
            ind1 = n
            ind2 = n+1
            if n==N-1: # N is odd
                ind1 = n-1
                ind2 = n
            new_point1,new_point2=self.crossover(point1=selected_points[ind1], point2=selected_points[ind2], method=crossover_method)
            self.mutate(new_point1, mutation_rate=mutation_rate)
            self.mutate(new_point2, mutation_rate=mutation_rate)
            
            new_data.append(new_point1)
            if n != N-1:
                new_data.append(new_point2)
            
        new_data = np.array(new_data)
        
        return new_data
    
    
    def tournament_selection(self, rank, dists_vec, prob_ts, k=2):
        N = len(rank)
        inds_num = np.array(range(N))
        # randomly selecting k points
        candidates=np.random.choice(inds_num, size=k, replace=False)
        
        # rank candidates
        rank_cand = rank[candidates] # prefer small rank
        # crowding distances
        dist_cand = -dists_vec[candidates] # perfer large distance
        # order these candidates
        order=np.lexsort( (dist_cand,rank_cand) )
        candidates_ordered = candidates[order]
        # assign probability
        probs = prob_ts*(1-prob_ts)**np.array(range(k)) 
        #inds_k = np.array( range(k) )
        #inds_k = inds_k[order]
        probs_cum=np.cumsum(probs)
        r=np.random.rand()
        sel_i = 0
        for i in range(k):
            sel_i = k-1 # initialize to the last
            if r<probs_cum[i]:
                sel_i = i
                break
        selected = candidates_ordered[sel_i]
        return selected

    
#    def tournament_selection_N(self, num, rank, dists_vec, prob_ts, k=2, n_jobs=-1):
#        """
#        Select num points.
#        k: scalar, number of points to be randomly selected from the population.
#        """
#        
#        pjob = Parallel(n_jobs=n_jobs, verbose=0)
#        selected_inds = pjob( delayed(self.tournament_selection)(rank, dists_vec, prob_ts, k) for n in range(num) )
#        return selected_inds
    
    def tournament_selection_N(self, num, rank, dists_vec, prob_ts, k=2, n_jobs=38):
        """
        Select num points.
        k: scalar, number of points to be randomly selected from the population.
        """
        selected_inds = [ self.tournament_selection(rank, dists_vec, prob_ts, k) for n in range(num) ]
        return selected_inds
    
    
    def crossover(self, point1, point2, method='linear'):
        """
        point1, point2: vector of size K, two data points.
        """
        K = point1.size
        
        if method == 'linear':
            d=0.25
            alpha = np.random.rand()
            alpha = -d + (1+2*d) * alpha 
            new_point1 = point1 + alpha*(point2 - point1)
            alpha = np.random.rand()
            alpha = -d + (1+2*d) * alpha 
            new_point2 = point1 + alpha*(point2 - point1)
        elif method == 'discrete':
            alpha=np.random.randint(K)
            new_point1 = np.zeros(K, dtype=np.float32)
            new_point1[:alpha]=point1[:alpha]
            new_point1[alpha:]=point2[alpha:]
            #alpha=np.random.randint(K)
            new_point2 = np.zeros(K, dtype=np.float32)
            new_point2[:alpha]=point2[:alpha]
            new_point2[alpha:]=point1[alpha:]
            #temp = np.copy( point2[:alpha] )
            #point2[:alpha] = point1[:alpha]
            #point1[:alpha] = temp
            #temp = np.copy( point2[alpha:] )
            #point2[alpha:] = point1[alpha:]
            #point1[alpha:] = temp
            
        return new_point1,new_point2
        
    
    def mutate(self, point, mutation_rate=0.01):
        p = np.random.rand()
        #mutation
        if p<mutation_rate:
            pos = np.random.randint(point.size)
            point[pos] = point[pos] + np.random.randn()
    
    
    def save_population(self, g):
        """
        g: integer, generation index.
        """
        if g == self.config.get('num_generations')-1:
            g='final'
        new_rank = pd.DataFrame(self.new_rank, columns=['rank']) 
        data_to_save = pd.concat( [self.new_pop, new_rank], axis=1 )
        data_to_save = data_to_save.sort_values('rank')
        filename = self.config.path('samples_del') / f"new_pop_{g}.csv"
        data_to_save.to_csv(filename)


    def generate_sample(self):
        pass
    
    
    def get_properties(self, samples, n_jobs=38):
        info = get_dataset_info(self.config.get('dataset'))
        
        columns = ["smiles", "fragments", "n_fragments"]
        samples = pd.DataFrame(samples, columns=columns)
        
        samples = add_atom_counts(samples, info, n_jobs)
        samples = add_bond_counts(samples, info, n_jobs)
        samples = add_ring_counts(samples, info, n_jobs)

        # add same properties as in training/test dataset 
        for prop in info['properties']:
            samples = add_property(samples, prop, n_jobs)
        
        #samples.to_csv(config.path('samples') / 'aggregated.csv')
        return samples
    
    
    def fast_nondominated_sort( self, P ):
        """
        P is an numpy array of N by M where N is the number of data points / solutions, and M is the number is scores.
        
        Test code:
        import numpy as np
        import matplotlib.pyplot as plt

    P = 100*np.random.rand( 1000,2)
    rank = nondominated_sort(P)
    M = rank.max()
    for m in range(M):
        plt.plot(P[rank==m][:,0], P[rank==m][:,1], ls = '', marker ='o', markersize=4)
        
        plt.show()
        """
        N,M = P.shape
        
        inds_all_num = np.array( range(N) )
        
        Np = np.zeros(N, dtype=int) # number of solutions which dominate solution p
        rank = np.zeros(N, dtype=int)
        Sp = [] # set of solutions that p dominate
        Fs = []

        for n in range(N):
            diffs = P[n] - P 
            inds_le = ((diffs)<=0).all(axis=1)
            inds_l = ((diffs)<0).any(axis=1)
            inds = inds_le & inds_l            
            Sp.append ( inds_all_num[inds] )
            
            # >= & >
            #inds = ~inds_l & ~inds_le
            inds = ~(inds_l | inds_le)
            Np[n] = inds.sum()
        
        F=[]
        F = inds_all_num[Np == 0]
        rank[F] = 0
        
        i=0 # rank
        while len(F)>0:
            Fs.append(np.array(F))
            Q=[]
            for p in F:
                for q in Sp[p]:
                    Np[q] = Np[q] - 1
                    if Np[q] ==0:
                        rank[q] = i+1
                        Q.append(q)
            i = i + 1
            F = Q
            
        return rank, Fs
        
        
    def crowding_distance_assignment( self, I, f_min, f_max ):
        """
        I: Numpy array of N by M. It can be the property matrix for one front. 
        """
        
        N,M = I.shape
        dists= np.zeros( N, dtype=float )
        for m in range(M): # for each property
            inds = np.argsort( I[:,m] )
            dists[inds[0]] = np.inf
            dists[inds[-1]] = np.inf
            dists[inds[1:-1]] = dists[inds[1:-1]] + (I[inds[2:],m] - I[inds[0:-2],m])/(f_max[m] - f_min[m])
            
        return dists
    
    
#    def crowding_distance_all_fronts( self, P, Fs, f_min, f_max, n_jobs=-1 ):
#        """
#        P: properties.
#        Fs: fronts.
#        f_min: min values of propreties
#        f_max: max value of properties
#        """
#        pjob = Parallel(n_jobs=n_jobs, verbose=0)
#        dists_all = pjob(delayed(self.crowding_distance_assignment)(P[F], f_min, f_max) for F in Fs)
#        dists_vec = np.zeros(P.shape[0])
#        for F,D in zip(Fs,dists_all):
#            dists_vec[F] = D
#        return dists_all, dists_vec
       
 
    def crowding_distance_all_fronts( self, P, Fs, f_min, f_max, n_jobs=38 ):
        """
        P: properties.
        Fs: fronts.
        f_min: min values of propreties
        f_max: max value of properties
        """
        
        dists_all =[ self.crowding_distance_assignment(P[F], f_min, f_max) for F in Fs ]
        dists_vec = np.zeros(P.shape[0])
        for F,D in zip(Fs,dists_all):
            dists_vec[F] = D
        return dists_all, dists_vec
    

        
    
