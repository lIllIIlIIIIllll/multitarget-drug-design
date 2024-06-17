import numpy as np
import pandas as pd
import os

dataset = ['ZINC', "PCBA", 'ZINCMOSES']


def get_rank_sum(properties):
    # Properties are sorted as qed, SAS, logP
    # Another way to get the rank is to call argsort twice on the array
    # qed_rank = np.argsort(np.argsort(qed))
    # But this can be computationally expensive for large arrays
    # Using the method below, we can avoid the need for a double sort

    N = len(properties)
    sas, logp, ca9, gpx4 = properties[:, 0], properties[:, 1], properties[:, 2], properties[:, 3]

    # Get Orders of Properties
    sas_order, logp_order, ca9_order, gpx4_order = np.argsort(sas), np.argsort(logp), np.argsort(ca9), np.argsort(gpx4)

    # Get Ranks of Properties
    sas_rank, logp_rank, ca9_rank, gpx4_rank = \
        np.empty_like(sas_order), np.empty_like(logp_order), np.empty_like(ca9_order), np.empty_like(gpx4_order)
    sas_rank[sas_order], logp_rank[logp_order], ca9_rank[ca9_order], gpx4_rank[gpx4_order] = \
        np.arange(N), np.arange(N), np.arange(N)

    # Get Rank Sums
    ranks = np.array([np.sum([sas_rank[i], logp_rank[i], ca9_rank[i], gpx4_rank[i]]) for i in range(N)])
    return ranks


for d in dataset:
    path = f'../DATA/{d}/PROCESSED/train.smi'
    samples = pd.read_csv(path, index_col=0, skip_blank_lines=True)
    samples.dropna(how="all", inplace=True)
    properties = samples.loc[:, ['SAS', 'logP', 'CA9', 'GPX4']]
    # properties['qed'] = -properties['qed']
    properties = properties.to_numpy()

    rank = get_rank_sum(properties)

    # Add Rank to samples
    samples['rank'] = rank
    print(samples)
    # Save Rank
    save_path = f'../DATA/{d}/RANKED'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    samples.to_csv(f'{save_path}/train.csv')
    print(f'Saved Ranked {d} Data to {save_path}')
