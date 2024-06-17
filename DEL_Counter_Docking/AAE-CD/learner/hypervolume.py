# Author: Karl Grantham
# 2022-03-26

import os
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from botorch.utils.multi_objective.hypervolume import Hypervolume

dtype = torch.double
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
print("device", device)

PROJ_DIR = Path(".")

# use_rbf = False
use_rbf = True

run_dirs = [

    # Windows
    # f'../RUNS/{directory}' for directory in os.listdir("../RUNS")
    # '../DATA/ZINC/RANKED'
    '../DATA/PCBA/RANKED'

    # Linux
    # PROJ_DIR / directory for directory in os.listdir(str(PROJ_DIR))
]


def rank_by_fronts(samples):
    samples = samples.drop(columns=['rank'])
    properties = samples.loc[:, ['qed', 'SAS', 'logP']]
    properties['qed'] = -properties['qed']
    properties = properties.to_numpy()
    p_min = np.min(properties, axis=0)
    p_max = np.max(properties, axis=0)

    rank, Fs = fast_nondominated_sort(properties)
    dists_all, dists_vec = crowding_distance_all_fronts(properties, Fs, p_min, p_max, n_jobs=-1)

    population_size = len(rank)
    new_pop = []
    new_rank = []
    count = 0
    i = 0
    # print(f'Number of Fronts: {len(Fs)}')
    #     Add Fronts 1, 2, 3, ... iteratively
    while (count + len(Fs[i])) <= population_size:
        new_pop.append(samples.loc[Fs[i], :])
        new_rank.append(rank[Fs[i]])
        count = count + len(Fs[i])
        i = i + 1
        if i >= len(Fs):
            break

        # put part of front i in new_pop
    if count < population_size:
        inds = np.argsort(dists_all[i])
        inds = inds[::-1]
        inds = inds[0:(population_size - count)]
        new_pop.append(samples.loc[Fs[i][inds], :])
        new_rank.append(rank[Fs[i][inds]])

    new_pop = pd.concat(new_pop, ignore_index=True)
    # print('The shape of new pop is', new_pop.shape)
    new_rank = np.concatenate(new_rank)
    # print('The shape of new rank is', new_rank.shape)
    new_rank = pd.DataFrame(new_rank, columns=["rank"])
    return pd.concat([new_pop, new_rank], axis=1)


def fast_nondominated_sort(P):
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


def crowding_distance_assignment(I, f_min, f_max):
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


def crowding_distance_all_fronts(P, Fs, f_min, f_max, n_jobs):
    '''
    P: Properties
    Fs: fronts
    f_min: min values of properties
    f_max: max values of properties
    '''

    dists_all = [crowding_distance_assignment(P[F], f_min, f_max) for F in Fs]
    dists_vec = np.zeros(P.shape[0])
    for F, D in zip(Fs, dists_all):
        dists_vec[F] = D
    return dists_all, dists_vec


count = 0
for run_dir in run_dirs[:1]:
    # print(run_dir)

    # Windows
    # finalGenStr = run_dir + '/results/samples_del/new_pop_final.csv'
    finalGenStr = run_dir + '/train.csv'

    # Linux
    # finalGenStr = run_dir / 'results/samples_del/new_pop_final.csv'

    # print('Final Gen String', finalGenStr)

    count = count + 1
    # print("New Count", count)

    finalGenData = pd.read_csv(
        finalGenStr,
        index_col=0,
        skip_blank_lines=True
    )

    # properties = ['qed', 'SAS', 'logP', 'rank']
    properties = ['qed', 'SAS', 'logP']
    # print("properties", properties)

    original_data_length = len(finalGenData)
    print("Number of Rows:", original_data_length)
    # finalGenData.dropna(inplace=True)
    finalGenData.dropna(subset=properties, inplace=True)
    finalGenData.reset_index(inplace=True, drop=True)
    post_dropna_data_length = len(finalGenData)
    # print("Number of Non-Null Samples:", post_dropna_data_length)
    # print("final gen data", finalGenData)

    finalGenData = finalGenData.head(200000)

    if use_rbf:
        finalGenData = rank_by_fronts(finalGenData)
        # print('After Rank By Fronts (Head)', finalGenData.head(5))
        # print('After Rank By Fronts (Tail)', finalGenData.tail(5))
    finalGenData = finalGenData.loc[finalGenData['rank'] == 0, :]
    front_zero_count = len(finalGenData)
    # print("Number of Front 0 Data Points:", front_zero_count)
    # print("final gen data min rank", finalGenData)

    # properties = ['qed', 'SAS', 'logP', 'rank']
    properties = ['qed', 'SAS', 'logP']
    # print("properties", properties)

    finalGenData = finalGenData.loc[:, properties]
    # print("final gen data loc properties", finalGenData)

    # finalGenData['qed'] = -finalGenData['qed']
    finalGenData['SAS'] = -finalGenData['SAS']
    finalGenData['logP'] = -finalGenData['logP']

    # print("final gen data -sas, -logp", finalGenData)

    # QED SAS logP, rank
    # ref_point = [qedMin[0], sasMin[0], logpMin[0], scoreMin[0]]
    # ref_point = [qedMin[0], sasMin[0], logpMin[0]]
    ref_point = [0, -10, -8.2521]
    # print("Reference Point", ref_point)

    hv = Hypervolume(ref_point=torch.tensor(ref_point, dtype=dtype, device=device))
    # print("hyper volume", hv)

    # pls = torch.tensor(ref_point, dtype=dtype, device=device)
    # print("pls", pls)

    torch_tensor10 = torch.tensor(finalGenData.values, dtype=dtype, device=device)
    # print("torch tensor", torch_tensor10)

    volume10 = hv.compute(torch_tensor10)
    # print("volume", volume10)

    print(f'{run_dir} Generation (Final): {volume10}')
    print(f'{str(run_dir)} ({front_zero_count}/{post_dropna_data_length}/{original_data_length}): {volume10:.4f}')
