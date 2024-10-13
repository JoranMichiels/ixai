import os
import random

import numpy as np
import torch
from scipy.stats import spearmanr

HUMAN_SPEARMAN_CEILING = 0.65753  # See https://github.com/serre-lab/Harmonization


def get_file_path(file):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, file)


figure_folder = get_file_path('figures/')

checkpoints_folder = get_file_path('checkpoints/')


def normalize_tensor(tensor, method='gaussian', samplewise=False):
    if samplewise:
        dims = list(range(1, (tensor.dim())))
    else:
        dims = list(range(tensor.dim()))
    if method == 'gaussian':
        return (tensor - tensor.mean(dim=dims, keepdim=True)) / (tensor.std(dim=dims, keepdim=True) + 1e-8)
    if method == 'minmax':
        return (tensor - tensor.amin(dim=dims, keepdim=True)) / (
                tensor.amax(dim=dims, keepdim=True) - tensor.amin(dim=dims, keepdim=True) + 1e-8)
    else:
        raise ValueError('Unknown normalization method')


def spearman_correlation(x, y, normalize_human=True):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    scores = []

    for xi, yi in zip(x, y):
        rho, _ = spearmanr(xi.flatten(), yi.flatten())

        scores.append(rho)

    if normalize_human:
        return np.mean(np.array(scores)) / HUMAN_SPEARMAN_CEILING
    return np.array(scores)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_gen():
    g = torch.Generator()
    g.manual_seed(torch.initial_seed())
    return g
