from __future__ import print_function

import time
import numpy as np
import pandas as pd
import mdtraj as md


class timing(object):
    "Context manager for printing performance"
    def __init__(self, iter):
        self.iter = iter

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, ty, val, tb):
        end = time.time()
        print("Round %d : %0.3f seconds" %
              (self.iter, end-self.start))
        return False


def adaptive(r=[-180, 180], *args):
    nbins = int(np.sqrt(args[0].shape[0]/5))
    d = np.cumsum(nbins*[1./nbins])[:-1]**len(args)
    sargs = tuple(np.sort(x) for x in np.vstack(tuple(args)))
    p = np.product([(x-x.min()).cumsum()/(x-x.min()).sum()
                    for x in np.vstack(sargs)], 0)
    ind = [sum(p < i) - 1 for i in d]
    bins = [np.hstack((r[0], x[ind], r[-1])) for x in sargs]
    return bins


def hist(r, *args):
    data = np.vstack((args)).T
    bins = adaptive(r, *args)
    return np.histogramdd(data, bins=bins)[0].flatten()


def shuffle(df, n=1):
    sdf = df.copy()
    sampler = np.random.permutation
    for i in range(n):
        sdf = sdf.apply(sampler, axis=0)
        sdf = sdf.apply(sampler, axis=1)
    return sdf


class Dihedrals(object):
    def __call__(self, traj):
        atoms, angles = self.method(traj)
        idx = [traj.topology.atom(i).residue.index
               for i in atoms[:, self.type]]
        return pd.DataFrame(180*angles/np.pi, columns=idx)

    def __init__(self, method, type):
        assert type < 3 or type > -1
        self.type = type
        self.method = method


def dihedrals(traj):
    kinds = [
        Dihedrals(md.compute_phi, 2),
        Dihedrals(md.compute_psi, 1),
        Dihedrals(md.compute_chi1, 0),
        Dihedrals(md.compute_chi2, 1)
        ]
    return [kind(traj) for kind in kinds]
