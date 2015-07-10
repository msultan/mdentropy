import numpy as np
from scipy import stats
from scipy.special import psi
from mdentropy.utils import hist


def ent(nbins, r, *args):
    bins = hist(nbins, r, *args)
    return stats.entropy(bins)


def entc(nbins, r, *args):
    N = args[0].shape[0]
    bins = hist(r, *args)
    return np.sum(bins*(np.log(N)
                  - np.nan_to_num(psi(bins))
                  - ((-1)**bins/(bins + 1))))/N


def mi(nbins, X, Y, r=[-180., 180.]):
    return (entc(nbins, r, X)
            + entc(nbins, r, Y)
            - entc(nbins, r, X, Y))


def ce(nbins, X, Y, r=[-180., 180.]):
    return (entc(nbins, r, X, Y)
            - entc(nbins, r, Y))


def cmi(nbins, X, Y, Z, r=[-180., 180.]):
    return (entc(nbins, r, X, Z)
            + entc(nbins, r, Y, Z)
            - entc(nbins, r, Z)
            - entc(nbins, r, X, Y, Z))


def ncmi(nbins, X, Y, Z, r=[-180., 180.]):
    return (1 + (entc(nbins, r, Y, Z)
            - entc(nbins, r, X, Y, Z))/ce(nbins, X, Z, r=r))
