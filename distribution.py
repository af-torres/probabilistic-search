from __future__ import annotations

import math
from typing import List
import numpy as np

class Normal:
    loc: float
    scale: float # Variance

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def sample(self, size: tuple | int) -> np.ndarray:
        return np.random.normal(loc=self.loc, scale=math.sqrt(self.scale), size=size)


class Gamma:
    shape: float
    scale: float

    def __init__(self, shape: float, scale: float):
        self.shape = shape
        self.scale = scale

    def sample(self, size: tuple | int) -> np.ndarray:
        return np.random.gamma(shape=self.shape, scale=self.scale, size=size)


class Dist:
    __mean: Normal
    __precision: Gamma

    def __init__(self, mean: Normal, precision: Gamma):
        self.__mean = mean
        self.__precision = precision

    def posterior(self, samples: np.ndarray, gibbs_sample_size=10000, burn=3000) -> Dist:
        def sample_posterior_precision(samples: np.ndarray, mean: float) -> float:
            shape = self.__precision.shape + len(samples) / 2
            scale = self.__precision.scale + np.sum(
                np.square(samples - mean)
            ) / 2

            return np.random.gamma(shape=shape, scale=scale, size=1)[0]

        def sample_posterior_mean(samples: np.ndarray, var: float) -> float:            
            m = len(samples)
            precision = (m / var) + (1 / self.__mean.scale)
            mean = (
                (np.mean(samples) * (m / var)) + (self.__mean.loc / self.__mean.scale)
            ) / precision
            
            return np.random.normal(loc=mean, scale=math.sqrt(1/precision), size=1)[0]
        
        # gibbs sampler
        posterior_sample_mean = []
        posterior_sample_precision = []
        precision: float = self.__precision.sample(1)[0] # markov chain initialization var
        for i in range(0, gibbs_sample_size):
            mean = sample_posterior_mean(samples, 1/precision)
            precision = sample_posterior_precision(samples, mean)

            if i < burn: # we skip some values to avoid adding inadequate samples from the chain
                continue

            posterior_sample_mean.append(mean)
            posterior_sample_precision.append(precision)

        # sum(xi) / n -> E[X] = u
        # sum((xi - E[X])^2) / (n-1) -> Var[X] = sigma2
        posterior_mean = Normal(
            loc=np.mean(posterior_sample_mean),
            scale=np.var(posterior_sample_mean)
        )

        # sum(pi) / n -> E[P] = shape * scale
        # sum((pi - E[P])^2) / (n-1) -> Var[P] = shape * scale^2
        E_p = np.mean(posterior_sample_precision)
        Var_p = np.var(posterior_sample_precision)
        scale = Var_p / E_p
        shape = E_p / scale
        posterior_precision = Gamma(shape=shape, scale=scale)

        return Dist(posterior_mean, posterior_precision)

    def sample(self, size: tuple | int) -> np.ndarray:
        # Y_pred ~ Normal(posterior_mean, posterior_mean_var + posterior_var)
        #   = Normal(posterior_mean, posterior_mean_var) + Normal(0, posterior_var)
        Y_pred = self.__mean.sample(size) + (
            np.random.normal(loc = 0, scale = 1 / self.__precision.sample(size))
        )
        return Y_pred


def compareDist(nodes: List[Dist], sample_size=30) -> List[float]:
    """
    compare generates samples from the provided nodes and computes the probability
    of finding the global minimum within each node children links
    """
    n = len(nodes)
    
    samples = np.ndarray((sample_size, n))
    for i, node in enumerate(nodes):
        samples[:, i] = node.sample(sample_size)
    
    probs = []
    nrange = np.arange(n)
    for curr in range(0, n):
        pivot = samples[:, curr].reshape((sample_size, 1))
        other = samples[:, nrange != curr]
        
        tot = np.logical_and.reduce(pivot < other, axis = 1)
        p = np.sum(tot) / sample_size
        probs.append(p)
    
    return probs
