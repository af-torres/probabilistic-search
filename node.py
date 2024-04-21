from __future__ import annotations

from typing import List, Tuple
from distribution import Dist, compareDist
import numpy as np

class Node:
    __region: Tuple[float, float]
    __dist: Dist

    def __init__(self, dist: Dist, region: Tuple[float, float]):
        self.__dist = dist
        self.__region = region

    def distribution(self) -> Dist: return self.__dist

    def updateDistribution(self, samples: np.ndarray) -> Node:
        return Node(self.__dist.posterior(samples), self.__region)

    def step(self) -> Tuple[Node, Node]:
        lowerLimit = self.__region[0]
        upperLimit = self.__region[1]
        pivot = (lowerLimit + upperLimit) / 2 # midpoint
        
        return (
            Node(self.distribution(), (lowerLimit, pivot)),
            Node(self.distribution(), (pivot, upperLimit))
        )
    
    def range(self) -> Tuple[float, float]: return self.__region


def compare(nodes: List[Node], sample_size=30)-> List[float]:
    """
    compare is a convenient wrapper for compareDist
    """
    dist = []
    for n in nodes:
        dist.append(n.distribution())

    return compareDist(dist, sample_size)