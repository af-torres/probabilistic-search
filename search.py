import math
import numpy as np

from typing import Callable, List
from node import Node, compare


def sampleParams(nodes: List[Node], size: int) -> List[List[float]]:
    """
    sampleParams generates a set of uniformly distributed samples for each Node.
    The number of samples generated for each node is determined by its probability
    of containing the value of interest within its region.

    Args:
        nodes (List[Node]): A list of Node objects representing regions.
        size (int): The total number of samples to be generated across all nodes.

    Returns:
        List[List[float]]: A list of lists where each inner list contains the uniformly
            distributed samples generated for a specific node.
    """
    probs = compare(nodes)
    section_size = list(
        map(lambda p: math.ceil(size * p), probs)
    )
    samples = []
    for idx, node in enumerate(nodes):
        size = section_size[idx]
        a, b = node.range()
        samples.append(np.random.uniform(low=a, high=b, size=size))

    return samples


def sampleError(
        evalFunction: Callable[[float], float],
        params: List[List[float]],
    ) -> List[List[np.ndarray]]:
    """
    sampleError generates a set of samples for each node, evaluates each sample
    using the provided evalFunction, and returns a list of lists containing the
    errors for each node's samples.

    Returns:
        List[List[float]]: A list of lists where each inner list contains the errors
            for the samples of a specific node.
    """
    samples = list(
        map(
            lambda ps: list(
                map(
                    lambda p: evalFunction(p),
                    ps
                )
            ),
            params
        )
    )

    return samples


def updateNodes(nodes: List[Node], samples: List[List[np.ndarray]]) -> List[Node]:
    updated = []
    for idx, node in enumerate(nodes):
        updated.append(node.updateDistribution(samples[idx]))

    return updated


def step(
        nodes: List[Node],
        evalFunction: Callable[[float], float],
        sample_size=30,
    ) -> List[Node]:
    nodes = updateNodes(
        nodes,
        sampleError(
            evalFunction,
            sampleParams(nodes, sample_size)
        )
    )

    next = []
    for node in nodes:
        next = [*next, *node.step()]
    
    return next
