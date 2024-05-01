from enum import Enum
import logging
import math
import numpy as np

from typing import Callable, List, Literal
from node import TAINT_DO_NOT_SPLIT, Node, compare


class SampleAlgo(Enum):
    proportional = "proportional"
    constant = "constant"

sample_algo = Literal[SampleAlgo.proportional, SampleAlgo.constant]

SampleFunc = Callable[[List[Node], int],  List[List[float]]]

EvalFunc = Callable[[float], float]


def sampleAlgoFromEnum(name: sample_algo) -> SampleFunc:
    if name == SampleAlgo.proportional.value: return propSample
    elif name == SampleAlgo.constant.value: return constSample
    else: raise Exception("Invalid Sample Algorithm")


def constSample(nodes: List[Node], size: int) -> List[List[float]]:
    samples = []
    for node in nodes:
        size = size
        a, b = node.range()
        samples.append(np.random.uniform(low=a, high=b, size=size))

    return samples


def propSample(nodes: List[Node], size: int) -> List[List[float]]:
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
    logging.info("total nodes with sufficient samples: %s", np.sum((np.array(probs) * size) > 2) )
    section_size = list(
        map(lambda p: math.ceil(size * p), probs) 
    )
    samples = []
    for idx, node in enumerate(nodes):
        size = section_size[idx]
        a, b = node.range()
        samples.append(np.random.uniform(low=a, high=b, size=size))

    return samples


def eval(
        evalFunction: Callable[[float], float],
        params: List[List[float]],
    ) -> List[List[float]]:
    """
    eval generates a set of samples for each node, evaluates each sample
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
    """
    Update the posterior distribution of a list of nodes based on provided samples.
    The 'samples' list is expected to be parallel to the 'nodes' list, meaning that the samples
    in samples[i] are specifically intended for updating the posterior distribution of the node nodes[i].

    If a node has less than two samples, it is marked with the TAINT_DO_NOT_SPLIT flag and excluded from updating
    to avoid collapses in the variance (the variance of one sample is zero).

    Returns:
    - updated (List[Node]): A list of Node objects with updated posterior distributions.
    """
    updated = []

    for idx, node in enumerate(nodes):
        s = samples[idx]
        if len(s) < 2:
            logging.warn("sample size < 2")
            node.taint(TAINT_DO_NOT_SPLIT, True)
            updated.append(node)
            continue
        
        updated.append(node.updateDistribution(s))

    return updated


def step(
        nodes: List[Node],
        evalFunction: Callable[[float], float],
        sample_algo: sample_algo=SampleAlgo.proportional,
        sample_size=30,
    ) -> List[Node]:
    sfunc = sampleAlgoFromEnum(sample_algo)
    
    params = sfunc(nodes, sample_size)
    samples = eval(
            evalFunction,
            params
    )
    nodes = updateNodes(
        nodes,
        samples
    )

    next = []
    for node in nodes:
        # when we do not have enough information on a node rage to generate
        # a posterior distribution, we avoid creating a new partition over the space
        if node.getTaint(TAINT_DO_NOT_SPLIT)[0] == True:
            next = [*next, node]
            continue

        next = [*next, *node.step()]
    
    return next
