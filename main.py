import argparse
from example_functions import examples
from distribution import Dist, Normal, Gamma
from node import Node
from graph import init_graph, draw
import search


class Config:
    sample_by: search.sample_algo
    sample_size: int

    def __init__(self, sample_by: search.sample_algo, sample_size: int):
        self.sample_by = sample_by
        self.sample_size = sample_size


def main(config: Config):    
    # init search space = [0, 1] with prior J ~ N(u, t),
    # where u ~ N(0,1) and t ~ inverse-gamma(0, 1)
    root = Node(Dist(Normal(0, 1), Gamma(2, 0.5)), (0., 1.))
    curr = [root]

    example = examples.get("example1")
    J: search.EvalFunc = example.get("function") # function over which we are minimizing
    Jname: str = example.get("name") # human readable name for J
    sample_size = config.sample_size # samples from J drawn on every iteration
    sample_algo = config.sample_by
    tot_J_evaluations = 0 # this is a "proxy" to the search algorithm efficiency as time for time complexity
    
    # plot
    fig, ax = init_graph(J, Jname)
    while True:
        arg = input("continue? ('n' to exit): ")
        if arg == "n":
            break
        
        draw(curr, fig, ax, J, Jname)
        curr = search.step(curr, J, sample_algo, sample_size)
        tot_J_evaluations += sample_size


def parse_config() -> Config:
    parser = argparse.ArgumentParser(
                    prog="Probabilistic Search",
                    description="""A search algorithm inspired by binary search that can be used
to find the minimum of non differentiable functions with respect to their arguments.
""")
    parser.add_argument("-a", "--sample_algo",
                        default="proportional",
                        choices=["proportional", "constant"])
    parser.add_argument("-s", "--sample_size",
                        default=100, type=int)
    args = parser.parse_args()

    return Config(
        args.sample_algo,
        args.sample_size
    )


if __name__ == "__main__":
    config = parse_config()
    main(config)
