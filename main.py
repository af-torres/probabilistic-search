import argparse
from example_functions import examples
from distribution import Dist, Normal, Gamma
from node import Node
import search
import plot

class Config:
    sample_by: search.sample_algo
    sample_size: int

    def __init__(self, sample_by: search.sample_algo, sample_size: int):
        self.sample_by = sample_by
        self.sample_size = sample_size


def totalEvaluations(algo: search.sample_algo, sample_size: int, totNodes: int, iters: int) -> int:
    # this computes a "proxy" to the search algorithm efficiency as time for time complexity
    if algo == search.SampleAlgo.proportional.value:
        return iters * sample_size
    elif algo == search.SampleAlgo.constant.value:
        prev = iters > 1
        curr = iters > 0
        return curr * totNodes * sample_size  + (totNodes / 2 * sample_size) * prev # we double nodes on every iter

    return 0


def run(config: Config):    
    # init search space = [0, 1] with prior J ~ N(u, t),
    # where u ~ N(0,1) and t ~ inverse-gamma(0, 1)
    root = Node(Dist(Normal(0, 1), Gamma(2, 0.5)), (0., 1.))
    curr = [root]

    example = examples.get("example1")
    J: search.EvalFunc = example.get("function") # function over which we are minimizing
    Jname: str = example.get("name") # human readable name for J
    sample_size = config.sample_size # samples from J drawn on every iteration
    sample_algo = config.sample_by
    
    # plot
    fig, ax = plot.init(J, Jname)
    iters = 0
    while True:
        arg = input("continue? ('n' to exit): ")
        if arg == "n":
            break
        
        plot.draw(
            curr,
            fig, ax,
            J, Jname,
            totalEvaluations(sample_algo, sample_size, len(curr), iters)
        )
        curr = search.step(
            curr,
            J,
            sample_algo,
            sample_size
        )
        iters += 1


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
                        default=30, type=int)
    args = parser.parse_args()

    return Config(
        args.sample_algo,
        args.sample_size
    )


def main():
    config = parse_config()
    run(config)


if __name__ == "__main__":
    main()
