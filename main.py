from distribution import Dist, Normal, Gamma
from node import Node, compare

def main():
    node0 = Node(Dist(Normal(0,1), Gamma(1,1)), (0., 1.))
    node1, node2 = node0.step()
    
    print(compare([node1, node2]))

    samplesNode1 = Normal(3, 1).sample(30)
    node1.updateDistribution(samplesNode1)

    samplesNode2 = Normal(2,1).sample(30)
    node2.updateDistribution(samplesNode2)

    node3, node4 = node1.step()
    node5, node6 = node2.step()
    print(compare([node3, node4, node5, node6]))


if __name__ == "__main__":
    main()
