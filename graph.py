from typing import List
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from node import Node, compare
import search


def eval_J(J: search.EvalFunc, min=0, max=1):
    x = np.linspace(min, max, 100)
    y = list(map(lambda x: J(x), x))

    return x, y


def draw_base(ax, J:search.EvalFunc, Jdescription: str):
    x, y = eval_J(J)
    ax.plot(x, y, linewidth=2.0, color="black")
    ax.set(
        xlim=(0, 1),
        xticks=np.arange(0, 1.125, 0.125),
        ylim=(0, 1.5),
        yticks=np.arange(0, 2, 0.25),
        title=Jdescription
    )


def init_graph(J: search.EvalFunc, Jdescription: str):
    fig, ax = plt.subplots()
    draw_base(ax, J, Jdescription)
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, ax


def draw(nodes: List[Node], fig: Figure, ax: Axes, J: search.EvalFunc, Jdescription: str):
    def get_color(isMax, val):
        if val == 0: return "red"
        elif isMax : return "green"
        else: return "yellow"
    def get_fill_color(isMax, val):
        if val == 0: return "indianred"
        elif isMax: return "lightgreen"
        else: return "yellow"

    # clear previous plot
    ax.cla()
    draw_base(ax, J, Jdescription)

    prob = np.array(compare(nodes, 500))
    maxIdx = np.argmax(prob)
    orderedIdx = np.flip(np.argsort(prob))[:4] # ordered index by prob (decreasing)
    zeros = np.argwhere(prob == 0).flatten()
    paint = np.append(orderedIdx, zeros)

    offset = 0.005
    liney = 1.45
    for idx in paint:
        # TODO: add label of range and prob for min node
        idx = int(idx)
        val = prob[idx]
        node: Node = nodes[idx]
        low, high = node.range()
        fillx, filly = eval_J(J, low, high)

        color = get_color(idx == maxIdx, val)
        fillcolor = get_fill_color(idx == maxIdx, val)

        ax.plot([low + offset], [liney], marker=">", color=color, markersize=4)
        ax.plot([high - offset], [liney], marker="<", color=color, markersize=4)
        ax.axhline(liney, low + offset, high - offset, color=color)
        ax.fill_between(fillx, filly, facecolor=fillcolor)

    fig.canvas.draw()
    fig.canvas.flush_events()
