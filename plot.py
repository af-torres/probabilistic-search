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
    ylim = round(np.max(y), 2) + .45
    ax.plot(x, y, linewidth=2.0, color="black")
    ax.set(
        xlim=(0, 1),
        xticks=np.arange(0, 1.125, 0.125),
        ylim=(0, ylim),
        yticks=np.linspace(0, ylim, 5),
        title=Jdescription
    )


def init(J: search.EvalFunc, Jdescription: str):
    fig, ax = plt.subplots()
    draw_base(ax, J, Jdescription)
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, ax


def draw(nodes: List[Node], fig: Figure, ax: Axes, J: search.EvalFunc, Jdescription: str, totalEvaluations: int):
    def get_liney(J):
        x, y = eval_J(J)
        return np.max(y) + .125
    def get_color(isMax, val):
        if val == 0: return "red"
        elif isMax : return "green"
        else: return "yellow"
    def get_fill_color(isMax, val):
        if val == 0: return "lightcoral"
        elif isMax: return "lightgreen"
        else: return "yellow"

    # clear previous plot
    ax.cla()
    draw_base(ax, J, Jdescription)

    prob = np.array(compare(nodes))
    maxIdx = np.argmax(prob)

    offset = 0.005
    liney = get_liney(J)
    for idx, val in enumerate(prob):
        node: Node = nodes[idx]
        low, high = node.range()
        fillx, filly = eval_J(J, low, high)

        color = get_color(idx == maxIdx, val)
        fillcolor = get_fill_color(idx == maxIdx, val)

        ax.plot([low + offset], [liney], marker=">", color=color, markersize=4)
        ax.plot([high - offset], [liney], marker="<", color=color, markersize=4)
        ax.axhline(liney, low + offset, high - offset, color=color)
        ax.fill_between(fillx, filly, facecolor=fillcolor)

    r = nodes[maxIdx].range()
    ax.text(x=.56, y=liney + 0.2, s=f"P(Min in [{round(r[0], 3):.3f}, {round(r[1], 3):.3f}]) = {round(prob[maxIdx], 2):.2f}")
    ax.text(x=.56, y=liney + 0.1, s=f"L(x) evaluations: {totalEvaluations}")

    fig.canvas.draw()
    fig.canvas.flush_events()
