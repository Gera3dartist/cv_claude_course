import numpy as np
import matplotlib.pyplot as plt



def plot_function(
    x: np.ndarray, 
    y: np.ndarray, 
    title: str, 
    x_title: str | None = None, 
    y_title: str | None = None
):
    fig, ax = plt.subplots()
    ax.plot(x, y, markeredgewidth=1)
    ax.set_title(title)
    if y_title:
        ax.set_ylabel(y_title)
    if x_title:
        ax.set_xlabel(x_title)

    plt.show()

def plot_multiple(pairs: list[tuple[np.ndarray, np.ndarray]], title: str):
    fig, ax = plt.subplots()
    for x, y in pairs:
        ax.plot(x, y, markeredgewidth=1)
    ax.set_title(title)
    # if y_title:
    #     ax.set_ylabel(y_title)
    # if x_title:
    #     ax.set_xlabel(x_title)
    plt.show()