import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def save_error_map(error_map: np.ndarray, path: str):

    print("Saving error map...")
    plt.subplots(figsize=(12, 10), tight_layout=True)
    ax = sns.heatmap(error_map, cmap='hot', vmin=0, vmax=5.0)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.savefig(path)
    plt.close()

    print("Done!")
