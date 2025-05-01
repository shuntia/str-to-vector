from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Add this import


class Graph2D:
    def __init__(self):
        self.df = pd.DataFrame({"raw_texts": [], "embeddings": []})
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)

        self.ax.set_xlim([0, 10])
        self.ax.set_ylim([0, 10])

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.ax.set_title("3‑D Vectors (matplotlib quiver)")

        self.ax.plot([-1e6, 1e6], [0, 0], [0, 0], color="black")  # x-axis
        self.ax.plot([0, 0], [-1e6, 1e6], [0, 0], color="black")  # y-axis

    def graph_vector(self, vector):
        v = np.array([4, 1, 2])  #
        self.ax.quiver(0, 0, v[0], v[1], length=1, normalize=True, alpha=1.0)

    def graph(self):
        plt.show()


class Graph3D:
    def __init__(self):
        self.df = pd.DataFrame({"raw_texts": [], "embeddings": []})
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")  # 3D axes

        self.ax.set_xlim([0, 10])
        self.ax.set_ylim([0, 10])
        self.ax.set_zlim([0, 10])  # Add z limit

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")  # Add z label
        self.ax.set_title("3‑D Vectors (matplotlib quiver)")

        self.ax.plot([-1e6, 1e6], [0, 0], [0, 0], color="black")  # x-axis
        self.ax.plot([0, 0], [-1e6, 1e6], [0, 0], color="black")  # y-axis
        self.ax.plot([0, 0], [0, 0], [-1e6, 1e6], color="black")  # z-axis

    def graph_vector(self, vector):
        v = np.array([4, 1, 2])  # Example 3D vector
        self.ax.quiver(0, 0, 0, v[0], v[1], v[2], length=1, normalize=True, alpha=1.0)

    def graph(self):
        plt.show()
