import pandas as pd
import numpy as np
import plotly.graph_objects as go


class Graph3D:
    def __init__(self, axis=True):
        self.fig = go.Figure()
        self.axis_min = -5
        self.axis_max = 5

        if axis:
            self.fig.add_trace(
                go.Scatter3d(
                    x=[self.axis_min, self.axis_max],
                    y=[0, 0],
                    z=[0, 0],
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                )
            )
            self.fig.add_trace(
                go.Scatter3d(
                    x=[0, 0],
                    y=[self.axis_min, self.axis_max],
                    z=[0, 0],
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                )
            )
            self.fig.add_trace(
                go.Scatter3d(
                    x=[0, 0],
                    y=[0, 0],
                    z=[self.axis_min, self.axis_max],
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                )
            )

        self.fig.update_layout(
            title="Sentence Vectorization",
            scene=dict(
                xaxis=dict(
                    title="" if axis else "x",
                    showticklabels=axis,
                    range=[self.axis_min, self.axis_max],
                    showbackground=axis,
                ),
                yaxis=dict(
                    title="" if axis else "y",
                    showticklabels=axis,
                    range=[self.axis_min, self.axis_max],
                    showbackground=axis,
                ),
                zaxis=dict(
                    title="" if axis else "z",
                    showticklabels=axis,
                    range=[self.axis_min, self.axis_max],
                    showbackground=axis,
                ),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )

    def graph_vector(self, vector):
        x, y, z = vector

        self.fig.add_trace(
            go.Scatter3d(
                x=[0, x],
                y=[0, y],
                z=[0, z],
                mode="lines+markers",
                line=dict(width=6),
                marker=dict(size=4),
                name=f"v = ({x}, {y}, {z})",
            )
        )

        # length = np.sqrt(x**2 + y**2 + z**2)
        # if length == 0:
        #     cone_x, cone_y, cone_z = x, y, z
        # else:
        #     factor = 1.05
        #     cone_x = x * factor
        #     cone_y = y * factor
        #     cone_z = z * factor

        # self.fig.add_trace(
        #     go.Cone(
        #         x=[cone_x],
        #         y=[cone_y],
        #         z=[cone_z],
        #         u=[x],
        #         v=[y],
        #         w=[z],
        #         sizemode="absolute",
        #         sizeref=1,
        #         anchor="tip",
        #         showscale=False,
        #         colorscale=[[0, "black"], [1, "black"]],
        #     )
        # )

    def clear(self):
        self.fig.data = []
        self.fig.add_trace(
            go.Scatter3d(
                x=[self.axis_min, self.axis_max],
                y=[0, 0],
                z=[0, 0],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
        )
        self.fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[self.axis_min, self.axis_max],
                z=[0, 0],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
        )
        self.fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[self.axis_min, self.axis_max],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
        )
        self.fig.update_layout(
            title="Sentence Vectorization",
            scene=dict(
                xaxis=dict(
                    title="x",
                    showticklabels=True,
                    range=[self.axis_min, self.axis_max],
                    showbackground=True,
                ),
                yaxis=dict(
                    title="y",
                    showticklabels=True,
                    range=[self.axis_min, self.axis_max],
                    showbackground=True,
                ),
                zaxis=dict(
                    title="z",
                    showticklabels=True,
                    range=[self.axis_min, self.axis_max],
                    showbackground=True,
                ),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )


class Graph2D:
    def __init__(self, axis=True):
        self.fig = go.Figure()
        self.axis_min = -20
        self.axis_max = 20

        self.fig.update_layout(
            title="2-D Vectors (Plotly)",
            xaxis=dict(
                title="" if axis else "x",
                showticklabels=axis,
                range=[self.axis_min, self.axis_max],
            ),
            yaxis=dict(
                title="" if axis else "y",
                showticklabels=axis,
                range=[self.axis_min, self.axis_max],
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )

    def graph_vector(self, vector):
        x, y = vector
        self.fig.add_trace(
            go.Scatter(
                x=[0, x],
                y=[0, y],
                mode="lines+markers",
                line=dict(width=6),
                marker=dict(size=4),
                name=f"v = ({x}, {y})",
            )
        )
        length = np.sqrt(x**2 + y**2)
        if length == 0:
            cone_x, cone_y = x, y
        else:
            factor = 1.05
            cone_x = x * factor
            cone_y = y * factor
        # Draw an arrowhead using a marker
        self.fig.add_trace(
            go.Scatter(
                x=[cone_x],
                y=[cone_y],
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="black"),
                showlegend=False,
            )
        )
