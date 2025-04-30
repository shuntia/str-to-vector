import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotlyGraph import Graph3D
from vectorizer import Vectorizer

app = dash.Dash(__name__)
graph3D = Graph3D(True)

app.layout = html.Div(
    [
        html.Button("Test add vector", id="add-vector-button", n_clicks=0),
        html.Div(
            [
                dcc.Graph(
                    id="3d-graph",
                    figure=graph3D.fig,
                    style={"height": "100vh", "width": "100vw"},
                )
            ],
            style={
                "height": "100vh",
                "width": "100vw",
                "margin": 0,
                "padding": 0,
                "overflow": "hidden",
            },
        ),
    ]
)


@app.callback(
    Output("3d-graph", "figure"),
    Input("add-vector-button", "n_clicks"),
    State("3d-graph", "figure"),
    prevent_initial_call=True,
)
def add_test_vector(n_clicks, current_fig):
    if n_clicks:
        graph3D.graph_vector([12, 4, 7])
        graph3D.fig.update_layout(uirevision="constant")
    return graph3D.fig


def main():
    df = pd.DataFrame({"raw_texts": [], "embeddings": []})
    graph3D = Graph3D(True)
    vectorizer = Vectorizer()
    text = input("Enter text to vectorize: ").splitlines()
    vectorized = vectorizer.vectorize(text)
    vectorized_df = pd.DataFrame(
        {"raw_texts": text, "embeddings": [vec for vec in vectorized.toarray()]}
    )
    df = pd.concat([df, vectorized_df], axis=0, ignore_index=True)
    print(df.head())

    graph3D.graph_vector([12, 4, 7])


if __name__ == "__main__":
    app.run(debug=True)
