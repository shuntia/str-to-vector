import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from vector_pipeline import VectorPipeline
import joblib

USE_3D = True

if USE_3D:
    from plotly_graph import Graph3D as Graph

    n_components = 3
else:
    from plotly_graph import Graph2D as Graph

    n_components = 2

app = dash.Dash(__name__)
graph = Graph(True)
pipe = joblib.load("umap_prefit2.pkl")
# pipe = VectorPipeline(n_components=n_components)
df = pd.DataFrame({"raw_texts": [], "embeddings": []})


app.layout = html.Div(
    [
        dcc.Input(
            id="input-text",
            type="text",
            placeholder="Enter text to vectorize",
        ),
        html.Button("Vectorize", id="vectorize-button", n_clicks=0),
        html.Button("Clear", id="clear-button", n_clicks=0),
        html.Div(
            [
                dcc.Graph(
                    id="graph",
                    figure=graph.fig,
                    style={"height": "100vh", "width": "100vw"},
                ),
            ],
            style={
                "height": "100vh",
                "width": "100vw",
                "margin": 0,
                "padding": 0,
            },
        ),
    ]
)


@app.callback(
    Output("graph", "figure"),
    [Input("clear-button", "n_clicks"), Input("vectorize-button", "n_clicks")],
    [State("input-text", "value")],
    prevent_initial_call=True,
)
def update_graph(clear_clicks, vectorize_clicks, value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return graph.fig
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    global df
    if button_id == "clear-button":
        graph.clear()
        df = pd.DataFrame({"raw_texts": [], "embeddings": []})
        return graph.fig
    elif button_id == "vectorize-button" and value:
        vectorized = pipe.vectorize([value])
        row = pd.DataFrame({"raw_texts": [value], "embeddings": [vectorized[0]]})
        df = pd.concat([df, row], ignore_index=True)
        graph.graph_vector(vectorized[0])
        graph.fig.update_layout(uirevision="constant")
        print(df.head())
        return graph.fig
    return graph.fig


if __name__ == "__main__":
    app.run(debug=True)
