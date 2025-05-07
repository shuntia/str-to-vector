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
graph.fig.update_layout(clickmode="event+select", uirevision="constant")
pipe = joblib.load("umap_prefit2.pkl")
# pipe = VectorPipeline(n_components=n_components)
df = pd.DataFrame({"raw_texts": [], "embeddings": []})


app.layout = html.Div(
    [
        dcc.Store(id="selected-traces", data=[]),
        dcc.Input(
            id="input-text",
            type="text",
            placeholder="Enter text to vectorize",
        ),
        html.Button("Vectorize", id="vectorize-button", n_clicks=0),
        html.Button("Clear", id="clear-button", n_clicks=0),
        html.Br(),
        html.Button("Calculate similarity", id="similarity-button", n_clicks=0),
        html.Div(id="similarity-output"),
        html.Div(
            [
                dcc.Graph(
                    id="graph",
                    figure=graph.fig,
                    style={"height": "100vh", "width": "98vw"},
                ),
            ],
            style={
                "height": "100vh",
                "width": "98vw",
                "margin": 0,
                "padding": 0,
            },
        ),
    ]
)


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    [Input("clear-button", "n_clicks"), Input("vectorize-button", "n_clicks")],
    [State("input-text", "value"), State("selected-traces", "data")],
    prevent_initial_call=True,
)
def update_graph(clear_clicks, vectorize_clicks, value, selected):
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
        graph.graph_vector(vectorized[0], df.iloc[-1]["raw_texts"])
        graph.fig.update_layout(uirevision="constant")
        print(df.head())
    
    for i, tr in enumerate(graph.fig.data):
        tr.opacity = 1.0 if (selected and i in selected) else 0.2

    return graph.fig


@app.callback(
    [
        Output("graph", "figure", allow_duplicate=True),
        Output("selected-traces", "data"),
    ],
    Input("graph", "clickData"),
    [State("graph", "figure"), State("selected-traces", "data")],
    prevent_initial_call=True,
)
def select_trace(clickData, figure, selected):
    idx = clickData["points"][0]["curveNumber"]
    # toggle in store
    if idx in selected:
        selected.remove(idx)
    else:
        selected.append(idx)

    # restyle all traces
    for i, tr in enumerate(figure["data"]):
        tr["opacity"] = 1.0 if i in selected else 0.2

    # figure["layout"]["uirevision"] = "constant"

    return figure, selected


if __name__ == "__main__":
    app.run(debug=True)
