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
# pipe = joblib.load("umap_wikitext.pkl")
# pipe = joblib.load("pca_prefit.pkl")
# pipe = VectorPipeline(n_components=n_components)
df = pd.DataFrame({"raw_texts": [], "embeddings": []})


app.layout = html.Div(
    [
        dcc.Store(id="selected-traces", data=[]),
        dcc.Store(id="df-store", data=[]),
        dcc.Input(id="input-text", type="text", placeholder="Enter text to vectorize"),
        html.Button("Vectorize", id="vectorize-button", n_clicks=0),
        html.Button("Clear", id="clear-button", n_clicks=0),
        html.Br(),
        html.Button("Calculate similarity", id="similarity-button", n_clicks=0),
        html.Div(id="similarity-output"),
        html.Div(
            dcc.Graph(
                id="graph",
                figure=graph.fig,
                style={"height": "100vh", "width": "98vw"},
            ),
            style={"height": "100vh", "width": "98vw", "margin": 0, "padding": 0},
        ),
    ]
)

# Global redraw from state callback
@app.callback(
    Output("graph", "figure"),
    Input("df-store", "data"),
    Input("selected-traces", "data"),
)
def redraw(df, selected):
    df = pd.DataFrame(df)
    graph.clear()

    for id, row in df.iterrows():
        graph.graph_vector(row["embeddings"], row["raw_texts"])

    for i, tr in enumerate(graph.fig.data):
        tr.opacity = 1.0 if i in selected else 0.3

    graph.fig.update_layout(
        scene=dict(
            xaxis=dict(
                autorange=True,
            ),
            yaxis=dict(
                autorange=True,
            ),
            zaxis=dict(
                autorange=True,
            ),
        )
    )

    return graph.fig

# Clear button callback
@app.callback(
    Output("df-store", "data", allow_duplicate=True),
    Output("selected-traces", "data", allow_duplicate=True),
    Input("clear-button", "n_clicks"),
    prevent_initial_call=True,
)
def clear_df(n_clicks):
    return [], []

# Add vector callback
@app.callback(
    Output("df-store", "data", allow_duplicate=True),
    Input("vectorize-button", "n_clicks"),
    State("input-text", "value"),
    State("df-store", "data"),
    prevent_initial_call=True,
)
def add_vector(n_clicks, value, df):
    if not value:
        return df

    emb = pipe.vectorize([value])[0]
    df.append({"raw_texts": value, "embeddings": emb.tolist()})
    return df


# Toggle selection callback
@app.callback(
    Output("selected-traces", "data"),
    Input("graph", "clickData"),
    State("selected-traces", "data"),
    prevent_initial_call=True,
)
def toggle_selection(clickData, selected):
    idx = clickData["points"][0]["curveNumber"]
    if idx in selected:
        selected.remove(idx)
    else:
        selected.append(idx)
    return selected

# Similarity callback
@app.callback(
    Output("similarity-output", "children"),
    Input("similarity-button", "n_clicks"),
    State("df-store", "data"),
    State("selected-traces", "data"),
)
def compute_similarity(n_clicks, df, selected):
    if len(selected) != 2:
        return "Select exactly 2 vectors"
    
    if not n_clicks:
        return "Click calculate similarity"
    
    index_offset = 3
    vec1 = df[selected[0] - index_offset]["embeddings"]
    vec2 = df[selected[1] - index_offset]["embeddings"]

    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return f"Similarity: {similarity:.2f}"

if __name__ == "__main__":
    app.run(debug=True)
