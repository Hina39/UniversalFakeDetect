import base64
from io import BytesIO
from typing import Any

import dash  # type: ignore
import dash_bootstrap_components as dbc  # type: ignore
import numpy as np
import plotly.express as px  # type: ignore
import plotly.graph_objs as go  # type: ignore
from dash import dcc, html
from dash.dependencies import Input, Output  # type: ignore
from PIL import Image

from visualization.dimension_reduction import transform_umap

# For detail please check original docs:
# https://umap-learn.readthedocs.io/en/latest/parameters.html#metric
UMAP_METRICS = {
    "cosine",
    "correlation",
    "euclidean",
}


def numpy_to_b64(image: np.ndarray) -> str:
    """Convert image data from 0-16 to 0-255. This is a sample specific
    function.
    """
    im_pil = Image.fromarray(image)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return im_b64


def create_app(
    labels: np.ndarray,
    # predicted_labels: np.ndarray,
    features: np.ndarray,
) -> dash.Dash:
    """Create Dash application to make easy to see dashboard.

    Args:
        labels (np.ndarray): A np.ndarray of true label values. Its
            shape should be [batch].
        predicted_labels (np.ndarray): A np.ndarray of predicted label
            values. Its shape should be [batch].
        features (np.ndarray): A np.ndarray of extracted features. Its
            shape should be [batch, feature_dim].

    Returns:
        dash.Dash: A created Dash application.

    """
    app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])

    # Define modules which are displayed as sidebar.
    sidebar = html.Div(
        [
            dbc.Row(
                [
                    html.H5(
                        "でぃーぷとぅるーす☆がーでぃあんず",
                        style={
                            "margin-top": "20px",
                            "margin-left": "20px",
                        },
                    )
                ],
                style={"height": "5vh"},
                className="bg-primary text-white font-italic",
            ),
            dbc.Row(
                [
                    html.Div(
                        [
                            # metric
                            html.P(
                                "metric",
                                style={
                                    "margin-top": "8px",
                                    "margin-bottom": "4px",
                                },
                                className="font-weight-bold",
                            ),
                            dcc.Dropdown(
                                id="metric-picker",
                                multi=False,
                                value="cosine",
                                options=[
                                    {"label": x, "value": x} for x in UMAP_METRICS
                                ],
                                style={"width": "320px"},
                            ),
                            # n_neighbors
                            html.P(
                                "n_neighbors",
                                style={
                                    "margin-top": "8px",
                                    "margin-bottom": "4px",
                                },
                                className="font-weight-bold",
                            ),
                            dcc.Slider(
                                5,
                                100,
                                step=1,
                                value=5,
                                marks={
                                    5: "5",
                                    100: "100",
                                },
                                tooltip={
                                    "placement": "bottom",
                                    "always_visible": True,
                                },
                                id="n_neighbors-slider",
                            ),
                            # min_dist
                            html.P(
                                "min_dist",
                                style={
                                    "margin-top": "8px",
                                    "margin-bottom": "4px",
                                },
                                className="font-weight-bold",
                            ),
                            dcc.Slider(
                                0.0,
                                0.99,
                                step=0.01,
                                value=0.3,
                                marks={
                                    0.0: "0.0",
                                    0.99: "0.99",
                                },
                                tooltip={
                                    "placement": "bottom",
                                    "always_visible": True,
                                },
                                id="min_dist-slider",
                            ),
                        ]
                    )
                ],
                style={"height": "50vh", "margin": "8px"},
            ),
        ]
    )

    # Define modules which are displayed as main content.
    content = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        "UMAP Visualization",
                                        style={
                                            "margin-top": "16px",
                                            "margin-bottom": "4px",
                                        },
                                        className="font-weight-bold",
                                    ),
                                    dcc.Graph(
                                        id="dimension-reduction-graph",
                                        className="bg-light",
                                    ),
                                ]
                            ),
                        ]
                    )
                ],
                style={"height": "60vh", "margin": "16px"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4(
                                id="label",
                                className="font-weight-bold",
                            ),
                            html.H4(
                                id="predicted-label",
                                className="font-weight-bold",
                            ),
                        ]
                    )
                ],
                style={"height": "40vh", "margin": "16px"},
            ),
            # dbc.Row(
            #     [
            #         dbc.Col(
            #             [
            #                 html.Div(
            #                     [
            #                         html.H4(
            #                             "Original Image",
            #                             className="font-weight-bold",
            #                         ),
            #                         html.Div(id="image"),
            #                     ],
            #                 )
            #             ],
            #         ),
            #     ],
            #     style={
            #         "height": "50vh",
            #         "margin-top": "16px",
            #         "margin-left": "16px",
            #         "margin-bottom": "8px",
            #         "margin-right": "8px",
            #     },
            # ),
        ]
    )

    # Define total layout.
    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(sidebar, width=3, className="bg-light"),
                    dbc.Col(content, width=9),
                ]
            ),
        ],
        fluid=True,
    )

    # Define callback functions from here.
    @app.callback(
        Output("label", "children"),
        [Input("dimension-reduction-graph", "hoverData")],
    )
    def display_label(hover_data: Any) -> str:
        """Callback to change displaying label dynamically."""
        if hover_data:
            idx = hover_data["points"][0]["pointIndex"]
            return f"Label: {labels[idx]}"

        return "Label:"

    # @app.callback(
    #     Output("predicted-label", "children"),
    #     [Input("dimension-reduction-graph", "hoverData")],
    # )
    # def display_predicted_label(hover_data: Any) -> str:
    #     """Callback to change displaying predicted label dynamically."""
    #     if hover_data:
    #         idx = hover_data["points"][0]["pointIndex"]
    #         return f"Predicted label: {predicted_labels[idx]}"

    #     return "Predicted label:"

    @app.callback(
        Output("dimension-reduction-graph", "figure"),
        [
            Input("metric-picker", "value"),
            Input("n_neighbors-slider", "value"),
            Input("min_dist-slider", "value"),
        ],
    )
    def update_figure(metric: Any, n_neighbors: Any, min_dist: Any) -> go.Figure:
        """Callback to change displaying figure dynamically."""
        transformed_data = transform_umap(
            data=features,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric=metric,
        )

        figure = px.scatter(
            transformed_data,
            x=0,
            y=1,
            # color=[str(label) for label in labels.tolist()],
            color=labels,
            color_continuous_scale=px.colors.qualitative.Plotly,
            # color_continuous_scale=px.colors.sequential.Teal,
        )

        return figure

    return app


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="input_path", type=pathlib.Path, required=True)
    parser.add_argument("-n", dest="num_data", type=int, default=10000)
    args = parser.parse_args()

    data = np.load(args.input_path)

    print(f"data['features'][0].shape: {data['features'][0].shape}")

    app = create_app(
        data["labels"][: args.num_data],
        # data["predicted_labels"][: args.num_data],
        data["features"][: args.num_data],
    )
    app.run_server(host="0.0.0.0", port=8050, debug=True)
