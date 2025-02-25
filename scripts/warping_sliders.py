import dash
import dash_core_components as dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pickle
from src.utils import CanonPart, CanonPartMetadata


def viz_whole_model(warp_file):
    # You should be able to put any trained warp model in here
    warp_model = pickle.load(open(warp_file, "rb"))

    n_components = len(warp_model.pca.components_)
    num_ticks = 20

    latent_means = warp_model.pca.mean_
    sample_range = 5

    app = dash.Dash(__name__)

    dashboard_features = []
    for i in range(n_components):
        dashboard_features += [
            html.P(f"Component {i}:"),
            dcc.Slider(
                latent_means[i] - sample_range,
                latent_means[i] + sample_range,
                sample_range / (num_ticks / 2),
                value=latent_means[i],
                id=f"Component_{i}",
            ),
        ]

    dashboard_features += [
        dcc.Graph(id="Reconstruction"),
    ]
    app.layout = html.Div(dashboard_features)

    colorscale = "Viridis"

    @app.callback(
        Output("Reconstruction", "figure"),
        [Input(f"Component_{i}", "value") for i in range(n_components)],
    )
    def generate_chart(*args):
        component_values = [*args]
        warp = warp_model.pca.inverse_transform(component_values)
        v = warp_model.canonical_pcl + warp.reshape(-1, 3)

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                marker={"size": 5, "color": v[:, 2], "colorscale": colorscale},
                mode="markers",
                opacity=1.0,
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    range=[-0.1, 0.1],
                ),
                yaxis=dict(
                    nticks=4,
                    range=[-0.1, 0.1],
                ),
                zaxis=dict(
                    range=[-0.1, 0.1],
                ),
            ),
        )

        fig.update_layout(scene_aspectmode="cube")

        return fig

    app.run_server(debug=True)


def viz_part_models(warp_files, part_names):
    # You should be able to put any trained warp model in here
    part_models = {}
    n_components = {}
    part_means = {}
    for warp_file, part_name in zip(warp_files, part_names):
        warp_model = pickle.load(open(warp_file, "rb"))
        part_models[part_name] = warp_model
        n_components[part_name] = len(warp_model.pca.components_)
        part_means[part_name] = warp_model.pca.mean_

    num_ticks = 20
    sample_range = 5

    app = dash.Dash(__name__)

    dashboard_features = []
    past_num = 0
    for part in part_names:
        for i in range(n_components[part]):
            dashboard_features += [
                html.P(f"Component {i}:"),
                dcc.Slider(
                    part_means[part][i] - sample_range,
                    part_means[part][i] + sample_range,
                    sample_range / (num_ticks / 2),
                    value=part_means[part][i],
                    id=f"Component_{past_num + i}",
                ),
            ]
        past_num += n_components[part]

    dashboard_features += [
        dcc.Graph(id="Reconstruction"),
    ]
    app.layout = html.Div(dashboard_features)

    colorscale = "Viridis"

    inputs = []
    past_num = 0
    for part in part_names:
        inputs += [
            Input(f"Component_{past_num + i}", "value")
            for i in range(n_components[part])
        ]
        past_num += n_components[part]

    @app.callback(
        Output("Reconstruction", "figure"),
        inputs,
    )
    def generate_chart(*args):
        fig = make_subplots(rows=1, cols=1)

        component_values = [*args]
        part_component_vals = {}
        part_pcls = {}
        used_components = 0
        for part in part_names:
            warp_model = part_models[part]
            part_component_vals[part] = component_values[
                used_components : used_components + n_components[part]
            ]

            warp = warp_model.pca.inverse_transform(part_component_vals[part])
            part_pcls[part] = warp_model.canonical_pcl + warp.reshape(-1, 3)
            used_components += n_components[part]

            fig.add_trace(
                go.Scatter3d(
                    x=part_pcls[part][:, 0],
                    y=part_pcls[part][:, 1],
                    z=part_pcls[part][:, 2],
                    marker={
                        "size": 5,
                        "color": part_pcls[part][:, 2],
                        "colorscale": colorscale,
                    },
                    mode="markers",
                    opacity=1.0,
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    range=[-0.1, 0.1],
                ),
                yaxis=dict(
                    nticks=4,
                    range=[-0.1, 0.1],
                ),
                zaxis=dict(
                    range=[-0.1, 0.1],
                ),
            ),
        )

        fig.update_layout(scene_aspectmode="cube")

        return fig

    app.run_server(debug=True)


if __name__ == "__main__":
    warp_file_stamp = '20250217-165848_10'

    #todo: generalize for other objects
    object_warp_file = f'./whole_mug_{warp_file_stamp}'
    concat_warp_file = f'./mug_concat_{warp_file_stamp}'
    viz_whole_model(concat_warp_file)
    # viz_part_models(part_files, part_names)