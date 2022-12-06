import datetime
import os
import shutil
import multiprocessing
import time
import dash
import dill
import pandas as pd
from dash import dcc, html
import plotly
from dash.dependencies import Input, Output, State
from os import listdir
from os.path import isfile, join
import dash_daq as daq
from dash.exceptions import PreventUpdate
from yaes.utils import train_dash

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
last_run_file = ".last_run"


def get_last_run():
    """
    Returns the last run

    """
    if os.path.isfile(last_run_file):
        with open(last_run_file, "r") as f:
            return f.read().split("\n")[:2]
    return 'CartPole-v1', ''


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, assets_folder="logs")
app.layout = html.Div(
    html.Div([
        html.H4('Evaluation', style={'textAlign': 'center'}),
        html.Div([
            "GYM Environment: ",
            dcc.Input(id='gym_name', value="CartPole-v1", type='text')
        ], style={'textAlign': 'center'}),
        html.Br(),
        html.Div([
            "⠀Library (optional): ",
            dcc.Input(id='gym_lib', value="", type='text')
        ], style={'textAlign': 'center'}),
        daq.StopButton(
            id='train-button-state',
            size=120,
            buttonText='Train',
            n_clicks=0,
            label=" "
        ),
        html.H5(id='live-update-text', style={'textAlign': 'center'}),
        dcc.Graph(id='live-update-graph'),
        html.Div([
        ], style={'textAlign': 'center'}, id="videos"),
        dcc.Interval(
            id='interval-component',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0
        ),
        html.Div(id='placeholder', children="placeholder", style={'display': 'none'})
    ])
)

finished = True
process = None


def train(gym_name, gym_lib):
    """
    Trains the agents
    """
    global finished
    finished = False
    gym_lib = gym_lib if gym_lib != "" else None
    train_dash(gym_name, gym_lib)
    finished = True


@app.callback(Output('placeholder', 'children'),
              Input('train-button-state', 'n_clicks'),
              State('gym_name', 'value'),
              State('gym_lib', 'value'), prevent_initial_callbacks=True)
def train_callback(train_button, gym_name, gym_lib):
    """
    Callback for training
    """
    context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if context == "train-button-state":
        global process
        shutil.rmtree("logs", ignore_errors=True)
        os.makedirs("logs", exist_ok=True)
        with open(last_run_file, "w") as f:
            f.write(gym_name + "\n" + gym_lib)
        process = multiprocessing.Process(target=train, args=(gym_name, gym_lib)).start()
    return ""


@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_title(n):
    """
    Updates the title
    """
    gym_env, gym_lib = get_last_run()
    return gym_env + (f" - {gym_lib}" if gym_lib != "" else "")


@app.callback(Output('train-button-state', 'disabled'),
              Input('interval-component', 'n_intervals'))
def update_button(n):
    """
    Updates the button
    """
    global finished, process
    if finished:
        if process is not None:
            process.terminate()
            process.join()
        return False
    return not finished or process is not None


last_children = None


@app.callback(Output('videos', 'children'),
              Input('interval-component', 'n_intervals'))
def update_videos(n):
    """
    Updates the videos
    """
    if not os.path.exists("logs"):
        raise PreventUpdate
    logs_path = os.path.abspath("logs")
    dirs = [f for f in listdir(logs_path) if not isfile(join(logs_path, f))]
    children = []
    for d in dirs:
        if d.startswith("monitor_stats_"):
            agent_name = d.split("_")[-1]
            child = [html.H5(agent_name)]
            if os.path.exists(join(logs_path, d, "model.pkl")):
                with open(join(logs_path, d, "model.pkl"), 'rb') as f:
                    best_agent = dill.load(f, fix_imports=False, encoding="ASCII", errors="")
                formula = best_agent.formula
                if type(formula) == str:
                    formula = [formula]
                for f in formula:
                    child.append(html.Div(f))
                    child.append(html.Br())
            if os.path.exists(join(logs_path, d, "video/rl-video-episode-0.mp4")):
                video_path = join(d, "video/rl-video-episode-0.mp4")
                child.append(
                    html.Video(src=app.get_asset_url(video_path), controls=True, autoPlay=True, loop=True)
                )
            children.append(html.Div(child))
    global last_children
    if last_children == children:
        raise PreventUpdate
    last_children = children
    return children


def get_logs():
    """
    Returns the logs
    """
    logs_path = "logs"
    dirs = [f for f in listdir(logs_path) if not isfile(join(logs_path, f))]
    logs = []
    for d in dirs:
        if d.startswith("monitor_stats_"):
            df = pd.read_csv(join(logs_path, d, "monitor.csv"), header=1)
            pair = (d.split("_")[-1], df)
            logs.append(pair)
    return logs


last_logs = tuple()


@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    """
    Updates the graph
    """
    fig = plotly.tools.make_subplots(rows=1, cols=2, vertical_spacing=0.2)
    # increase font size
    fig['layout']['font']['size'] = 20

    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    if not os.path.exists("logs"):
        raise PreventUpdate
    logs = get_logs()
    # fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    global last_logs
    current_logs = tuple([len(l[1]) for l in logs])

    if current_logs == last_logs and dash.callback_context.triggered[0]['prop_id'].split('.')[
        0] == "interval-component":
        raise PreventUpdate
    else:
        last_logs = current_logs

    if len(logs) == 0:
        return fig
    # extend the x-axis range to accommodate the new data
    max_t = max([df["t"].max() for _, df in logs])
    # fig['layout']['xaxis1']['range'] = [-1, max_t + 1]
    fig['layout']['xaxis1']['range'] = [0, max_t + 1]
    max_length = max([len(df) for _, df in logs])
    # fig['layout']['xaxis2']['range'] = [-int(max_length / max_t), int(max_length * (max_t + 1) / max_t)]
    fig['layout']['xaxis2']['range'] = [0, int(max_length * (max_t + 1) / max_t)]

    fig['layout']['xaxis1']['title'] = "Training duration (s)"
    fig['layout']['xaxis2']['title'] = "Environment steps"
    fig['layout']['yaxis1']['title'] = "Reward"
    fig['layout']['yaxis2']['title'] = "Reward"

    dashes = ["solid", "dashdot", "dot", "dash", "longdash", "longdashdot"]

    # get available colors
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    for i, log in enumerate(logs):
        name = log[0]
        if name == "RLAgent":
            name = "PPOAgent"
        df = log[1]
        fig.append_trace({
            'x': df["t"],
            'y': df["r"].cummax(),
            'name': name,
            'mode': 'lines',
            'type': 'scatter',
            'line': {'color': colors[i % len(colors)], 'width': 4, 'dash': dashes[i % len(dashes)]},
            'legendgroup': f'group{i}',
        }, 1, 1)
        # get color from the last trace

        fig.append_trace({
            'x': list(range(len(df))),
            'y': df["r"].cummax(),
            'name': name,
            'mode': 'lines',
            'type': 'scatter',
            'line': {'color': colors[i % len(colors)], 'width': 4, 'dash': dashes[i % len(dashes)]},
            'showlegend': False,
            'legendgroup': f'group{i}',
        }, 1, 2)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            traceorder="normal",
        )
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
