import datetime
import os
import shutil
import multiprocessing
import time

import dash
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
# external_stylesheets = []

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('Evaluation', style={'textAlign': 'center'}),
        html.Div([
            "GYM Environment: ",
            dcc.Input(id='gym_name', value='CartPole-v1', type='text')
        ], style={'textAlign': 'center'}),
        # new line
        html.Br(),
        html.Div([
            "Library (optional): ",
            dcc.Input(id='gym_lib', value="", type='text')
        ], style={'textAlign': 'center'}),
        daq.StopButton(
            id='train-button-state',
            size=120,
            buttonText='Train',
            n_clicks=0,
            label=" "
        ),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
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
    context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    print("context", context)
    if context == "train-button-state":
        global process
        shutil.rmtree("logs")
        os.makedirs("logs", exist_ok=True)
        # start in a new thread
        process = multiprocessing.Process(target=train, args=(gym_name, gym_lib)).start()
    return ""


#
@app.callback(Output('train-button-state', 'disabled'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    global finished, process
    if finished:
        if process is not None:
            process.terminate()
            process.join()
        return False
    print(not finished or process is not None)
    return not finished or process is not None


def get_logs():
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


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    logs = get_logs()
    fig = plotly.tools.make_subplots(rows=1, cols=2, vertical_spacing=0.2)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    # fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    global last_logs
    current_logs = tuple([len(l[1]) for l in logs])
    print("current logs", current_logs)
    print("last logs", last_logs)
    if current_logs == last_logs and dash.callback_context.triggered[0]['prop_id'].split('.')[0] == "interval-component":
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

    # get available colors
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    for i, log in enumerate(logs):
        name = log[0]
        df = log[1]
        # set x name

        fig.append_trace({
            'x': df["t"],
            'y': df["r"].cummax(),
            'name': name,
            'mode': 'lines',
            'type': 'scatter',
            'line': {'color': colors[i % len(colors)]},
            # 'showlegend': True,
        }, 1, 1)
        # get color from the last trace

        fig.append_trace({
            'x': list(range(len(df))),
            'y': df["r"].cummax(),
            'name': name,
            'mode': 'lines',
            'type': 'scatter',
            'line': {'color': colors[i % len(colors)]},
            # 'showlegend': False,
        }, 1, 2)

    names = set()
    fig.for_each_trace(
        lambda trace:
        trace.update(showlegend=False)
        if (trace.name in names) else names.add(trace.name))

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
