import os
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
from sklearn.metrics import accuracy_score


path_results = "results/"
files = os.listdir(path_results)
variables = {"prop1":[], "prop2":[], "win_size":[], "scores_size":[], "file_type":[]}
columns = []
for file in files:
    f = file[:-4]
    prop1, prop2, window_size, scores_size, file_type = f.split("_")
    if file_type != "drift":  
        variables["prop1"].append(float(prop1))
        variables["prop2"].append(float(prop2))
        variables["win_size"].append(int(window_size))
        variables["scores_size"].append(int(scores_size))
        variables["file_type"].append(file_type)
    
        df = pd.read_csv(f"{path_results}{file}")
    
        for c in list(df.columns):
            columns.append(c)

columns = set(columns)
columns = list(columns)
columns.remove("real")

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])


##################################           STYLE           #################################


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "2rem 1rem",
    "color":"#F5F5F5",
    "background-color": "#2e2f2e",
    "box-shadow":"rgba(0, 0, 0, 0.35) 0px 5px 15px"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "25rem",
    "margin-right": "0",
    "top":0,
    "padding": "0 0",
    "background-color": "#F5F5F5",
    "height":"300vw",
}


sidebar = html.Div(
    [
    html.H2("DataStreams-Quantification", className="display-6 text-center", style={"color":"#F5F5F5"}),
    html.Hr(),
    html.P(
        "Select the options to be shown on the graphs", className="lead text-center", style={"font-size":"11px"},
    ),
    dbc.Nav([
        dbc.Row([
            dbc.Col([
                html.Label('Algorithms', className="pt-3"),
                dcc.Dropdown(columns, placeholder="All",
                                style={"color":"#2e2f2e"}, id='algorithms-dropdown', multi=True),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Label('Window Size', className="pt-3"),
                dcc.Slider(min=min(variables["win_size"]),
                            max=max(variables["win_size"]),
                            value=1000,
                            marks={"0": "0", "1000": "1000", "2000":"2000"},
                            id='win-size-slider',
                            tooltip={"placement": "bottom", "always_visible": True})
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Label('Scores Size', className="pt-3"),
                dcc.Slider(min=min(variables["scores_size"]),
                            max=max(variables["scores_size"]),
                            value=50,
                            marks={"50": "50", "100": "100", "500":"500"},
                            id='scores-size-slider',
                            tooltip={"placement": "bottom", "always_visible": True})
            ])
        ]),

        dbc.Row([
            dbc.Col([
                html.Label('Proportion of context 1', className="pt-3"),
                dcc.Slider(min=min(variables["prop1"]),
                            max=max(variables["prop1"]),
                            marks={"0.2": "20%", "0.5": "50%", "0.8":"80%"},
                            value=0.5,
                            step=0.3,
                            id='prop1-slider',
                            tooltip={"placement": "bottom", "always_visible": True})
                ])
            ]),
        dbc.Row([
            dbc.Col([
                html.Label('Proportion of context 2', className="pt-3"),
                dcc.Slider(min=min(variables["prop2"]),
                            max=max(variables["prop2"]),
                            marks={"0.2": "20%", "0.5": "50%", "0.8":"80%"},
                            value=0.5,
                            step=0.3,
                            id='prop2-slider',
                            tooltip={"placement": "bottom", "always_visible": True})
                ])
            ]),
        ], vertical=True, pills=True),
    ], style=SIDEBAR_STYLE,
)


##################################           CONTENT           #################################


content = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="line-plot")
        ], width=10)
    ], justify="around", style={"margin-top":"40px"}),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id="box-plot")
        ], width=10)
    ], justify="around", style={"margin-top":"40px"}),
], id="page-content", style=CONTENT_STYLE)


##################################           APP          #################################



app.layout = html.Div([
    sidebar,
    content,
])





##################################           CALLBACKS           #################################

@callback(
    Output("line-plot", "figure"),
    Output("box-plot", "figure"),
    Input("algorithms-dropdown", "value"),
    Input("win-size-slider", "value"),
    Input("scores-size-slider", "value"),
    Input("prop1-slider", "value"),
    Input("prop2-slider", "value"),
)

def update_graph(algorithms, size, score, prop1, prop2):
    df = pd.read_csv(f"{path_results}{prop1}_{prop2}_{size}_{score}_pred.csv")
    real = df["real"].tolist()
    df.drop("real", inplace=True, axis=1)
    window_size = int(size)
    alg = list(df.columns)
    
    if algorithms:
        df = df[algorithms]
        alg = algorithms
    
    df_acc = {} 
    for a in df.columns:
        mean_acc = []
        for i in range(0, len(df), window_size): 
            acc = round(accuracy_score(y_true = real[i:i+int(window_size)], y_pred = df[a].tolist()[i:i+int(window_size)]), 2)      
            mean_acc.append(acc) 
        df_acc[a] = mean_acc
    df_acc = pd.DataFrame(df_acc)
    
    line = px.line(df_acc, title="Accuracy of each Algorithms by time", labels={"variable":"Algorithms", "value":"Accuracy"}, height=700)
    box = px.box(df_acc, title="Algorithms accuracy", labels={"variable":"Algorithms", "value":"Accuracy"}, height=500)

    lin = np.linspace(window_size, len(df), len(df_acc))

    line.update_layout(
        xaxis_title="Instances",
        yaxis_title="Accuracy",
        xaxis = dict(
            tickmode = 'array',
            tickvals = list(range(0, len(df_acc))),
            ticktext = list(lin)
        )
    )
    print(df_acc)

    return line, box


##################################           RUN           #################################


if __name__ == "__main__":
    app.run_server(debug=True, port=8120)