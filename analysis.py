import os
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score


path_results = "results/"
path_tests = "datasets/test/"
files = os.listdir(path_results)
variables = {}
detectors = ["baseline"]
columns = []
for file in files:
    f = file[:-4]
    dataset, prop1, prop2, window_size, scores_size, file_type = f.split("_")
    if not dataset in variables.keys():
        variables[dataset] =  {"prop1":[], "prop2":[], "win_size":[], "scores_size":[], "file_type":[]}
    if file_type != "drift":  
        variables[dataset]["prop1"].append(float(prop1))
        variables[dataset]["prop2"].append(float(prop2))
        variables[dataset]["win_size"].append(int(window_size))
        variables[dataset]["scores_size"].append(int(scores_size))
        variables[dataset]["file_type"].append(file_type)
    
        df = pd.read_csv(f"{path_results}{file}")
        
        
        if file_type == "pred":
            for c in list(df.columns):
                columns.append(c)

datasets = list(variables.keys())

columns = list(set(columns))
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
    "padding": "60px 20px",
    "background-color": "#3e3f3e",
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
                html.Label('Datasets', className="pt-3"),
                dcc.Dropdown(datasets, placeholder= "None",
                                style={"color":"#2e2f2e"}, id='datasets-dropdown'),
            ])
        ]),
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
                dcc.Slider( marks={"0": "0", "1000": "1000", "2000":"2000"},
                           step=None,
                            id='win-size-slider',
                            tooltip={"placement": "bottom", "always_visible": True})
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Label('Scores Size', className="pt-3"),
                dcc.Slider(marks={"50":"50", "100": "100", "500":"500"},
                           step=None,
                            id='scores-size-slider',
                            tooltip={"placement": "bottom", "always_visible": True})
            ])
        ]),

        dbc.Row([
            dbc.Col([
                html.Label('Proportion of context 1', className="pt-3"),
                dcc.Slider(marks={"0.2": "20%", "0.5": "50%", "0.8":"80%"},
                            step=0.3,
                            id='prop1-slider',
                            tooltip={"placement": "bottom", "always_visible": True})
                ])
            ]),
        dbc.Row([
            dbc.Col([
                html.Label('Proportion of context 2', className="pt-3"),
                dcc.Slider(marks={"0.2": "20%", "0.5": "50%", "0.8":"80%"},
                            step=0.3,
                            id='prop2-slider',
                            tooltip={"placement": "bottom", "always_visible": True})
                ])
            ]),
        dbc.Row([
            dbc.Col([
                html.Label('Detector Real Proportions', className="pt-3"),
                dcc.Dropdown(detectors, placeholder="All",
                                style={"color":"#2e2f2e"}, id='real-proportions-dropdown', multi=True),
            ])
        ]),
        ], vertical=True, pills=True),
    ], style=SIDEBAR_STYLE,
)


##################################           CONTENT           #################################


content = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="concept-drift-plot")
        ], width=10)
    ], justify="around", style={"margin-top":"0px"}),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="line-plot")
        ], width=10)
    ], justify="around", style={"margin-top":"40px"}),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="proportions-plot")
        ], width=10)
    ], justify="around", style={"margin-top":"40px"}),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="box-plot")
        ], width=10)
    ], justify="around", style={"margin-top":"40px"}),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="line-discrepances")
        ], width=10)
    ], justify="around", style={"margin-top":"40px"}),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="line-dys_distances")
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
    Output("win-size-slider", "min"),
    Output("scores-size-slider", "min"),
    Output("prop1-slider", "min"),
    Output("prop2-slider", "min"),
    Input('datasets-dropdown', "value"),
)

def update_sliders_min(dataset):
    if not dataset:
        return 0, 0, 0, 0
    win_size = min(variables[dataset]["win_size"])
    scores_size = min(variables[dataset]["scores_size"])
    prop1 = min(variables[dataset]["prop1"])
    prop2 = min(variables[dataset]["prop2"])
    return win_size, scores_size, prop1, prop2




@callback(
    Output("win-size-slider", "max"),
    Output("scores-size-slider", "max"),
    Output("prop1-slider", "max"),
    Output("prop2-slider", "max"),
    Input('datasets-dropdown', "value"),
)

def update_sliders_max(dataset):
    if not dataset:
        return 0, 0, 0, 0
    win_size = max(variables[dataset]["win_size"])
    scores_size = max(variables[dataset]["scores_size"])
    prop1 = max(variables[dataset]["prop1"])
    prop2 = max(variables[dataset]["prop2"])
    return win_size, scores_size, prop1, prop2



@callback(
    Output("win-size-slider", "value"),
    Output("scores-size-slider", "value"),
    Output("prop1-slider", "value"),
    Output("prop2-slider", "value"),
    Input('datasets-dropdown', "value"),
)

def update_sliders_value(dataset):
    if not dataset:
        return 0, 0, 0, 0
    win_size = variables[dataset]["win_size"][0]
    scores_size = variables[dataset]["scores_size"][0]
    prop1 = variables[dataset]["prop1"][0]
    prop2 = variables[dataset]["prop2"][0]
    return win_size, scores_size, prop1, prop2




@callback(
    Output("concept-drift-plot", "figure"),
    Output("line-plot", "figure"),
    Output("box-plot", "figure"),
    Output("proportions-plot", "figure"),
    Output("line-discrepances", "figure"),
    Output("line-dys_distances", "figure"),
    Input("algorithms-dropdown", "value"),
    Input("win-size-slider", "value"),
    Input("scores-size-slider", "value"),
    Input("prop1-slider", "value"),
    Input("prop2-slider", "value"),
    Input("real-proportions-dropdown", "value"),
    Input('datasets-dropdown', "value")
)

def update_graph(algorithms, size, score, prop1, prop2, real_prop, dataset):
    
    if not dataset:
        concept_line = px.line()
        line = px.line()
        box = px.box()
        propline = px.line()
        discrepances = px.line()
        dys_distances = px.line()
        return concept_line, line, box, propline, discrepances, dys_distances
    
    
    concept_df = pd.read_csv(f"{path_tests}{dataset}/{prop1}_{prop2}.csv")
    concept_df = concept_df["context"].iloc[size:]

    concept_line = px.line(concept_df, line_shape='hv', height=250, labels={"value":"concepts", "index":"instances"}, title="Concepts along stream")
    
    
    df_prop = pd.read_csv(f"{path_results}{dataset}_{prop1}_{prop2}_{size}_{score}_prop.csv")
    
    if not real_prop: 
        df_prop_reals = df_prop[[f"real_{detec}" for detec in detectors]]
    else:
        df_prop_reals = df_prop[[f"real_{real_prop}"]]
    
    
    df = pd.read_csv(f"{path_results}{dataset}_{prop1}_{prop2}_{size}_{score}_pred.csv")
    real = df["real"].tolist()
    df.drop("real", inplace=True, axis=1)
    window_size = int(size)
    
    df_prop_noreal = df_prop.drop([f"real_{detec}" for detec in detectors], axis=1)
    if algorithms:
        df = df[algorithms]
        df_prop_noreal = df_prop[[p for p in algorithms if p not in detectors]]
    
    
    df_all_prop = pd.concat([df_prop_noreal, df_prop_reals], axis=1)
    
    df_acc = {} 
    for a in df.columns:
        mean_acc = []
        for i in range(0, len(df_all_prop), int(size/2)): 
            acc = round(accuracy_score(y_true = real[i:i+int(size)], y_pred = df[a].tolist()[i:i+int(size)]), 2)      
            mean_acc.append(acc) 
        df_acc[a] = mean_acc
    df_acc = pd.DataFrame(df_acc)
    
    
    prop_final = {} 
    for a in df_all_prop.columns:
        median_prop = []
        for i in range(0, len(df), int(size/2)): 
            median =   df_all_prop[a].iloc[i:i+int(size)].median()    
            median_prop.append(median) 
        prop_final[a] = median_prop
    prop_final = pd.DataFrame(prop_final)
    
    
    c = ['hsl('+str(h)+',65%'+',65%)' for h in np.linspace(0, 360, len(df_acc.columns))]
    
    
    
    line = px.line(df_acc, title="Accuracy of each Algorithms by time", labels={"variable":"Algorithms", "value":"Accuracy"}, height=700)
    box = go.Figure(data=[go.Box(name=list(df_acc.columns)[i],
                                 y=df_acc.iloc[:, i], 
                                 marker_color=c[i]) for i in range(int(len(df_acc.columns)))],)
    box.update_layout(
        xaxis_title="Quantifiers",
        yaxis_title="Accuracy",
        title="Algorithms accuracy",
        height=500
    )

    lin = np.linspace(window_size, len(df), len(df_acc), dtype=int)

    line.update_layout(
        xaxis_title="Instances",
        yaxis_title="Accuracy",
        xaxis = dict(
            tickmode = 'array',
            tickvals = list(range(0, len(df_acc))),
            ticktext = list(lin),
            tickangle=-45,
        )
    )
    
    
    
    propline = px.line( prop_final, title="Proportions of the windows", labels={"variable":"Algorithms", "value":"Proportions"}, height=700)
    propline.update_layout(
        xaxis_title="Instances",
        yaxis_title="Window Proportions",
        xaxis = dict(
            tickmode = 'array',
            tickvals = list(range(0, len(df_acc))),
            ticktext = list(lin),
            tickangle=-45,
        )
    )
    
    
    dp = pd.read_csv(f"{path_results}{dataset}_{prop1}_{prop2}_{size}_{score}_discrepances.csv")
    
    utils_final = {} 
    for a in dp.columns:
        median_d = []
        for i in range(0, len(dp), int(size/2)): 
            median =   dp[a].iloc[i:i+int(size)].median()    
            median_d.append(median) 
        utils_final[a] = median_d
    utils_final = pd.DataFrame(utils_final)
    
    discrepances_line = px.line(utils_final["baseline_discrepances"], height=500, title="Discrepances of the scores window")
    discrepances_line.update_layout(
        xaxis_title="Instances",
        yaxis_title="Discrepances",
        xaxis = dict(
            tickmode = 'array',
            tickvals = list(range(0, len(df_acc))),
            ticktext = list(lin),
            tickangle=-45,
        )
    )
    
    
    
    
    dys_distance_line = px.line(utils_final["dys_distances"], height=500, title="DyS Distance of instances")
    dys_distance_line.update_layout(
        xaxis_title="Instances",
        yaxis_title="DyS Distances",
        xaxis = dict(
            tickmode = 'array',
            tickvals = list(range(0, len(df_acc))),
            ticktext = list(lin),
            tickangle=-45,
        )
    )
    
    
    
    #print(utils_final['baseline_discrepances'].corr(utils_final['dys_distances']))
    
    

    return concept_line, line, box, propline, discrepances_line, dys_distance_line


##################################           RUN           #################################


if __name__ == "__main__":
    app.run_server(debug=True, port=8120)