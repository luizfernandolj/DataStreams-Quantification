import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc

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
    
    style=SIDEBAR_STYLE,
)


##################################           CONTENT           #################################


content = html.Div([
    
], id="page-content", style=CONTENT_STYLE)


##################################           APP          #################################



app.layout = html.Div([
    sidebar,
    content,
])





##################################           CALLBACKS           #################################








##################################           RUN           #################################



if __name__ == "__main__":
    app.run_server(debug=True, port=8000)