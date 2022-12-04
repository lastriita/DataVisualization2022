# Importamos las librerias mínimas necesarias
import numpy as np
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.figure_factory as ff #importing a new function from plotly

from dash.dependencies import Input, Output, State
import logging

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import plotly.express as px

import pandas as pd
import seaborn as sns #data visualization


# A la hora de desarrollar una aplicación para visualizar datos tendremos que combinar 
# elementos de HTML y CSS con elementos propios de Dash. Lo primero que tendremos que 
# hacer siempre es inicializar una aplicación de Dash

app = dash.Dash()

logging.getLogger('werkzeug').setLevel(logging.INFO)

about_md = '''
### Dash and Vaex: Big data exposed

An example of an interactive dashboard created with [Vaex](https://github.com/vaexio/vaex) and
[Dash](https://plotly.com/dash/). Vaex is a high performance DataFrame library enabling efficient, out-of-core computing
for large datasets comprising millions or billions of samples. Thie example uses Vaex as an engine for computing statistics
and aggregations which are passed to Plotly to create beautiful diagrams. The dataset shown comprises nearly 120
million trips conducted by the Yellow Taxies throughout New York City in 2012, and is available via the [Taxi &
Limousine Commission](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

Read [this article](link_placeholder) to learn how to create such dashboards with Vaex and Dash.
'''

fig_layout_defaults = dict(
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
)

def create_figure_empty():
    layout = go.Layout(plot_bgcolor='white', width=10, height=10,
                       xaxis=go.layout.XAxis(visible=False),
                       yaxis=go.layout.YAxis(visible=False))
    return go.Figure(layout=layout)


# Taken from https://dash.plotly.com/datatable/conditional-formatting
def data_bars(df, column):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [
        ((df[column].max() - df[column].min()) * i) + df[column].min()
        for i in bounds
    ]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        styles.append({
            'if': {
                'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': column
            },
            'background': (
                """
                    linear-gradient(90deg,
                    #96dbfa 0%,
                    #96dbfa {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(max_bound_percentage=max_bound_percentage)
            ),
            'paddingBottom': 2,
            'paddingTop': 2
        })

    return styles
# Una vez hemos inicializado la aplicacion, modificamos el diseño de la aplicacion

# IMPORTANTE: Hay que ser extremadamente ordenado con el código para que se entienda
# correctamente que se está haciendo en cada parte. Se recomienda un vistazo a la 
# libreria Black para formateo del código.

#######################################
# Initialize
#######################################
df = pd.read_csv("https://raw.githubusercontent.com/lastriita/DataVisualization2022/main/Bank%20Customer%20Churn%20Prediction.csv")

#######################################
# Figure/plotly function
#######################################
def create_figure_pie():
    level_count = df["churn"].value_counts()
    data = [
        go.Pie(
            labels=["Never Left the Bank","Has Left the Bank at Some Point"],
            values=level_count,
            textinfo='percent',
            insidetextorientation='radial',
            marker_colors = ["lightblue", "mediumseagreen"],
            rotation = -55,
        )
    ]

    layout = go.Layout(title = "Representación del Churn del Banco",
                       **fig_layout_defaults,
                       showlegend=False)

    return go.Figure(data = data, layout = layout)


def create_figure_boxplot():
    data = [
        go.Box(

            y = df["balance"].loc[df["churn"]==0],
            marker_color = "firebrick",
            name = "Clients that Never Left the Bank",
            boxmean=True
        ),
        go.Box(

            y = df["balance"].loc[df["churn"]==1],
            marker_color = "lightblue",
            name = "Client that has Left the Bank at Some Point",
            boxmean=True
        )
    ]

    layout = go.Layout(title = "Balance en Cuenta vs Churn",
                       yaxis_title = "Balance en Cuenta",
                       **fig_layout_defaults,
                       showlegend=False)

    return go.Figure(data = data, layout = layout)

def create_figure_density():
    x1=df[df["churn"]==0]["tenure"]
    x2=df[df["churn"]==1]["tenure"]
    hist_data = [x1,x2]

    group_labels = ['Churn no','Churn yes']
    colors = ['#DC3912', '#FFA15A']

    # Create distplot with curve_type set to 'normal'
    distplot1 = ff.create_distplot(hist_data, group_labels, curve_type = 'normal', show_hist=False, colors=colors)

    # Add title
    distplot1.update_layout(title_text='Churn Distribution based on Tenure',
                            **fig_layout_defaults)
    distplot1.update_xaxes(title_text='Years')
    distplot1.update_yaxes(title_text='Density')

    return distplot1



# Primer dashboard

# The app layout
app.layout = html.Div(className='app-body', children=[
    # Stores
    dcc.Store(id='map_clicks', data=0),
    # About the app + logos
    html.Div(className="row", children=[
        html.Div(className='twelve columns', children=[
            html.Div(style={'float': 'left'}, children=[
                html.H1('Churn Dashboard'),
                html.H4('Million Taxi trips in Real Time')
            ]
                     ),
            html.Div(style={'float': 'right'}, children=[
                html.A(
                    html.Img(
                        src=app.get_asset_url("logopng.png"),
                        style={'float': 'right', 'height': '120px'}
                    ),
                    href="https://www.comillas.edu/en/")
            ]),
        ]),
    ]),
    # Control panel
    html.Div(className="row", id='control-panel', children=[
        html.Div(className="four columns pretty_container", children=[
            dcc.Loading(
                className="loader",
                id="loading",
                type="default",
                children=[
                    html.Div(id='loader-trigger-1', style={"display": "none"}),
                    html.Div(id='loader-trigger-2', style={"display": "none"}),
                    html.Div(id='loader-trigger-3', style={"display": "none"}),
                    html.Div(id='loader-trigger-4', style={"display": "none"}),
                    dcc.Markdown(id='data_summary_filtered', children='data_summary_filtered_md'),
                    html.Progress(id="selected_progress", max=f"{1000}", value=f"{1000}"),
                ]),
        ]),
        html.Div(className="four columns pretty_container", children=[
            html.Label('Select pick-up age'),
            dcc.RangeSlider(id='age',
                            value=[0, 23],
                            min=0, max=23,
                            marks={i: str(i) for i in range(0, 24, 3)}),
        ]),
        html.Div(className="four columns pretty_container", children=[
            html.Label('Select pick-up days'),
            dcc.Dropdown(id='days',
                         placeholder='Select a day of week',
                         options=[{'label': 'Monday', 'value': 0},
                                  {'label': 'Tuesday', 'value': 1},
                                  {'label': 'Wednesday', 'value': 2},
                                  {'label': 'Thursday', 'value': 3},
                                  {'label': 'Friday', 'value': 4},
                                  {'label': 'Saturday', 'value': 5},
                                  {'label': 'Sunday', 'value': 6}],
                         value=[],
                         multi=True),
        ]),
    ]),

    # The Visuals
    dcc.Tabs(id='tab', children=[
        dcc.Tab(label='EDA', children=[
            html.Div(className="row", children=[
                html.Div(className="eight columns pretty_container", children=[
                    dcc.Graph(id='boxplot',
                              figure=create_figure_boxplot()
                              )
                ]),
                html.Div(className="four columns pretty_container", children=[
                    dcc.Graph(id='pie',
                              figure=create_figure_pie()),
                ])
            ]),
            html.Div(className="row", children=[
                html.Div(className="fix columns pretty_container", children=[
                    dcc.Graph(id='density',
                              figure=create_figure_density()
                              )
                ]),
                html.Div(className="fix columns pretty_container", children=[
                    ]),
            ]),
        ]),
        dcc.Tab(label='Trip planner', children=['''
            html.Div(className="row", children=[
                html.Div(className="seven columns pretty_container", children=[
                    dcc.Markdown(children='_Click on the map to select trip start and destination._'),
                    dcc.Graph(id='heatmap_figure',
                              figure=create_figure_heatmap(heatmap_data_initial,
                                                           heatmap_limits_initial,
                                                           trip_start_initial,
                                                           trip_end_initial),
                              config={"modeBarButtonsToRemove": ['lasso2d', 'select2d', 'hoverCompareCartesian']})
                ]),
                html.Div(className="five columns pretty_container", children=[
                    dcc.Graph(id='trip_summary_amount_figure'),
                    dcc.Graph(id='trip_summary_duration_figure'),
                    dcc.Markdown(id='trip_summary_md'),
                ])
            ]),
        ''']),
    ]),
    html.Hr(),
    dcc.Markdown(children=about_md),

])

def compute_flow_data(days, age):
    return 0

# Flow section
@app.callback(
    Output('pie', 'figure'),
    Input('days', 'value'),
    Input('age', 'value')
)
def update_flow_figures(days, age):
    flow_data = compute_flow_data(days, age)

    fig_pie = create_figure_pie()

    return fig_pie

if __name__ == '__main__':
    app.run_server()

#%%
