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

from sklearn.model_selection import train_test_split, cross_val_score
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

import pandas as pd
import seaborn as sns #data visualization

from sklearn.linear_model import LogisticRegression ,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from plotly.subplots import make_subplots

# A la hora de desarrollar una aplicación para visualizar datos tendremos que combinar 
# elementos de HTML y CSS con elementos propios de Dash. Lo primero que tendremos que 
# hacer siempre es inicializar una aplicación de Dash
df = pd.read_csv("https://raw.githubusercontent.com/lastriita/DataVisualization2022/main/Bank%20Customer%20Churn%20Prediction.csv")

app = dash.Dash()

logging.getLogger('werkzeug').setLevel(logging.INFO)

about_md = '''
Developed by (AAA): Álvaro Lastra, Armando Sala, and Álvaro Diez de Rivera
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
maxValues = df.max()
minValues = df.min()

max_age = maxValues["age"]
min_age = minValues["age"]

max_balance = maxValues["balance"]
min_balance = minValues["balance"]

max_credit_score = maxValues["credit_score"]
min_credit_score = minValues["credit_score"]

max_estimated_salary = maxValues["estimated_salary"]
min_estimated_salary = minValues["estimated_salary"]

data_summary_filtered_md_template = 'Selected {:,} clients'
data_summary_filtered_md = data_summary_filtered_md_template.format(len(df))

prob_md_template = 'Probability {:,} %'
prob_md = prob_md_template.format(0)
#######################################
# Models
#######################################
df2 = df.drop(columns = ["customer_id"])

X = df2.iloc[:, 0:10].values
y = df2.iloc[:, 10].values
label_encoder_country = LabelEncoder()
label_encoder_gender = LabelEncoder()
X[:,1] = label_encoder_country.fit_transform(X[:,1])
X[:,2] = label_encoder_gender.fit_transform(X[:,2])

#Standard Scaler
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)
X_train_s, X_test_s, y_train, y_test = train_test_split(X_standard, y, test_size = 0.2, random_state = 0)

models = {'Logistic Regression': LogisticRegression(),
          'KNN': KNeighborsClassifier(),
          'Decision Tree': DecisionTreeClassifier(),
          'Random Forest': RandomForestClassifier(),
          'Gradient Boosting Classifier':GradientBoostingClassifier(),
          'Support Vector Machine': SVC(),
          'Stochastic Gradien Descent': SGDClassifier(),
          'Naive Bayes': GaussianNB(),
          'xgb Classifier': XGBClassifier()}

from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix

def my_custom_loss_func(y, y_pred, **kwargs):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    cost = fn*150000 + tp*100000 + fp*50000
    return cost

my_scorer = make_scorer(my_custom_loss_func, greater_is_better=False)


def fit_score(models, X_train, X_test, y_train, y_test):
    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = cross_val_score(model,
                                             X_test,
                                             y_test,
                                             scoring=my_scorer,
                                             cv=5
                                             ).mean()

    return model_scores

import warnings
warnings.filterwarnings("ignore")
model_scores = fit_score(models,X_train_s,X_test_s,y_train,y_test)


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train_s, y_train)
predictions_xgb = xgb.predict(X_test_s)
acc_xgb = accuracy_score(y_test, predictions_xgb)
classification_xgb = (classification_report(y_test, predictions_xgb))

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import det_curve
from numpy import argmax
from numpy import argmin
from matplotlib import pyplot
from collections import Counter

# predict probabilities
yhat = xgb.predict_proba(X_test_s)
# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
# calculate roc curves
fpr, fnr, thresholds = det_curve(y_test, yhat)

counter = Counter(y_test)
cost = fnr*counter[1]*150000 + (1-fnr)*counter[1]*100000 + fpr*counter[0]*50000


# locate the index of the largest f score
ix_cost = argmin(cost)
print('Best Threshold=%f' % (thresholds[ix_cost]))


from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)

xgb_final = XGBClassifier()
xgb_final.load_model("model_final.json")

predictions_xgb = xgb_final.predict(X_test)
acc_xgb = accuracy_score(y_test, predictions_xgb)
classification_xgb = (classification_report(y_test, predictions_xgb))

#######################################
# Figure/plotly function
#######################################
def create_figure_pie(flow_data):
    level_count = flow_data["churn"].value_counts()
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


def create_figure_boxplot(flow_data):
    data = [
        go.Box(

            y = flow_data["balance"].loc[flow_data["churn"]==0],
            marker_color = "firebrick",
            name = "Clients that Never Left the Bank",
            boxmean=True
        ),
        go.Box(

            y = flow_data["balance"].loc[flow_data["churn"]==1],
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

def create_figure_density(flow_data):
    x1=flow_data[flow_data["churn"]==0]["tenure"]
    x2=flow_data[flow_data["churn"]==1]["tenure"]
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

def create_heatmap():
    data=[[0, 50000], [150000, 100000]]
    fig = px.imshow(data,
                    labels=dict(x="Predicted", y="Actual", color="Cost (€)"),
                    x=['0', '1'],
                    y=['0', '1'],
                    text_auto=True
                    )
    fig.update_layout(title_text='Coste de cada predicción',
                            **fig_layout_defaults)
    return fig

def create_barplotmodels():
    models = pd.DataFrame(model_scores, index=["accuracy"])
    fig = px.bar(models.T)
    fig.update_layout(title_text='Coste de cada modelo',
                      **fig_layout_defaults,
                      showlegend=False)

    return fig

def create_fnr_scatter():
    fig = px.scatter(x=fnr, y=fpr)
    fig.add_scatter(x=[fnr[ix_cost]], y=[fpr[ix_cost]], name="Best",
                    marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')))

    fig.update_layout(title_text='FNR and FPR curve',
                      **fig_layout_defaults,
                      showlegend=False)

    fig.update_xaxes(title_text='fnr')
    fig.update_yaxes(title_text='fpr')
    return fig

def create_threshold_scatter():
    fig = px.scatter(x=thresholds, y=cost)
    fig.add_scatter(x=[thresholds[ix_cost]], y=[cost[ix_cost]], name="Best", marker=dict(size=12,line=dict(width=2,  color='DarkSlateGrey')))

    fig.update_layout(title_text='Threshold and Cost curve',
                      **fig_layout_defaults,
                      showlegend=False)

    fig.update_xaxes(title_text='Threshold')
    fig.update_yaxes(title_text='Cost')

    return fig

def create_bar_importancia():
    importances = pd.Series(xgb_final.feature_importances_, index=df2.columns[:10]).sort_values()
    fig = px.bar(importances)

    fig.update_layout(title_text='Feature Importances XGBoost model',
                      **fig_layout_defaults,
                      showlegend=False)

    fig.update_xaxes(title_text='Features')
    fig.update_yaxes(title_text='Importance')

    return fig

def create_conf_mat():
    from sklearn.metrics import confusion_matrix

    conf = confusion_matrix(y_test, predictions_xgb)

    fig = px.imshow(conf,
                    labels=dict(x="Predicted", y="Actual", color="Total"),
                    x=['No churn', 'Churn'],
                    y=['No churn', 'Churn'],
                    text_auto=True
                    )
    fig.update_layout(title_text='Matriz de confusión',
                      **fig_layout_defaults,
                      showlegend=False)
    return fig

def create_subplot(categorical, continuous):
    fig = make_subplots(rows = 1,
                        cols = 3,
                        specs =[[{"type": "pie"}, {"type": "bar"},{"type": "bar"}]],
                        subplot_titles=("Composición de los datos por "+categorical, "Churn % por "+categorical,
                                        "Violin Plot: "+continuous +" por " +categorical))
    if categorical=='gender':
        labels=["Mujeres","Hombres"]
    else:
        labels=["Yes","No"]

    # primer gráfico
    level_count = df[categorical].value_counts()

    fig.add_trace(
        go.Pie(
            labels = labels,
            values=level_count,
            textinfo='label + percent',
            insidetextorientation='radial',
            marker_colors = ["lightblue", "mediumseagreen"],
            showlegend = False,
            domain=dict(x=[0, 0.5])
        ),
        row = 1,
        col = 1
    )

    # Segundo gráfico
    groups = df.groupby(["churn",categorical])["churn"].count()

    fig.add_trace(
        go.Bar(
            y = [groups.values[0],groups.values[1],groups.values[2],groups.values[3]],
            x = [labels[0]+" Clients that Never Left the Bank",labels[1]+" Clients that Never Left the Bank",labels[0]+" Clients that Have Left the Bank", labels[1]+" Clients that Have Left the Bank"],
            showlegend = False,
            marker_color = ["lightblue", "mediumseagreen", "gold", "darkorange"],
        ),
        row = 1,
        col = 2
    )

    # Tercer gráfico
    fig.add_trace(
        go.Violin(
            x=df[categorical][ df['churn'] == 1 ],
            y=df[continuous][ df['churn'] == 1 ],
            legendgroup='Churn Yes', scalegroup='Churn Yes', name='Churn Yes',
            side='negative',
            line_color='blue'
        ),
        row = 1,
        col = 3
    )


    fig.add_trace(
        go.Violin(
            x=df[categorical][ df['churn'] == 0 ],
            y=df[continuous][ df['churn'] == 0],
            legendgroup='Churn No', scalegroup='Churn No', name='Churn No',
            side='positive',
            line_color='orange'
        ),
        row = 1,
        col = 3
    )
    # Modifico las dimensiones totales y el titulo global
    fig.update_layout(**fig_layout_defaults,
                      title = "Información sobre los clientes del banco", bargap = 0.1)

    return fig

def create_kde(flow_data):
    # primer gráfico
    x1=flow_data[flow_data["churn"]==0]["age"]
    x2=flow_data[flow_data["churn"]==1]["age"]
    hist_data = [x1,x2]

    group_labels = ['Churn no','Churn yes']
    fig1 = ff.create_distplot(hist_data, group_labels, curve_type = 'normal', show_hist=False,show_rug=False)

    # Segundo gráfico
    z1=flow_data[flow_data["churn"]==0]["products_number"]
    z2=flow_data[flow_data["churn"]==1]["products_number"]
    hist_data_2 = [z1,z2]

    fig2 = ff.create_distplot(hist_data_2, group_labels, curve_type = 'normal', show_hist=False,show_rug=False)

    fig = make_subplots(rows = 2,
                        cols = 1,

                        subplot_titles=("Distribución del Churn por la edad", "Distribución del Churn por el Balance en Cuenta",
                                        "Distribución del Churn para el credit score"))

    # primer gráfico


    for trace in fig1.select_traces():
        fig.add_trace(trace, row=1, col=1)

    # Segundo gráfico

    for trace in fig2.select_traces():
        fig.add_trace(trace, row=2, col=1)

    # Modifico las dimensiones totales y el titulo global
    fig['layout']['xaxis']['title']='Años en el Banco'
    fig['layout']['xaxis2']['title']='Número de Productos'
    fig['layout']['yaxis']['title']='Densidad'
    fig['layout']['yaxis2']['title']='Densidad'
    fig.update_layout(**fig_layout_defaults,
                      title = "Distribución de los clientes del banco",
                      bargap = 0.1,showlegend=False)

    return fig

def create_map(flow_data):
    df2 = flow_data.replace('Spain', 'ESP').replace('France', 'FRA').replace('Germany', 'DEU')
    count = df2['country'].value_counts()
    #count['count']
    fig = px.choropleth(count, locations=count.index,
                        color='country',
                        #hover_name="country", # column to add to hover information
                        color_continuous_scale=px.colors.sequential.Plasma)

    fig.update_geos(projection_type="orthographic")
    fig.update_geos(fitbounds="locations")
    fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)
    fig.update_layout(**fig_layout_defaults)

    return fig

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
                html.H4('Bank Analysis')
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

    # The Visuals
    dcc.Tabs(id='tab', children=[
        dcc.Tab(label='EDA', children=[
            # Control panel
            html.Div(className="row", children=[
                html.Div(className="twelve columns pretty_container", children=[
                    html.Div(className="row", children=[
                        html.Div(className="six columns pretty_container", children=[
                            html.Label('Select categorical variable'),
                            dcc.Dropdown(id='categorical',
                                         placeholder='Select categorical variable',
                                         options=[{'label': 'Gender', 'value': 'gender'},
                                                  {'label': 'Credit Card', 'value': 'credit_card'},
                                                  {'label': 'Active', 'value': 'active_member'}],
                                         value=['gender'],
                                         multi=False),
                        ]),
                        html.Div(className="six columns pretty_container", children=[
                            html.Label('Select continuous variable'),
                            dcc.Dropdown(id='continuous',
                                         placeholder='Select continuous variable',
                                         options=[{'label': 'Age', 'value': 'age'},
                                                  {'label': 'Balance', 'value': 'balance'},
                                                  {'label': 'Credit score', 'value': 'credit_score'},
                                                  {'label': 'Salary', 'value': 'estimated_salary'}],
                                         value=['Male'],
                                         multi=False),
                        ]),
                    ]),
                    dcc.Graph(id='subplot',
                              figure=create_subplot('gender', 'estimated_salary')
                              )
                ]),
            ]),
            html.Hr(),
            html.Div(className="row", id='control-panel', children=[
                html.Div(className="four columns pretty_container", children=[
                    dcc.Loading(
                        className="loader",
                        id="loading",
                        type="default",
                        children=[
                            html.Div(id='loader-trigger-2', style={"display": "none"}),
                            html.Div(id='loader-trigger-3', style={"display": "none"}),
                            html.Div(id='loader-trigger-4', style={"display": "none"}),
                            dcc.Markdown(id='data_summary_filtered', children=data_summary_filtered_md),
                            html.Progress(id="selected_progress", max=f"{len(df)}", value=f"{len(df)}"),
                        ]),
                ]),
                html.Div(className="four columns pretty_container", children=[
                    html.Label('Select pick-up age'),
                    dcc.RangeSlider(id='age',
                                    value=[min_age, max_age],
                                    min=min_age, max=max_age,
                                    marks={i: str(i) for i in range(min_age, max_age, 6)}),
                ]),
                html.Div(className="four columns pretty_container", children=[
                    html.Label('Select pick-up gender'),
                    dcc.Dropdown(id='gender',
                                 placeholder='Select gender',
                                 options=[{'label': 'Female', 'value': 'Female'},
                                          {'label': 'Male', 'value': 'Male'},
                                          {'label': 'Binary', 'value': '2'}],
                                 value=[],
                                 multi=True),
                ]),
            ]),
            html.Div(className="row", children=[
                html.Div(className="eight columns pretty_container", children=[
                    dcc.Graph(id='boxplot',
                              figure=create_figure_boxplot(df)
                              )
                ]),
                html.Div(className="four columns pretty_container", children=[
                    dcc.Graph(id='pie',
                              figure=create_figure_pie(df)),
                ])
            ]),
            html.Div(className="row", children=[
                html.Div(className="twelve columns pretty_container", children=[
                    dcc.Graph(id='kde',
                              figure=create_kde(df)
                              )
                ]),
            ]),
            html.Div(className="row", children=[
                html.Div(className="fix columns pretty_container", children=[
                    dcc.Graph(id='density',
                              figure=create_figure_density(df)
                              )
                ]),
                html.Div(className="fix columns pretty_container", children=[
                    dcc.Graph(id='map',
                              figure=create_map(df)
                              )
                ]),
            ]),
        ]),
        dcc.Tab(label='Modeling', children=[
            html.Div(className="row", children=[
                html.Div(className="seven columns pretty_container", children=[
                    dcc.Graph(id='models',
                              figure=create_barplotmodels()
                              )
                ]),
                html.Div(className="five columns pretty_container", children=[
                    dcc.Graph(id='heatmap',
                              figure=create_heatmap()
                              )
                ])
            ]),
            html.Hr(),
            html.Div(className="row", children=[
                html.Div(style={'float': 'left'}, children=[
                    html.H2('XGBoost Threshold Optimization')
                ]
                         )
            ]),
            html.Div(className="row", children=[
                html.Div(style={'float': 'left'}, children=[
                    html.H3('Loss Function:')
                ]
                         )
            ]),
            html.Div(className="row", children=[
                html.Div(style={'display': 'inline-block'}, children=[
                    html.Img(
                        src=app.get_asset_url("loss.png"),
                        style={'float': 'center', 'height': '80px'}
                    )
                ]),
            ]),
            html.Div(className="row", children=[
                html.Div(className="six columns pretty_container", children=[
                    dcc.Graph(id='fnr_curve',
                              figure=create_fnr_scatter()
                              )
                ]),
                html.Div(className="six columns pretty_container", children=[
                    dcc.Graph(id='threshold',
                              figure=create_threshold_scatter()
                              )
                ])
            ]),
            html.Hr(),
            html.Div(className="row", children=[
                html.Div(style={'float': 'left'}, children=[
                    html.H2('Best Model (Balanced Data):')
                ]
                         )
            ]),
            html.Div(className="row", children=[
                html.Div(className="five columns pretty_container", children=[
                    dcc.Graph(id='conf_mat',
                              figure=create_conf_mat()
                              )
                ]),
                html.Div(className="seven columns pretty_container", children=[
                    dcc.Graph(id='importancia',
                              figure=create_bar_importancia()
                              )
                ])
            ]),
        ]),
        dcc.Tab(label='Churn Prob Calculator', children=[
            # Control panel
            html.Div(className="row", children=[
                html.Div(className="three columns pretty_container", children=[
                    html.Label('Select pick-up age'),
                    dcc.Slider(id='age2',
                               value=[min_age],
                               min=min_age, max=max_age,
                               marks={i: str(i) for i in range(min_age, max_age, 6)}),
                ]),
                html.Div(className="three columns pretty_container", children=[
                    html.Label('Select balance'),
                    dcc.Slider(id='balance',
                               value=[min_balance],
                               min=min_balance, max=max_balance,
                               marks={i: str(i) for i in range(int(min_balance), int(max_balance), int((min_balance - max_balance)/10))}),

                ]),
                html.Div(className="three columns pretty_container", children=[
                    html.Label('Select Credit Score'),
                    dcc.Slider(id='credit_score',
                               value=[min_credit_score],
                               min=min_credit_score, max=max_credit_score,
                               marks={i: str(i) for i in range(int(min_credit_score), int(max_credit_score), int((min_credit_score - max_credit_score)/10))}),

                ]),
                html.Div(className="three columns pretty_container", children=[
                    html.Label('Select salary'),
                    dcc.Slider(id='salary',
                               value=[min_estimated_salary],
                               min=min_estimated_salary, max=max_estimated_salary,
                               marks={i: str(i) for i in range(int(min_estimated_salary), int(max_estimated_salary), int((min_estimated_salary - max_estimated_salary)/10))}),
                ]),
            ]),
            html.Div(className="row", children=[
                html.Div(className="two columns pretty_container", children=[
                    html.Label('Select pick-up gender'),
                    dcc.Dropdown(id='gender2',
                                 placeholder='Select gender',
                                 options=[{'label': 'Female', 'value': 1},
                                          {'label': 'Male', 'value': 0}],
                                 value=['Male'],
                                 multi=False),
                ]),
                html.Div(className="two columns pretty_container", children=[
                    html.Label('Select if credit card'),
                    dcc.Dropdown(id='credit_card',
                                 placeholder='Select if credit card',
                                 options=[{'label': 'Yes', 'value': 1},
                                          {'label': 'No', 'value': 0}],
                                 value=[1],
                                 multi=False),
                ]),
                html.Div(className="two columns pretty_container", children=[
                    html.Label('Select pick-up Country'),
                    dcc.Dropdown(id='country',
                                 placeholder='Select Country',
                                 options=[{'label': 'Spain', 'value': 0},
                                          {'label': 'France', 'value': 1},
                                          {'label': 'Germany', 'value': 2}],
                                 value=['Spain'],
                                 multi=False),
                ]),
                html.Div(className="two columns pretty_container", children=[
                    html.Label('Select N of products'),
                    dcc.Dropdown(id='product',
                                 placeholder='Select N Products',
                                 options=[{'label': '1', 'value': 1},
                                          {'label': '2', 'value': 2},
                                          {'label': '3', 'value': 3},
                                          {'label': '4', 'value': 4}],
                                 value=[1],
                                 multi=False),
                ]),
                html.Div(className="two columns pretty_container", children=[
                    html.Label('Select if active'),
                    dcc.Dropdown(id='active',
                                 placeholder='Select if active',
                                 options=[{'label': 'Yes', 'value': 1},
                                          {'label': 'No', 'value': 0}],
                                 value=[1],
                                 multi=False),
                ]),
                html.Div(className="two columns pretty_container", children=[
                    html.Label('Select Tenure'),
                    dcc.Dropdown(id='tenure',
                                 placeholder='Select Tenure',
                                 options=[{'label': '1', 'value': 1},
                                          {'label': '2', 'value': 2},
                                          {'label': '3', 'value': 3},
                                          {'label': '4', 'value': 4},
                                          {'label': '5', 'value': 5},
                                          {'label': '6', 'value': 6},
                                          {'label': '7', 'value': 7},
                                          {'label': '8', 'value': 8},
                                          {'label': '9', 'value': 9},
                                          {'label': '10', 'value': 10}],
                                 value=[1],
                                 multi=False),
                ]),
            ]),
            html.Div(className="twelve columns pretty_container", children=[

                dcc.Loading(
                    className="loader",
                    id="loading2",
                    type="default",
                    children=[
                        html.Div(id='loader-trigger-1', style={"display": "none"}),
                        html.H1(id='prob', children=prob_md),
                        html.Progress(id="prob2", max=f"{1}", value=f"0"),
                    ]),
            ]),
        ]),
    ]),
    html.Hr(),
    dcc.Markdown(children=about_md),


])

def compute_flow_data(gender, age):
    df_copy = df.copy()
    if age:
        age_min, age_max = age
        df_copy = df_copy[(df_copy['age'] >= age_min) & (df_copy['age'] <= age_max)]
    if gender:
        df_copy = df_copy[df_copy['gender'].isin(gender)]

    return df_copy

# Flow section
@app.callback(
    Output('pie', 'figure'),
    Output('boxplot', 'figure'),
    Output('density', 'figure'),
    Output('kde', 'figure'),
    Output('map', 'figure'),
    Output('selected_progress', 'value'),
    Output('loader-trigger-3', 'children'),
    Output('data_summary_filtered', 'children'),
    Input('gender', 'value'),
    Input('age', 'value')
)
def update_flow_figures(gender, age):
    flow_data = compute_flow_data(gender, age)

    fig_pie = create_figure_pie(flow_data)
    fig_box = create_figure_boxplot(flow_data)
    fig_density = create_figure_density(flow_data)
    fig_kde = create_kde(flow_data)
    fig_map = create_map(flow_data)

    count = len(flow_data)
    markdown_text = data_summary_filtered_md_template.format(count)

    return fig_pie, fig_box, fig_density, fig_kde, fig_map, count, "trigger loader", markdown_text

@app.callback(
    Output('subplot', 'figure'),
    Input('categorical', 'value'),
    Input('continuous', 'value')
)
def update_flow_figures(categorical, continuous):
    fig = create_subplot(categorical, continuous)

    return fig

@app.callback(
    Output('prob', 'children'),
    Output('prob2', 'value'),
    Output('loader-trigger-1', 'children'),
    Input('age2', 'value'),
    Input('balance', 'value'),
    Input('credit_score', 'value'),
    Input('salary', 'value'),
    Input('gender2', 'value'),
    Input('credit_card', 'value'),
    Input('country', 'value'),
    Input('product', 'value'),
    Input('active', 'value'),
    Input('tenure', 'value')
)
def update_prob(age2, balance, credit_score, salary, gender2, credit_card, country, product, active, tenure):
    prob = xgb_final.predict_proba([[int(credit_score), country, gender2, age2,tenure,balance,product, credit_card, active, salary]])[:,1]

    markdown_text = prob_md_template.format(float('%.3f'%(prob[0]*100)))
    return markdown_text, prob[0], "trigger loader"

if __name__ == '__main__':
    app.run_server()

#%%
