import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.dependencies import Input, Output
import pandas as pd
from dash_table.Format import Format, Align
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

df_clean = pd.read_csv('ist_clean_central_data.csv')
df_merged = pd.read_csv('ist_merged.csv')
df_table = df_clean.round(2)

gr = go.Figure()
gr.add_trace(go.Scatter(x=df_merged.index, y=df_merged.Power_kW))

# #####################################################################################################
# ------------------------------------DATA--CLEANING---------------------------------------------------
# ####################################################################################################

df_merged = df_merged.set_index('Date', drop=True)
gr1 = go.Figure()
gr1.add_trace(go.Scatter(x=df_merged.index, y=df_merged.Power_kW, name='Power_kW before cleaning'))

# -----------------------------------------------Z-SCORE------------------
z = np.abs(stats.zscore(df_merged['Power_kW']))
threshold = 3
df_merged1 = df_merged[(z < 3)]

# ---------------------------------------------QUARTILE------------------------
Q1a = df_merged['Power_kW'].quantile(0.25)
Q3a = df_merged['Power_kW'].quantile(0.75)
IQRa = Q3a - Q1a
df_merged2 = df_merged[((df_merged['Power_kW'] > (Q1a - 1.5 * IQRa)) & (df_merged['Power_kW'] < (Q3a + 1.5 * IQRa)))]

# ------------------------------------------------QUARTILE+CUT LOWER-------------
df_merged3 = df_merged[df_merged['Power_kW'] > df_merged['Power_kW'].quantile(0.03)]
Q1b = df_merged3['Power_kW'].quantile(0.25)
Q3b = df_merged3['Power_kW'].quantile(0.75)
IQRb = Q3b - Q1b
df_merged3 = df_merged3[((df_merged3['Power_kW'] > (Q1b - 1.5 * IQRb)) & (df_merged3['Power_kW'] < (Q3b + 1.5 * IQRb)))]

# ################################################################################################
# -----------------------------------FEATURE---SELECTION----------------------------------
# ##############################################################################################

df_clean['Date'] = pd.to_datetime(df_clean['Date'], dayfirst=True)
df_clean = df_clean.set_index('Date', drop=True)
df_clean['solarRad'] = df_clean['solarRad_W/m2']
X = df_clean.values
Y = X[:, 8]
X = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]]

# ------------------------------Solar Radiation, Temp, Logtemp-------------------
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df_clean.index, y=df_clean.temp_C * 15, mode='lines', name='temp_C'))
fig4.add_trace(go.Scatter(x=df_clean.index, y=df_clean.solarRad * 0.5, mode='lines', name='solarRad'))
fig4.add_trace(go.Scatter(x=df_clean.index, y=df_clean.Power_kW, mode='lines', name='Power_kW'))
fig4.add_trace(go.Scatter(x=df_clean.index, y=df_clean.logtemp * 100, mode='lines', name='logtemp'))

# ------------------------------Ensemble Method--------------------------------
model = RandomForestRegressor()
model.fit(X, Y)
fig3 = px.bar(x=[i for i in range(len(model.feature_importances_))], y=model.feature_importances_)

# -----------------------------------------kBest---------------------------------------
features = SelectKBest(k=3, score_func=f_regression).fit(X, Y)
fig1 = px.bar(x=[i for i in range(len(features.scores_))], y=features.scores_)

# -------------------------------Recursive Feature Elimination (RFE)------------------------
model = LinearRegression()
rfe = RFE(model, 1)
fit = rfe.fit(X, Y)
fig2 = px.bar(x=[i for i in range(len(fit.ranking_))], y=fit.ranking_)

ft = list(df_clean.columns)
graph = px.scatter(x=[0], y=[0])

# -------------------------------------------------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.H1('Central Building Energy Prediction Model'),
    html.H2('Beatriz Carrasco 84219'),
    dcc.Tabs(id='tabs', value='tab-0', children=[
        dcc.Tab(label='Explore Raw Data', value='tab-0'),
        dcc.Tab(label='Data Cleaning', value='tab-1'),
        dcc.Tab(label='Clustering', value='tab-2'),
        dcc.Tab(label='Feature Selection', value='tab-3'),
        dcc.Tab(label='Regression', value='tab-4')
    ]),
    html.Div(id='tabs-content')
])


# ################################################################################
# -----------------------------------TABS----------------------------------
# ###################################################################################

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-0':  # Explore Raw Data
        return html.Div([
            html.H4('Power Consumption'),
            dcc.Graph(id='graph0', figure=gr),
            html.H4('Full Feature Table'),
            html.Plaintext('2 years of data: each table-page represents 1 day'),
            dash_table.DataTable(
                id='table', columns=[{"name": i, "id": i} for i in df_table.columns],
                data=df_table.to_dict('records'),
                virtualization=True,
                page_size=24,
                fixed_rows={'headers': True},
                style_table={'height': '800px'},
                style_cell={'minWidth': 50, 'maxWidth': 50, 'width': 50, 'font-family': 'sans-serif'},

            )
        ])
    if tab == 'tab-1':  # Data Cleaning
        return html.Div([
            html.H4('Choose the Data Cleaning Method you want to see:'),
            html.H6('In the final model "Quartile + Cut Low Values" was used'),
            html.Button('Z-Score', id='btn1', n_clicks=0),
            html.Button('Quartile', id='btn2', n_clicks=0),
            html.Button('Quartile + Cut Low Values', id='btn3', n_clicks=0),
            html.Button('Reset', id='btn4', n_clicks=0),
            dcc.Graph(id='graph1', figure=gr1)

        ])
    elif tab == 'tab-2':  # Clustering
        return html.Div([
            html.H4('Seeing Daily Patterns'),
            html.Br(),
            html.H6('Choosing the best number of clusters:'),
            html.Img(src=app.get_asset_url('download.png')),
            html.H6('The three Daily Patterns:'),
            html.Img(src=app.get_asset_url('download1.png'))
        ])
    elif tab == 'tab-3':  # Features
        return html.Div([
            html.H4('Features:'),
            html.H6("0-temp_C  |  1-HR   |  2-windSpeed_m/s  | 3-windGust_m/s  |  4-pres_mbar  | 5-solarRad_W/m2"),
            html.H6("6-rain_mm/h  | 7-rain_day |  8-Holiday+Weekend  |  9-Hour "
                    "|  10-DayWeek | 11-Power-1  |  12-logtemp"),
            html.H4('Choose the Feature Selection Criteria you want to see:'),
            dcc.Dropdown(
                id='FeatureSelection',
                options=[
                    {'label': 'kBest', 'value': 'drop1'},
                    {'label': 'Recursive Feature Elimination (RFE)', 'value': 'drop2'},
                    {'label': 'Ensemble Method', 'value': 'drop3'},
                    {'label': 'Solar Radiation, Temp and Logtemp Comparison', 'value': 'drop4'}],
                value='drop1'
            ),
            dcc.Graph(id='graph3', figure=fig2)
        ])

    elif tab == 'tab-4':  # Regression
        return html.Div([
            html.H3('Choose the features you want to use:'),
            html.H6('The features used in the final model were solarRad_W/m2,Holiday+Weekend,Hour and Power-1'),
            html.Plaintext('Please choose more than 1 feature'),
            dcc.Dropdown(id='Regression1',
                         multi=True,
                         options=[{'label': i, 'value': i} for i in ft]
                         ),
            html.H3('Choose the regression method:'),
            html.H6('The final and best regression method was Random Forest '),
            dcc.Dropdown(id='Regression2',
                         options=[{'label': 'Linear Regression', 'value': 'reg1'},
                                  {'label': 'Support Vector Regressor', 'value': 'reg2'},
                                  {'label': 'Random Forest', 'value': 'reg3'},
                                  {'label': 'Gradient Boosting', 'value': 'reg4'},
                                  {'label': 'Bootstrapping', 'value': 'reg5'}]
                         ),
            html.Plaintext('The plot might take a few seconds to load'),
            dcc.Graph(id='graph4', figure=graph)
        ])


# ################################################################################
# -------------------------FEATURE--SELECTION---GRAPH--------------------------------
# ###################################################################################

@app.callback(
    dash.dependencies.Output('graph3', 'figure'),
    [dash.dependencies.Input('FeatureSelection', 'value')])
def update_graph(values):
    if values == 'drop4':
        return fig4
    elif values == 'drop3':
        return fig3
    elif values == 'drop2':
        return fig2
    elif values == 'drop1':
        return fig1


# ################################################################################
# -------------------------DATA--CLEANING---GRAPH----------------------------------
# ###################################################################################

@app.callback(Output('graph1', 'figure'),
              Input('btn1', 'n_clicks'),
              Input('btn2', 'n_clicks'),
              Input('btn3', 'n_clicks'),
              Input('btn4', 'n_clicks'))
def displayclick(b1, b2, b3, b4):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn1' in changed_id:
        gr1.add_trace(go.Scatter(x=df_merged1.index, y=df_merged1.Power_kW,
                                 name='ZScore', line_color='red'))
    elif 'btn2' in changed_id:
        gr1.add_trace(go.Scatter(x=df_merged2.index, y=df_merged2.Power_kW,
                                 name='Quartile', line_color='orange'))
    elif 'btn3' in changed_id:
        gr1.add_trace(go.Scatter(x=df_merged3.index, y=df_merged3.Power_kW,
                                 name='Quartile+Cut Low Values', line_color='green'))
    elif 'btn4' in changed_id:
        gr1.data = []
        gr1.add_trace(go.Scatter(x=df_merged.index, y=df_merged.Power_kW,
                                 name='Power_kW before cleaning', line_color='blue'))
    else:
        gr1.data = []
        gr1.add_trace(go.Scatter(x=df_merged.index, y=df_merged.Power_kW,
                                 name='Power_kW before cleaning', line_color='blue'))
    return gr1


# ################################################################################
# -------------------------REGRESSION---GRAPH----------------------------------
# ###################################################################################


@app.callback(Output('graph4', 'figure'),
              Input('Regression1', 'value'),
              Input('Regression2', 'value'))
def update_graph(value1, value2):
    if value2 == 'reg1':
        g = []
        for a in value1:
            for i in range(len(ft)):
                if ft[i] == a:
                    g.append(i)
        x1 = df_clean.values
        y1 = x1[:, 8]
        x1 = x1[:, g]
        X_train, X_test, y_train, y_test = train_test_split(x1, y1)
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        y_pred_LR = regr.predict(X_test)
        MAE_LR = str(metrics.mean_absolute_error(y_test, y_pred_LR))
        MSE_LR = str(metrics.mean_squared_error(y_test, y_pred_LR))
        RMSE_LR = np.sqrt(metrics.mean_squared_error(y_test, y_pred_LR))
        cvRMSE_LR = RMSE_LR / np.mean(y_test)
        RMSE_LR = str(RMSE_LR)
        cvRMSE_LR = str(cvRMSE_LR)
        graph = px.scatter(x=y_test, y=y_pred_LR, marginal_y="violin")
        graph.update_layout(xaxis_title='Test Data', yaxis_title='Predicted Data')
        graph.add_annotation(dict(font=dict(size=15),
                                  x=0,
                                  y=-1.3,
                                  showarrow=False,
                                  text="MAE:" + MAE_LR + "   MSE: " + MSE_LR + "   RMSE: " + RMSE_LR + "   cvRMSE:" + cvRMSE_LR,
                                  xanchor='left'))
    if value2 == 'reg2':
        g = []
        for a in value1:
            for i in range(len(ft)):
                if ft[i] == a:
                    g.append(i)
        x1 = df_clean.values
        y1 = x1[:, 8]
        x1 = x1[:, g]
        X_train, X_test, y_train, y_test = train_test_split(x1, y1)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train_SVR = sc_X.fit_transform(X_train)
        y_train_SVR = sc_y.fit_transform(y_train.reshape(-1, 1))
        regr = SVR(kernel='rbf')
        regr.fit(X_train_SVR, y_train_SVR)
        y_pred_SVR = regr.predict(sc_X.fit_transform(X_test))
        MAE_SVR = str(metrics.mean_absolute_error(y_test, y_pred_SVR))
        MSE_SVR = str(metrics.mean_squared_error(y_test, y_pred_SVR))
        RMSE_SVR = np.sqrt(metrics.mean_squared_error(y_test, y_pred_SVR))
        cvRMSE_SVR = (RMSE_SVR / np.mean(y_test))
        RMSE_SVR = str(RMSE_SVR)
        cvRMSE_SVR = str(cvRMSE_SVR)
        graph = px.scatter(x=y_test, y=y_pred_SVR, marginal_y="violin")
        graph.update_layout(xaxis_title='Test Data', yaxis_title='Predicted Data')
        graph.add_annotation(dict(font=dict(size=15),
                                  x=0,
                                  y=-1.3,
                                  showarrow=False,
                                  text="MAE:" + MAE_SVR + "   MSE: " + MSE_SVR + "   RMSE: " + RMSE_SVR + "   cvRMSE:" + cvRMSE_SVR,
                                  xanchor='left'))
    if value2 == 'reg3':
        g = []
        for a in value1:
            for i in range(len(ft)):
                if ft[i] == a:
                    g.append(i)
        x1 = df_clean.values
        y1 = x1[:, 8]
        x1 = x1[:, g]
        X_train, X_test, y_train, y_test = train_test_split(x1, y1)
        parameters = {'bootstrap': True,
                      'min_samples_leaf': 3,
                      'n_estimators': 200,
                      'min_samples_split': 20,
                      'max_features': 'sqrt',
                      'max_depth': 40,
                      'max_leaf_nodes': None}
        RF_model = RandomForestRegressor(**parameters)
        RF_model.fit(X_train, y_train)
        y_pred_RF = RF_model.predict(X_test)
        MAE_RF = str(metrics.mean_absolute_error(y_test, y_pred_RF))
        MSE_RF = str(metrics.mean_squared_error(y_test, y_pred_RF))
        RMSE_RF = np.sqrt(metrics.mean_squared_error(y_test, y_pred_RF))
        cvRMSE_RF = RMSE_RF / np.mean(y_test)
        RMSE_RF = str(RMSE_RF)
        cvRMSE_RF = str(cvRMSE_RF)
        graph = px.scatter(x=y_test, y=y_pred_RF, marginal_y="violin")
        graph.update_layout(xaxis_title='Test Data', yaxis_title='Predicted Data')
        graph.add_annotation(dict(font=dict(size=15),
                                  x=0,
                                  y=-1.3,
                                  showarrow=False,
                                  text="MAE:" + MAE_RF + "   MSE: " + MSE_RF + "   RMSE: " + RMSE_RF + "   cvRMSE:" + cvRMSE_RF,
                                  xanchor='left'))
    if value2 == 'reg4':
        g = []
        for a in value1:
            for i in range(len(ft)):
                if ft[i] == a:
                    g.append(i)
        x1 = df_clean.values
        y1 = x1[:, 8]
        x1 = x1[:, g]
        X_train, X_test, y_train, y_test = train_test_split(x1, y1)
        params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        GB_model = GradientBoostingRegressor(**params)
        GB_model = GradientBoostingRegressor()
        GB_model.fit(X_train, y_train)
        y_pred_GB = GB_model.predict(X_test)
        MAE_GB = str(metrics.mean_absolute_error(y_test, y_pred_GB))
        MSE_GB = str(metrics.mean_squared_error(y_test, y_pred_GB))
        RMSE_GB = np.sqrt(metrics.mean_squared_error(y_test, y_pred_GB))
        cvRMSE_GB = RMSE_GB / np.mean(y_test)
        RMSE_GB = str(RMSE_GB)
        cvRMSE_GB = str(RMSE_GB)
        graph = px.scatter(x=y_test, y=y_pred_GB, marginal_y="violin")
        graph.update_layout(xaxis_title='Test Data', yaxis_title='Predicted Data')
        graph.add_annotation(dict(font=dict(size=15),
                                  x=0,
                                  y=-1.3,
                                  showarrow=False,
                                  text="MAE:" + MAE_GB + "   MSE: " + MSE_GB + "   RMSE: " + RMSE_GB + "   cvRMSE:" + cvRMSE_GB,
                                  xanchor='left'))
    if value2 == 'reg5':
        g = []
        for a in value1:
            for i in range(len(ft)):
                if ft[i] == a:
                    g.append(i)
        x1 = df_clean.values
        y1 = x1[:, 8]
        x1 = x1[:, g]
        X_train, X_test, y_train, y_test = train_test_split(x1, y1)
        BT_model = BaggingRegressor()
        BT_model.fit(X_train, y_train)
        y_pred_BT = BT_model.predict(X_test)
        MAE_BT = str(metrics.mean_absolute_error(y_test, y_pred_BT))
        MSE_BT = str(metrics.mean_squared_error(y_test, y_pred_BT))
        RMSE_BT = np.sqrt(metrics.mean_squared_error(y_test, y_pred_BT))
        cvRMSE_BT = RMSE_BT / np.mean(y_test)
        cvRMSE_BT = str(cvRMSE_BT)
        RMSE_BT = str(RMSE_BT)
        graph = px.scatter(x=y_test, y=y_pred_BT, marginal_y="violin")
        graph.update_layout(xaxis_title='Test Data', yaxis_title='Predicted Data')
        graph.add_annotation(dict(font=dict(size=15),
                                  x=0,
                                  y=-1.3,
                                  showarrow=False,
                                  text="MAE:" + MAE_BT + "   MSE: " + MSE_BT + "   RMSE: " + RMSE_BT + "   cvRMSE:" + cvRMSE_BT,
                                  xanchor='left'))
    return graph


if __name__ == '__main__':
    app.run_server(debug=True)
