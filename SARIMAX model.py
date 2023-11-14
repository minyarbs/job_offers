# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plt.style.use('seaborn-v0_8-darkgrid')

# Modelling and Forecasting
# ==============================================================================
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

#Import Data
df = pd.read_csv('jobs.csv')
df['date'] = pd.to_datetime(df['date'],format='%m/%y')
categories=['Administratif','Aéronautique','Agriculture','Agro-alimentaire','Animation','Architecture','Artisanat','Assistante de direction','Assurances','Audiovisuel','Audit','Automobile','Aviation','Banque','Biologie','Call Center','Chimie','Commerce','Communication','Comptabilité','Confection','Consulting','Cosmétique','Cuisine','Décor','Documentation','Droit','Economie','Energie','Enseignement','Finance','Fiscalité','Génie civil','Génie électrique','Gestion','Hydraulique','Immobilier','Import / Export','Industrie électro-mécaniques','Industrie Métalliques','Informatique','Journalisme','Langues','Logistique','Maintenance industrielle','Management','Marketing','Plasturgie','Production','Psychologie','Rédaction','Ressources Humaines','Santé','Sécurité','Sport','Statistiques','Telecommunication','Textile','Tourisme','Traduction','Transport']


# Indexing the Month column and dropping missing values
df.set_index('date', inplace = True)
df.dropna(inplace=True)
df = df.asfreq('MS')
df = df.sort_index()
steps = 5

#Training Data
steps=30
regressor = KNeighborsRegressor(
    n_neighbors=4,
    weights='distance',
    )
forecaster = ForecasterAutoreg(
    regressor=regressor,
    lags=10,
    transformer_y=StandardScaler()
    )
forecaster.fit(y=df['Secteur privé'])
private_sec_predictions=forecaster.predict(steps=steps)
regressor = KNeighborsRegressor(
    n_neighbors=3,
    weights='distance',
    )
forecaster = ForecasterAutoreg(
    regressor=regressor,
    lags=3,
    transformer_y=StandardScaler()
    )
forecaster.fit(y=df['Secteur public'])
public_sec_predictions=forecaster.predict(steps=steps)
regressor = KNeighborsRegressor(
    n_neighbors=6,
    weights='distance',
    )
forecaster = ForecasterAutoreg(
    regressor=regressor,
    lags=11,
    transformer_y=StandardScaler()
    )
forecaster.fit(y=df['pib'])
pib_predictions=forecaster.predict(steps=steps)
regressor = KNeighborsRegressor(
    n_neighbors=2,
    weights='distance',
    )
forecaster = ForecasterAutoreg(
    regressor=regressor,
    lags=3,
    transformer_y=StandardScaler()
    )
forecaster.fit(y=df['Freelance'])
Freelance_predictions=forecaster.predict(steps=steps)
predictions_df = pd.DataFrame({
    'Secteur privé': private_sec_predictions,
    'pib': pib_predictions,
    'Freelance': Freelance_predictions,
    'Secteur public': public_sec_predictions
})
regressor = KNeighborsRegressor(
    n_neighbors=3,
    weights='distance',
    )
forecaster = ForecasterAutoreg(
    regressor=regressor,
    lags=9,
    transformer_y=StandardScaler()
    )
predictions_list = []
for x in categories:
    model = auto_arima(df[x], seasonal=True, stepwise=True, trace=False)
    # Fit SARIMAX model
    sarimax_model = SARIMAX(df[x],
                            order=model.order,
                            seasonal_order=model.seasonal_order,
                            exog= df[['Secteur privé','pib','Freelance','Secteur public']])
    sarimax_fit = sarimax_model.fit()

    # Make predictions using the SARIMAX model
    forecast_results = sarimax_fit.get_forecast(steps=steps, exog=predictions_df)
    predictions_backtest = forecast_results.predicted_mean
    predictions_list.append(int(predictions_backtest[29]))
indices_of_largest = np.argsort(predictions_list)[-3:]
print('the most common jobs in 2026 are:' )
for i in range(3):
 print(categories[indices_of_largest[i]])

data_train = df[:-steps]
data_test  = df[-steps:]

#OUTPUT:
#The most common jobs in 2026 are:
#Administratif
#Commerce
#Informatique
