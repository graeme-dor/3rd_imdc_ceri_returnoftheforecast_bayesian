import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import scipy.stats as st
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from bayesian.bayesian_nb_glmm import PyTorchNBGLMMThermal

BayesianThermalModel = PyTorchNBGLMMThermal

# Adjacency list of Brazilian states (excluding Espírito Santo ES)
NEIGHBORS = {
    'AC': ['RO', 'AM'],
    'AL': ['PE', 'SE', 'BA'],
    'AM': ['AC', 'RO', 'MT', 'PA', 'RR'],
    'AP': ['PA'],
    'BA': ['AL', 'SE', 'PE', 'PI', 'TO', 'GO', 'MG'],
    'CE': ['PI', 'RN', 'PB', 'PE'],
    'DF': ['GO', 'MG'],
    'GO': ['DF', 'TO', 'BA', 'MG', 'MS', 'MT'],
    'MA': ['PA', 'TO', 'PI'],
    'MG': ['BA', 'GO', 'DF', 'MS', 'SP', 'RJ'],
    'MS': ['MT', 'GO', 'MG', 'SP', 'PR'],
    'MT': ['RO', 'AM', 'PA', 'TO', 'GO', 'MS'],
    'PA': ['AM', 'RR', 'AP', 'MA', 'TO', 'MT'],
    'PB': ['RN', 'CE', 'PE'],
    'PE': ['PB', 'CE', 'PI', 'BA', 'AL'],
    'PI': ['MA', 'TO', 'BA', 'PE', 'CE'],
    'PR': ['SP', 'MS', 'SC'],
    'RJ': ['MG', 'SP'],
    'RN': ['CE', 'PB'],
    'RO': ['AC', 'AM', 'MT'],
    'RR': ['AM', 'PA'],
    'RS': ['SC'],
    'SC': ['PR', 'RS'],
    'SE': ['AL', 'BA'],
    'SP': ['MS', 'MG', 'RJ', 'PR'],
    'TO': ['PA', 'MA', 'PI', 'BA', 'GO', 'MT']
}

class HistoricalMedianModel:
    """
    Baseline model that predicts the median and quantiles of cases
    for each epidemiological week of the year across historical training years.
    """
    def __init__(self):
        self.quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        self.stats = {}  # Store quantiles for each (uf, week_of_year)
        
    def _get_week_of_year(self, dates):
        return pd.to_datetime(dates).dt.isocalendar().week.astype(int)

    def fit(self, df_train):
        df = df_train.copy()
        df['week_of_year'] = self._get_week_of_year(df['date'])
        
        self.stats = {}
        for (uf, week), group in df.groupby(['uf', 'week_of_year']):
            q_vals = np.percentile(group['casos'], [q * 100 for q in self.quantiles])
            self.stats[(uf, week)] = dict(zip(self.quantiles, q_vals))
            
        self.national_fallback = {}
        for week, group in df.groupby('week_of_year'):
            q_vals = np.percentile(group['casos'], [q * 100 for q in self.quantiles])
            self.national_fallback[week] = dict(zip(self.quantiles, q_vals))
            
    def predict(self, df_target):
        df = df_target.copy()
        df['week_of_year'] = self._get_week_of_year(df['date'])
        
        pred_dict = {f'q_{q}': [] for q in self.quantiles}
        
        for idx, row in df.iterrows():
            uf = row['uf']
            week = row['week_of_year']
            
            stats_dict = self.stats.get((uf, week))
            if stats_dict is None:
                stats_dict = self.national_fallback.get(week, {q: 0.0 for q in self.quantiles})
                
            for q in self.quantiles:
                pred_dict[f'q_{q}'].append(stats_dict[q])
                
        df_out = df_target[['uf', 'date', 'casos']].copy()
        for q in self.quantiles:
            df_out[f'q_{q}'] = pred_dict[f'q_{q}']
            
        return df_out


class SARIMABaselineModel:
    """
    Autoregressive statistical baseline model using SARIMA(1, 1, 0) x (1, 0, 0)_52.
    Underlying data is log-transformed to stabilize variance and prevent negative forecasts.
    Prediction intervals are computed analytically from the forecast standard error.
    """
    def __init__(self):
        self.quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        self.state_list = sorted(list(NEIGHBORS.keys()))
        self.history = None

    def fit(self, df_train):
        # Store train pivot to reconstruct history for prediction
        df_train_sorted = df_train.sort_values('date')
        self.history = df_train_sorted.pivot(index='date', columns='uf', values='casos').sort_index()

    def predict(self, df_target):
        # We need to forecast from the end of history up to the target dates
        target_dates = sorted(df_target['date'].unique())
        
        # Convert history index and target dates to datetime to build full timeline
        history_end = pd.to_datetime(self.history.index[-1])
        target_end = pd.to_datetime(target_dates[-1])
        
        # Generate complete weekly timeline to include gap weeks (EW26 to EW40)
        full_range = pd.date_range(start=history_end, end=target_end, freq='W-SUN')
        full_range_str = full_range.strftime('%Y-%m-%d').tolist()
        
        # Total forecasting steps from the end of history
        steps = len(full_range_str) - 1
        
        # Target step indices in the forecast array (0-indexed)
        target_indices = [full_range_str.index(d) - 1 for d in target_dates]
        
        predictions = []
        
        # Suppress statsmodels warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for uf in self.state_list:
                y = np.log1p(self.history[uf])
                
                # Fit SARIMA(1, 1, 0) x (1, 0, 0)_52
                model = SARIMAX(y, order=(1, 1, 0), seasonal_order=(1, 0, 0, 52), enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                
                # Forecast
                forecast_res = res.get_forecast(steps=steps)
                mean = forecast_res.predicted_mean.values
                se = (forecast_res.var_pred_mean ** 0.5).values
                
                # Map target dates to predictions
                for idx, target_date in enumerate(target_dates):
                    step_idx = target_indices[idx]
                    
                    mean_val = mean[step_idx]
                    se_val = se[step_idx]
                    
                    pred_quantiles = {}
                    for q in self.quantiles:
                        z_q = st.norm.ppf(q)
                        q_log = mean_val + z_q * se_val
                        q_cases = max(0.0, np.expm1(q_log))
                        pred_quantiles[q] = q_cases
                        
                    actual_row = df_target[(df_target['uf'] == uf) & (df_target['date'] == target_date)]
                    actual_cases = actual_row['casos'].values[0] if len(actual_row) > 0 else 0.0
                    
                    row_dict = {
                        'uf': uf,
                        'date': target_date,
                        'casos': actual_cases
                    }
                    for q in self.quantiles:
                        row_dict[f'q_{q}'] = pred_quantiles[q]
                        
                    predictions.append(row_dict)
                    
        return pd.DataFrame(predictions)
