import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import scipy.stats as st
import warnings
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from bayesian.bayesian_nb_glmm import PyTorchNBGLMM, PyTorchNBGLMMNoCovariates, PyTorchNBGLMMDataDriven, PyTorchNBGLMMRegionalLags, PyTorchNBGLMMInteractions, PyTorchNBGLMMThermal, PyTorchNBGLMMSpatialThermal, PyTorchNBGLMMGravityThermal, PyTorchNBGLMMMobilityThermal

BayesianThermalModel = PyTorchNBGLMMThermal
BayesianSpatialThermalModel = PyTorchNBGLMMSpatialThermal
BayesianGravityThermalModel = PyTorchNBGLMMGravityThermal
BayesianMobilityThermalModel = PyTorchNBGLMMMobilityThermal

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


class GraphSpatioTemporalModel:
    """
    Spatio-temporal model using Ridge regression.
    Features: local lagged cases, spatial lagged cases (average of neighbors), and target week seasonality.
    Method: Direct multi-step forecasting for horizons 16 to 67.
    Quantiles: Estimated via empirical quantiles of training residuals for each horizon.
    """
    def __init__(self):
        self.quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        self.neighbors = NEIGHBORS
        self.models = {}  # Store model for each horizon h (16 to 67)
        self.residuals = {}  # Store training residuals for each horizon h
        self.state_list = sorted(list(NEIGHBORS.keys()))
        self.lag_weeks = 5  # Number of lag weeks to use as features

    def _get_week_of_year(self, dates):
        return pd.to_datetime(dates).dt.isocalendar().week.astype(int)

    def _compute_features_and_targets(self, df, is_training=True):
        df_feats = df.copy()
        df_feats['inc'] = (df_feats['casos'] / df_feats['population']) * 100000.0
        
        # Pivot incidence and cases
        df_inc_pivot = df_feats.pivot(index='date', columns='uf', values='inc').sort_index()
        df_pop_pivot = df_feats.pivot(index='date', columns='uf', values='population').sort_index()
        df_mob_pivot = df_feats.pivot(index='date', columns='uf', values='mobility_import_risk').sort_index()
        
        dates = df_inc_pivot.index.tolist()
        num_weeks = len(dates)
        
        # Precompute neighbor averages
        df_neigh_pivot = pd.DataFrame(index=df_inc_pivot.index, columns=df_inc_pivot.columns)
        for uf in self.state_list:
            neighs = [n for n in self.neighbors[uf] if n in df_inc_pivot.columns]
            if neighs:
                df_neigh_pivot[uf] = df_inc_pivot[neighs].mean(axis=1)
            else:
                df_neigh_pivot[uf] = df_inc_pivot[uf]
                
        if not is_training:
            return df_inc_pivot, df_neigh_pivot, df_mob_pivot, df_pop_pivot, dates
            
        # Convert to numpy arrays for speed
        inc_array = df_inc_pivot.values
        neigh_array = df_neigh_pivot.values
        mob_array = df_mob_pivot.values
        
        # Precompute week of year and sin/cos values
        dt_index = pd.to_datetime(dates)
        week_numbers = dt_index.isocalendar().week.values
        sin_vals = np.sin(2.0 * np.pi * week_numbers / 52.8)
        cos_vals = np.cos(2.0 * np.pi * week_numbers / 52.8)
        
        start_t = self.lag_weeks
        end_t = num_weeks - 67
        
        X_samples = []
        y_samples = {h: [] for h in range(16, 68)}
        
        num_states = len(self.state_list)
        state_dummies = np.eye(num_states)
        
        for t in range(start_t, end_t):
            local_lags_all = inc_array[t-self.lag_weeks+1:t+1, :].T
            spatial_lags_all = neigh_array[t-self.lag_weeks+1:t+1, :].T
            mobility_lags_all = mob_array[t-self.lag_weeks+1:t+1, :].T
            
            for uf_idx in range(num_states):
                local_lags = local_lags_all[uf_idx]
                spatial_lags = spatial_lags_all[uf_idx]
                mobility_lags = mobility_lags_all[uf_idx]
                state_dummy = state_dummies[uf_idx]
                base_feat = np.concatenate([local_lags, spatial_lags, mobility_lags, state_dummy])
                
                for h in range(16, 68):
                    target_idx = t + h
                    sin_sec = sin_vals[target_idx]
                    cos_sec = cos_vals[target_idx]
                    
                    feat = np.concatenate([base_feat, [sin_sec, cos_sec]])
                    X_samples.append((h, feat))
                    
                    y_val = inc_array[target_idx, uf_idx]
                    y_samples[h].append(y_val)
                    
        return X_samples, y_samples

    def fit(self, df_train):
        X_samples, y_samples = self._compute_features_and_targets(df_train, is_training=True)
        
        X_by_h = {h: [] for h in range(16, 68)}
        for h, feat in X_samples:
            X_by_h[h].append(feat)
            
        self.models = {}
        self.residuals = {}
        
        for h in range(16, 68):
            X_h = np.array(X_by_h[h])
            y_h = np.array(y_samples[h])
            
            model = Ridge(alpha=1.0)
            model.fit(X_h, y_h)
            self.models[h] = model
            
            y_pred = model.predict(X_h)
            res = y_h - y_pred
            self.residuals[h] = res
            
    def predict(self, df_target):
        df_full = pd.read_csv('data/processed/state_weekly_features.csv')
        
        target_dates = sorted(df_target['date'].unique())
        min_target_date = target_dates[0]
        
        df_history = df_full[df_full['date'] < min_target_date].copy()
        
        df_inc_pivot, df_neigh_pivot, df_mob_pivot, df_pop_pivot, dates = self._compute_features_and_targets(df_history, is_training=False)
        
        t = len(dates) - 1
        date_t = dates[t]
        
        predictions = []
        
        for uf in self.state_list:
            local_lags = df_inc_pivot.loc[dates[t-self.lag_weeks+1:t+1], uf].values
            spatial_lags = df_neigh_pivot.loc[dates[t-self.lag_weeks+1:t+1], uf].values
            mobility_lags = df_mob_pivot.loc[dates[t-self.lag_weeks+1:t+1], uf].values
            
            state_dummy = np.zeros(len(self.state_list))
            state_dummy[self.state_list.index(uf)] = 1.0
            
            base_feat = np.concatenate([local_lags, spatial_lags, mobility_lags, state_dummy])
            pop = df_pop_pivot.loc[date_t, uf]
            
            for idx, target_date in enumerate(target_dates):
                h = idx + 16
                
                if h in self.models:
                    model = self.models[h]
                    
                    target_dt = pd.to_datetime(target_date)
                    week_of_year = target_dt.isocalendar().week
                    sin_sec = np.sin(2.0 * np.pi * week_of_year / 52.8)
                    cos_sec = np.cos(2.0 * np.pi * week_of_year / 52.8)
                    
                    feat = np.concatenate([base_feat, [sin_sec, cos_sec]]).reshape(1, -1)
                    pred_inc = model.predict(feat)[0]
                    
                    res_h = self.residuals[h]
                    pred_quantiles = {}
                    for q in self.quantiles:
                        q_offset = np.percentile(res_h, q * 100.0)
                        q_inc = max(0.0, pred_inc + q_offset)
                        q_cases = q_inc * pop / 100000.0
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


class CovariateModel:
    """
    Spatio-temporal model incorporating climate reanalysis and ocean indices.
    Features: local lagged cases, spatial lagged cases, local climate lags, ocean indices lags,
              target week climatological normals, target week seasonality, and state dummies.
    Method: Direct multi-step forecasting for horizons 16 to 67 using Random Forest Regressor.
    Quantiles: Estimated via empirical quantiles of training residuals for each horizon.
    """
    def __init__(self):
        self.quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        self.neighbors = NEIGHBORS
        self.models = {}  # Store model for each horizon h (16 to 67)
        self.residuals = {}  # Store training residuals for each horizon h
        self.state_list = sorted(list(NEIGHBORS.keys()))
        self.lag_weeks = 5  # Lags for cases
        
        # Climate variables to use as features
        self.climate_vars = ['temp_med', 'precip_med', 'rel_humid_med']

    def _compute_features_and_targets(self, df, is_training=True):
        df_feats = df.copy()
        df_feats['inc'] = (df_feats['casos'] / df_feats['population']) * 100000.0
        
        # Pivot incidence, cases, population, and climate features
        df_inc_pivot = df_feats.pivot(index='date', columns='uf', values='inc').sort_index()
        df_pop_pivot = df_feats.pivot(index='date', columns='uf', values='population').sort_index()
        df_mob_pivot = df_feats.pivot(index='date', columns='uf', values='mobility_import_risk').sort_index()
        
        # Climate pivots
        climate_pivots = {}
        for var in self.climate_vars:
            climate_pivots[var] = df_feats.pivot(index='date', columns='uf', values=var).sort_index()
            
        # Ocean indicators (same for all states, just weekly values)
        df_ocean_uniq = df_feats[['date', 'enso', 'iod', 'pdo']].drop_duplicates().set_index('date').sort_index()
        
        dates = df_inc_pivot.index.tolist()
        num_weeks = len(dates)
        
        # Precompute neighbor averages for cases
        df_neigh_pivot = pd.DataFrame(index=df_inc_pivot.index, columns=df_inc_pivot.columns)
        for uf in self.state_list:
            neighs = [n for n in self.neighbors[uf] if n in df_inc_pivot.columns]
            if neighs:
                df_neigh_pivot[uf] = df_inc_pivot[neighs].mean(axis=1)
            else:
                df_neigh_pivot[uf] = df_inc_pivot[uf]
                
        # Compute climatological normals (average temperature, precipitation, humidity per state and week of year)
        df_feats['week_of_year'] = pd.to_datetime(df_feats['date']).dt.isocalendar().week.astype(int)
        
        # Dict of (uf, week) -> average climate values
        cli_normals = df_feats.groupby(['uf', 'week_of_year'])[self.climate_vars].mean().to_dict('index')
        
        # Convert week numbers and sin/cos values
        dt_index = pd.to_datetime(dates)
        week_numbers = dt_index.isocalendar().week.values
        sin_vals = np.sin(2.0 * np.pi * week_numbers / 52.8)
        cos_vals = np.cos(2.0 * np.pi * week_numbers / 52.8)
        
        # Precompute normals for all dates and states in a fast numpy array
        num_states = len(self.state_list)
        normals_array = np.zeros((num_weeks, num_states, 3))
        for uf_idx, uf in enumerate(self.state_list):
            for t_idx in range(num_weeks):
                w = week_numbers[t_idx]
                normals = cli_normals.get((uf, w), {'temp_med': 25.0, 'precip_med': 2.0, 'rel_humid_med': 75.0})
                normals_array[t_idx, uf_idx, :] = [normals['temp_med'], normals['precip_med'], normals['rel_humid_med']]
                
        if not is_training:
            return df_inc_pivot, df_neigh_pivot, df_mob_pivot, df_pop_pivot, climate_pivots, df_ocean_uniq, normals_array, week_numbers, dates
            
        # Convert to numpy arrays for speed
        inc_array = df_inc_pivot.values
        neigh_array = df_neigh_pivot.values
        mob_array = df_mob_pivot.values
        
        climate_arrays = {var: climate_pivots[var].values for var in self.climate_vars}
        
        enso_arr = df_ocean_uniq['enso'].values
        iod_arr = df_ocean_uniq['iod'].values
        pdo_arr = df_ocean_uniq['pdo'].values
        
        start_t = max(self.lag_weeks, 12)  # ocean lags require at least 12 weeks of history
        end_t = num_weeks - 67
        
        X_samples = []
        y_samples = {h: [] for h in range(16, 68)}
        
        state_dummies = np.eye(num_states)
        
        for t in range(start_t, end_t):
            local_lags_all = inc_array[t-self.lag_weeks+1:t+1, :].T
            spatial_lags_all = neigh_array[t-self.lag_weeks+1:t+1, :].T
            mobility_lags_all = mob_array[t-self.lag_weeks+1:t+1, :].T
            
            # Climate lags at t, t-4
            cli_lags_all = {}
            for var in self.climate_vars:
                arr = climate_arrays[var]
                cli_lags_all[var] = np.column_stack([arr[t, :], arr[t-4, :]]) # shape: (num_states, 2)
                
            # Ocean lags at t, t-4, t-8, t-12 (same for all states)
            ocean_lags = np.array([
                enso_arr[t], enso_arr[t-4], enso_arr[t-8], enso_arr[t-12],
                iod_arr[t], iod_arr[t-4], iod_arr[t-8], iod_arr[t-12],
                pdo_arr[t], pdo_arr[t-4], pdo_arr[t-8], pdo_arr[t-12]
            ])
            
            for uf_idx in range(num_states):
                local_lags = local_lags_all[uf_idx]
                spatial_lags = spatial_lags_all[uf_idx]
                mobility_lags = mobility_lags_all[uf_idx]
                local_cli = np.concatenate([cli_lags_all[var][uf_idx] for var in self.climate_vars])
                state_dummy = state_dummies[uf_idx]
                
                # Base features (local cases, spatial cases, mobility lags, local climate, ocean oscillations, state ID)
                base_feat = np.concatenate([local_lags, spatial_lags, mobility_lags, local_cli, ocean_lags, state_dummy])
                
                for h in range(16, 68):
                    target_idx = t + h
                    sin_sec = sin_vals[target_idx]
                    cos_sec = cos_vals[target_idx]
                    
                    # Target week climatological normals
                    normals_feat = normals_array[target_idx, uf_idx, :]
                    
                    feat = np.concatenate([base_feat, [sin_sec, cos_sec], normals_feat])
                    X_samples.append((h, feat))
                    
                    y_val = inc_array[target_idx, uf_idx]
                    y_samples[h].append(y_val)
                    
        return X_samples, y_samples

    def fit(self, df_train):
        X_samples, y_samples = self._compute_features_and_targets(df_train, is_training=True)
        
        X_by_h = {h: [] for h in range(16, 68)}
        for h, feat in X_samples:
            X_by_h[h].append(feat)
            
        self.models = {}
        self.residuals = {}
        
        for h in range(16, 68):
            X_h = np.array(X_by_h[h])
            y_h = np.array(y_samples[h])
            
            # Using Random Forest Regressor
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X_h, y_h)
            self.models[h] = model
            
            y_pred = model.predict(X_h)
            res = y_h - y_pred
            self.residuals[h] = res
            
    def predict(self, df_target):
        df_full = pd.read_csv('data/processed/state_weekly_features.csv')
        
        target_dates = sorted(df_target['date'].unique())
        min_target_date = target_dates[0]
        
        df_history = df_full[df_full['date'] < min_target_date].copy()
        
        # Extract features for prediction
        df_inc_pivot, df_neigh_pivot, df_mob_pivot, df_pop_pivot, climate_pivots, df_ocean_uniq, normals_array, week_numbers, dates = \
            self._compute_features_and_targets(df_history, is_training=False)
            
        t = len(dates) - 1
        date_t = dates[t]
        
        inc_array = df_inc_pivot.values
        neigh_array = df_neigh_pivot.values
        mob_array = df_mob_pivot.values
        
        climate_arrays = {var: climate_pivots[var].values for var in self.climate_vars}
        
        enso_arr = df_ocean_uniq['enso'].values
        iod_arr = df_ocean_uniq['iod'].values
        pdo_arr = df_ocean_uniq['pdo'].values
        
        ocean_lags = np.array([
            enso_arr[t], enso_arr[t-4], enso_arr[t-8], enso_arr[t-12],
            iod_arr[t], iod_arr[t-4], iod_arr[t-8], iod_arr[t-12],
            pdo_arr[t], pdo_arr[t-4], pdo_arr[t-8], pdo_arr[t-12]
        ])
        
        predictions = []
        
        for uf_idx in range(len(self.state_list)):
            uf = self.state_list[uf_idx]
            
            local_lags = inc_array[t-self.lag_weeks+1:t+1, uf_idx]
            spatial_lags = neigh_array[t-self.lag_weeks+1:t+1, uf_idx]
            mobility_lags = mob_array[t-self.lag_weeks+1:t+1, uf_idx]
            local_cli = np.concatenate([climate_arrays[var][t-4:t+1:4, uf_idx][::-1] for var in self.climate_vars])
            
            state_dummy = np.zeros(len(self.state_list))
            state_dummy[uf_idx] = 1.0
            
            base_feat = np.concatenate([local_lags, spatial_lags, mobility_lags, local_cli, ocean_lags, state_dummy])
            pop = df_pop_pivot.loc[date_t, uf]
            
            for idx, target_date in enumerate(target_dates):
                h = idx + 16
                
                if h in self.models:
                    model = self.models[h]
                    
                    target_dt = pd.to_datetime(target_date)
                    week_of_year = target_dt.isocalendar().week
                    sin_sec = np.sin(2.0 * np.pi * week_of_year / 52.8)
                    cos_sec = np.cos(2.0 * np.pi * week_of_year / 52.8)
                    
                    hist_idx = np.where(week_numbers == week_of_year)[0]
                    if len(hist_idx) > 0:
                        normals_feat = normals_array[hist_idx[0], uf_idx, :]
                    else:
                        normals_feat = np.array([25.0, 2.0, 75.0])
                    
                    feat = np.concatenate([base_feat, [sin_sec, cos_sec], normals_feat]).reshape(1, -1)
                    pred_inc = model.predict(feat)[0]
                    
                    res_h = self.residuals[h]
                    pred_quantiles = {}
                    for q in self.quantiles:
                        q_offset = np.percentile(res_h, q * 100.0)
                        q_inc = max(0.0, pred_inc + q_offset)
                        q_cases = q_inc * pop / 100000.0
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


class GraphLightGBMModel:
    """
    Spatio-temporal model using LightGBM.
    Features: local lagged cases, spatial lagged cases (average of neighbors), and target week seasonality.
    Method: Direct multi-step forecasting for horizons 16 to 67.
    Quantiles: Estimated via empirical quantiles of training residuals for each horizon.
    """
    def __init__(self):
        self.quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        self.neighbors = NEIGHBORS
        self.models = {}  # Store model for each horizon h (16 to 67)
        self.residuals = {}  # Store training residuals for each horizon h
        self.state_list = sorted(list(NEIGHBORS.keys()))
        self.lag_weeks = 5  # Number of lag weeks to use as features

    def _compute_features_and_targets(self, df, is_training=True):
        df_feats = df.copy()
        df_feats['inc'] = (df_feats['casos'] / df_feats['population']) * 100000.0
        
        # Pivot incidence and cases
        df_inc_pivot = df_feats.pivot(index='date', columns='uf', values='inc').sort_index()
        df_pop_pivot = df_feats.pivot(index='date', columns='uf', values='population').sort_index()
        df_mob_pivot = df_feats.pivot(index='date', columns='uf', values='mobility_import_risk').sort_index()
        
        dates = df_inc_pivot.index.tolist()
        num_weeks = len(dates)
        
        # Precompute neighbor averages
        df_neigh_pivot = pd.DataFrame(index=df_inc_pivot.index, columns=df_inc_pivot.columns)
        for uf in self.state_list:
            neighs = [n for n in self.neighbors[uf] if n in df_inc_pivot.columns]
            if neighs:
                df_neigh_pivot[uf] = df_inc_pivot[neighs].mean(axis=1)
            else:
                df_neigh_pivot[uf] = df_inc_pivot[uf]
                
        if not is_training:
            return df_inc_pivot, df_neigh_pivot, df_mob_pivot, df_pop_pivot, dates
            
        # Convert to numpy arrays for speed
        inc_array = df_inc_pivot.values
        neigh_array = df_neigh_pivot.values
        mob_array = df_mob_pivot.values
        
        # Precompute week of year and sin/cos values
        dt_index = pd.to_datetime(dates)
        week_numbers = dt_index.isocalendar().week.values
        sin_vals = np.sin(2.0 * np.pi * week_numbers / 52.8)
        cos_vals = np.cos(2.0 * np.pi * week_numbers / 52.8)
        
        start_t = self.lag_weeks
        end_t = num_weeks - 67
        
        X_samples = []
        y_samples = {h: [] for h in range(16, 68)}
        
        num_states = len(self.state_list)
        state_dummies = np.eye(num_states)
        
        for t in range(start_t, end_t):
            local_lags_all = inc_array[t-self.lag_weeks+1:t+1, :].T
            spatial_lags_all = neigh_array[t-self.lag_weeks+1:t+1, :].T
            mobility_lags_all = mob_array[t-self.lag_weeks+1:t+1, :].T
            
            for uf_idx in range(num_states):
                local_lags = local_lags_all[uf_idx]
                spatial_lags = spatial_lags_all[uf_idx]
                mobility_lags = mobility_lags_all[uf_idx]
                state_dummy = state_dummies[uf_idx]
                base_feat = np.concatenate([local_lags, spatial_lags, mobility_lags, state_dummy])
                
                for h in range(16, 68):
                    target_idx = t + h
                    sin_sec = sin_vals[target_idx]
                    cos_sec = cos_vals[target_idx]
                    
                    feat = np.concatenate([base_feat, [sin_sec, cos_sec]])
                    X_samples.append((h, feat))
                    
                    y_val = inc_array[target_idx, uf_idx]
                    y_samples[h].append(y_val)
                    
        return X_samples, y_samples

    def fit(self, df_train):
        X_samples, y_samples = self._compute_features_and_targets(df_train, is_training=True)
        
        X_by_h = {h: [] for h in range(16, 68)}
        for h, feat in X_samples:
            X_by_h[h].append(feat)
            
        self.models = {}
        self.residuals = {}
        
        for h in range(16, 68):
            X_h = np.array(X_by_h[h])
            y_h = np.array(y_samples[h])
            
            # Using LightGBM Regressor
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_h, y_h)
            self.models[h] = model
            
            y_pred = model.predict(X_h)
            res = y_h - y_pred
            self.residuals[h] = res
            
    def predict(self, df_target):
        df_full = pd.read_csv('data/processed/state_weekly_features.csv')
        
        target_dates = sorted(df_target['date'].unique())
        min_target_date = target_dates[0]
        
        df_history = df_full[df_full['date'] < min_target_date].copy()
        
        df_inc_pivot, df_neigh_pivot, df_mob_pivot, df_pop_pivot, dates = self._compute_features_and_targets(df_history, is_training=False)
        
        t = len(dates) - 1
        date_t = dates[t]
        
        predictions = []
        
        for uf in self.state_list:
            local_lags = df_inc_pivot.loc[dates[t-self.lag_weeks+1:t+1], uf].values
            spatial_lags = df_neigh_pivot.loc[dates[t-self.lag_weeks+1:t+1], uf].values
            mobility_lags = df_mob_pivot.loc[dates[t-self.lag_weeks+1:t+1], uf].values
            
            state_dummy = np.zeros(len(self.state_list))
            state_dummy[self.state_list.index(uf)] = 1.0
            
            base_feat = np.concatenate([local_lags, spatial_lags, mobility_lags, state_dummy])
            pop = df_pop_pivot.loc[date_t, uf]
            
            for idx, target_date in enumerate(target_dates):
                h = idx + 16
                
                if h in self.models:
                    model = self.models[h]
                    
                    target_dt = pd.to_datetime(target_date)
                    week_of_year = target_dt.isocalendar().week
                    sin_sec = np.sin(2.0 * np.pi * week_of_year / 52.8)
                    cos_sec = np.cos(2.0 * np.pi * week_of_year / 52.8)
                    
                    feat = np.concatenate([base_feat, [sin_sec, cos_sec]]).reshape(1, -1)
                    pred_inc = model.predict(feat)[0]
                    
                    res_h = self.residuals[h]
                    pred_quantiles = {}
                    for q in self.quantiles:
                        q_offset = np.percentile(res_h, q * 100.0)
                        q_inc = max(0.0, pred_inc + q_offset)
                        q_cases = q_inc * pop / 100000.0
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


class STGNN(nn.Module):
    def __init__(self, in_channels, out_horizons, num_quantiles):
        super(STGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, 64)
        self.fc = nn.Linear(64, out_horizons * num_quantiles)
        self.out_horizons = out_horizons
        self.num_quantiles = num_quantiles

    def forward(self, x, edge_index, edge_weight=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        batch_size, num_nodes, num_features = x.shape
        
        out_list = []
        for i in range(batch_size):
            h1 = torch.relu(self.conv1(x[i], edge_index, edge_weight=edge_weight))
            h2 = torch.relu(self.conv2(h1, edge_index, edge_weight=edge_weight))
            out_i = self.fc(h2) # shape: (26, 52 * 9)
            out_list.append(out_i)
            
        out = torch.stack(out_list) # shape: (Batch, 26, 52 * 9)
        out = out.view(batch_size, num_nodes, self.out_horizons, self.num_quantiles)
        
        if batch_size == 1:
            out = out.squeeze(0)
            
        return out


def pinball_loss(preds, targets, quantiles):
    loss = 0.0
    targets = targets.unsqueeze(-1)
    
    for i, q in enumerate(quantiles):
        error = targets - preds[..., i:i+1]
        loss_q = torch.max(q * error, (q - 1) * error)
        loss += loss_q.mean()
        
    return loss / len(quantiles)


class STGCNModel:
    """
    Spatio-temporal Graph Convolutional Network using PyTorch Geometric.
    Trained end-to-end using Pinball Loss to output all 9 quantiles directly.
    """
    def __init__(self):
        self.quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        self.state_list = sorted(list(NEIGHBORS.keys()))
        self.lag_weeks = 5
        self.device = torch.device('cpu')
        
        # Load mobility matrix to define edge index and edge weights
        df_mob = pd.read_csv('data/processed/state_mobility_matrix.csv', index_col=0)
        df_mob = df_mob.reindex(index=self.state_list, columns=self.state_list).fillna(0.0)
        
        edges = []
        weights = []
        for src_uf in self.state_list:
            src_idx = self.state_list.index(src_uf)
            for dest_uf in self.state_list:
                dest_idx = self.state_list.index(dest_uf)
                w_val = df_mob.loc[dest_uf, src_uf]
                if w_val > 0.0:
                    edges.append([src_idx, dest_idx])
                    weights.append(w_val)
                    
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_weight = torch.tensor(weights, dtype=torch.float32).to(self.device)
        self.model = None

    def fit(self, df_train):
        df_feats = df_train.copy()
        df_feats['inc'] = (df_feats['casos'] / df_feats['population']) * 100000.0
        
        df_inc_pivot = df_feats.pivot(index='date', columns='uf', values='inc').sort_index()
        dates = df_inc_pivot.index.tolist()
        num_weeks = len(dates)
        
        inc_array = df_inc_pivot.values
        
        X_train = []
        y_train = []
        for t in range(self.lag_weeks, num_weeks - 67):
            X_t = inc_array[t-self.lag_weeks+1:t+1, :].T # shape: (26, 5)
            y_t = inc_array[t+16:t+68, :].T # shape: (26, 52)
            X_train.append(X_t)
            y_train.append(y_t)
            
        X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
        y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
        
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model = STGNN(in_channels=self.lag_weeks, out_horizons=52, num_quantiles=9)
        self.model.to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        
        self.model.train()
        for epoch in range(60):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(batch_X, self.edge_index, self.edge_weight)
                loss = pinball_loss(preds, batch_y, self.quantiles)
                loss.backward()
                optimizer.step()

    def predict(self, df_target):
        df_full = pd.read_csv('data/processed/state_weekly_features.csv')
        
        target_dates = sorted(df_target['date'].unique())
        min_target_date = target_dates[0]
        
        df_history = df_full[df_full['date'] < min_target_date].copy()
        df_feats = df_history.copy()
        df_feats['inc'] = (df_feats['casos'] / df_feats['population']) * 100000.0
        
        df_inc_pivot = df_feats.pivot(index='date', columns='uf', values='inc').sort_index()
        df_pop_pivot = df_feats.pivot(index='date', columns='uf', values='population').sort_index()
        
        dates = df_inc_pivot.index.tolist()
        t = len(dates) - 1
        date_t = dates[t]
        
        x = torch.tensor(df_inc_pivot.iloc[-self.lag_weeks:].values.T, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(x, self.edge_index, self.edge_weight) # shape: (26, 52, 9)
            
        predictions = []
        for uf_idx, uf in enumerate(self.state_list):
            pop = df_pop_pivot.loc[date_t, uf]
            
            for idx, target_date in enumerate(target_dates):
                h_idx = idx # horizon index in predictions (0 to 51)
                
                # Fetch predictions for state and horizon
                q_preds = preds[uf_idx, h_idx].cpu().numpy() # shape: (9,)
                
                pred_quantiles = {}
                for q_i, q in enumerate(self.quantiles):
                    pred_inc = q_preds[q_i]
                    q_cases = max(0.0, pred_inc * pop / 100000.0)
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
