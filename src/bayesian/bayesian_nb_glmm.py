import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats as st

def override_target_climate(df_combined, train_max_date):
    """
    Overwrites the climate variables in df_combined for dates after train_max_date
    using the state-level weekly delta-adjusted climate forecasts.
    """
    import os
    cutoff_dt = pd.to_datetime(train_max_date)
    cutoffs = {
        pd.to_datetime('2022-06-19'): 'round_1',
        pd.to_datetime('2023-06-18'): 'round_2',
        pd.to_datetime('2024-06-16'): 'round_3',
        pd.to_datetime('2025-06-15'): 'round_4',
        pd.to_datetime('2026-03-08'): 'round_5_forecast_2026_2027'
    }
    # Find closest cutoff to determine the round name
    best_cutoff = min(cutoffs.keys(), key=lambda d: abs((d - cutoff_dt).days))
    round_name = cutoffs[best_cutoff]
    
    # Read the state-level delta-adjusted forecasts
    fc_path = 'data/processed/state_forecasting_climate_processed.csv'
    if not os.path.exists(fc_path):
        raise FileNotFoundError(f"State-level delta-adjusted climate forecasts file not found at {fc_path}. Please run aggregation first.")
    
    df_fc = pd.read_csv(fc_path)
    df_fc_round = df_fc[df_fc['round'] == round_name].copy()
    
    # Define columns to overwrite
    cols_to_overwrite = ['temp_min', 'temp_med', 'rainy_days', 'rel_humid_med']
    
    # Create subset for merge
    df_fc_round_subset = df_fc_round[['uf', 'date'] + cols_to_overwrite].rename(
        columns={col: f'fc_{col}' for col in cols_to_overwrite}
    )
    
    # Merge on state and date
    df_combined = pd.merge(df_combined, df_fc_round_subset, on=['uf', 'date'], how='left')
    
    # Target mask: dates strictly greater than the training cutoff
    target_mask = df_combined['date'] > train_max_date
    
    # Overwrite variables in target period using forecasts
    for col in cols_to_overwrite:
        df_combined.loc[target_mask, col] = df_combined.loc[target_mask, f'fc_{col}'].fillna(df_combined.loc[target_mask, col])
        df_combined = df_combined.drop(columns=[f'fc_{col}'])
        
    return df_combined

class PyTorchNBGLMM:
    """
    Negative Binomial Generalized Linear Mixed Model (NB-GLMM) in PyTorch.
    Fits fixed effects (climate covariates) and random effects (state-specific intercepts
    and Fourier seasonal cycles) via Maximum A Posteriori (MAP) estimation.
    Generates probabilistic forecasts by sampling from the negative binomial predictive distribution.
    """
    def __init__(self, lag_weeks=4):
        self.quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        self.state_list = []
        self.lag_weeks = lag_weeks
        self.device = torch.device('cpu')
        self.params = {}
        
    def _prepare_data(self, df):
        # Sort data
        df = df.sort_values(['uf', 'date']).copy()
        
        # Get unique states
        if len(self.state_list) == 0:
            self.state_list = sorted(df['uf'].unique().tolist())
            
        num_states = len(self.state_list)
        state_to_idx = {uf: idx for idx, uf in enumerate(self.state_list)}
        
        # Convert date to week of year
        df['dt'] = pd.to_datetime(df['date'])
        df['week'] = df['dt'].dt.isocalendar().week.astype(float)
        
        # Fourier features for annual seasonality (52.8 week period)
        df['sin1'] = np.sin(2.0 * np.pi * df['week'] / 52.8)
        df['cos1'] = np.cos(2.0 * np.pi * df['week'] / 52.8)
        df['sin2'] = np.sin(4.0 * np.pi * df['week'] / 52.8)
        df['cos2'] = np.cos(4.0 * np.pi * df['week'] / 52.8)
        
        # Lags for climate variables (e.g., 4 weeks lag)
        for col in ['temp_med', 'precip_med', 'rel_humid_med', 'enso']:
            df[f'{col}_lag'] = df.groupby('uf')[col].shift(self.lag_weeks)
            
        # Drop rows with NaN (from shifts)
        df_clean = df.dropna(subset=['temp_med_lag', 'precip_med_lag', 'rel_humid_med_lag', 'enso_lag']).copy()
        
        # Extract features
        X_fixed = df_clean[['temp_med_lag', 'precip_med_lag', 'rel_humid_med_lag', 'enso_lag']].values
        
        # Scale fixed features
        if not hasattr(self, 'fixed_mean'):
            self.fixed_mean = X_fixed.mean(axis=0)
            self.fixed_std = X_fixed.std(axis=0)
            self.fixed_std[self.fixed_std == 0] = 1.0
            
        X_fixed_scaled = (X_fixed - self.fixed_mean) / self.fixed_std
        
        # Fourier features (used for both fixed and random effects)
        X_fourier = df_clean[['sin1', 'cos1', 'sin2', 'cos2']].values
        
        # State indices
        state_idxs = df_clean['uf'].map(state_to_idx).values
        
        # Target cases and population
        y = df_clean['casos'].values
        pop = df_clean['population'].values
        
        return {
            'X_fixed': torch.tensor(X_fixed_scaled, dtype=torch.float32),
            'X_fourier': torch.tensor(X_fourier, dtype=torch.float32),
            'state_idxs': torch.tensor(state_idxs, dtype=torch.long),
            'y': torch.tensor(y, dtype=torch.float32),
            'pop': torch.tensor(pop, dtype=torch.float32),
            'df_clean': df_clean
        }

    def fit(self, df_train, num_epochs=1000):
        data = self._prepare_data(df_train)
        X_fixed = data['X_fixed']
        X_fourier = data['X_fourier']
        state_idxs = data['state_idxs']
        y = data['y']
        pop = data['pop']
        
        num_states = len(self.state_list)
        num_fixed_feats = X_fixed.shape[1]
        num_fourier_feats = X_fourier.shape[1]
        
        # Model parameters to optimize via MAP
        # Fixed effects (intercept + scaled climate covariates + fourier base)
        beta_fixed = nn.Parameter(torch.zeros(num_fixed_feats))
        beta_fourier = nn.Parameter(torch.zeros(num_fourier_feats))
        intercept = nn.Parameter(torch.log(y.mean() / pop.mean()))
        
        # Random effects: (random intercept + random fourier slopes) per state
        # shape: (num_states, 5)
        u_rand = nn.Parameter(torch.zeros(num_states, 1 + num_fourier_feats))
        
        # Prior standard deviations (regularization strength)
        log_sigma_fixed = nn.Parameter(torch.tensor(0.0))  # fixed effects prior variance
        log_sigma_u = nn.Parameter(torch.zeros(1 + num_fourier_feats))  # random effects prior variances
        
        # Negative Binomial dispersion parameter (log scale)
        log_phi = nn.Parameter(torch.tensor(0.0))
        
        optimizer = optim.Adam([
            beta_fixed, beta_fourier, intercept, u_rand, log_sigma_fixed, log_sigma_u, log_phi
        ], lr=0.01)
        
        # Training loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Predict log-mean
            log_mu_fixed = intercept + torch.matmul(X_fixed, beta_fixed) + torch.matmul(X_fourier, beta_fourier)
            
            # Random intercept + random fourier slopes
            u_intercept = u_rand[state_idxs, 0]
            u_fourier = torch.sum(u_rand[state_idxs, 1:] * X_fourier, dim=1)
            
            log_mu = torch.log(pop) + log_mu_fixed + u_intercept + u_fourier
            mu = torch.exp(log_mu).clamp(min=1e-5)
            
            # Negative Binomial Negative Log-Likelihood
            phi = torch.exp(log_phi)
            lgamma_y_phi = torch.lgamma(y + 1.0/phi)
            lgamma_phi = torch.lgamma(1.0/phi)
            lgamma_y = torch.lgamma(y + 1.0)
            
            nll = -(lgamma_y_phi - lgamma_phi - lgamma_y + y * torch.log(phi * mu) - (y + 1.0/phi) * torch.log(1.0 + phi * mu))
            loss_data = nll.mean()
            
            # Priors (L2 MAP penalties)
            sigma_fixed = torch.exp(log_sigma_fixed)
            sigma_u = torch.exp(log_sigma_u)
            
            loss_prior_fixed = 0.5 * torch.sum(beta_fixed**2) / (sigma_fixed**2) + 0.5 * torch.sum(beta_fourier**2) / (sigma_fixed**2)
            loss_prior_u = 0.5 * torch.sum(u_rand**2 / (sigma_u**2))
            loss_prior_sigma = 0.1 * (log_sigma_fixed**2 + torch.sum(log_sigma_u**2))
            
            # Total Loss
            loss = loss_data + 1e-4 * (loss_prior_fixed + loss_prior_u + loss_prior_sigma)
            
            loss.backward()
            optimizer.step()
            
        # Save fitted parameters
        self.params = {
            'beta_fixed': beta_fixed.detach().clone(),
            'beta_fourier': beta_fourier.detach().clone(),
            'intercept': intercept.detach().clone(),
            'u_rand': u_rand.detach().clone(),
            'phi': torch.exp(log_phi).detach().clone(),
            'sigma_fixed': torch.exp(log_sigma_fixed).detach().clone(),
            'sigma_u': torch.exp(log_sigma_u).detach().clone()
        }
        
    def predict(self, df_target):
        # Read full features dataset to correctly construct lag values for target dates
        df_full = pd.read_csv('data/processed/state_weekly_features.csv')
        target_dates = sorted(df_target['date'].unique())
        min_target_date = target_dates[0]
        
        df_history = df_full[df_full['date'] < min_target_date].copy()
        
        # Combine history and target to preserve shifts
        df_combined = pd.concat([df_history, df_target], ignore_index=True)
        
        data = self._prepare_data(df_combined)
        df_clean = data['df_clean']
        
        # Keep only target rows
        target_mask = df_clean['date'].isin(target_dates)
        df_clean_target = df_clean[target_mask].copy()
        
        idx_target = np.where(target_mask)[0]
        X_fixed_t = data['X_fixed'][idx_target]
        X_fourier_t = data['X_fourier'][idx_target]
        state_idxs_t = data['state_idxs'][idx_target]
        pop_t = data['pop'][idx_target]
        
        beta_fixed = self.params['beta_fixed']
        beta_fourier = self.params['beta_fourier']
        intercept = self.params['intercept']
        u_rand = self.params['u_rand']
        phi = self.params['phi'].item()
        
        # Predict target mean
        log_mu_fixed = intercept + torch.matmul(X_fixed_t, beta_fixed) + torch.matmul(X_fourier_t, beta_fourier)
        u_intercept = u_rand[state_idxs_t, 0]
        u_fourier = torch.sum(u_rand[state_idxs_t, 1:] * X_fourier_t, dim=1)
        
        log_mu = torch.log(pop_t) + log_mu_fixed + u_intercept + u_fourier
        mu = torch.exp(log_mu).numpy()
        
        predictions = []
        n = 1.0 / phi
        
        # Sample predictions from the Negative Binomial distribution to calculate quantiles
        for i in range(len(df_clean_target)):
            row = df_clean_target.iloc[i]
            mu_i = mu[i]
            p_i = 1.0 / (1.0 + phi * mu_i)
            
            # Draw 5000 samples to construct smooth quantiles
            samples = st.nbinom.rvs(n, p_i, size=5000)
            q_vals = np.percentile(samples, [q * 100 for q in self.quantiles])
            
            row_dict = {
                'uf': row['uf'],
                'date': row['date'],
                'casos': row['casos']
            }
            for q_i, q in enumerate(self.quantiles):
                row_dict[f'q_{q}'] = q_vals[q_i]
                
            predictions.append(row_dict)
            
        df_out = pd.DataFrame(predictions)
        
        # Merge back with target to guarantee consistency
        df_final = pd.merge(df_target[['uf', 'date', 'casos']], df_out, on=['uf', 'date'], how='left')
        
        # Fill missing values
        fill_cols = [f'q_{q}' for q in self.quantiles]
        df_final[fill_cols] = df_final[fill_cols].fillna(0.0)
        df_final['casos'] = df_final['casos_x']
        df_final = df_final.drop(columns=['casos_x', 'casos_y'])
        
        return df_final

class PyTorchNBGLMMNoCovariates:
    """
    Negative Binomial Generalized Linear Mixed Model (NB-GLMM) WITHOUT climate covariates.
    Serves as a secondary baseline model, using only population offset, a global intercept,
    and state-specific random Fourier seasonal cycles.
    """
    def __init__(self):
        self.quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        self.state_list = []
        self.device = torch.device('cpu')
        self.params = {}
        
    def _prepare_data(self, df):
        # Sort data
        df = df.sort_values(['uf', 'date']).copy()
        
        # Get unique states
        if len(self.state_list) == 0:
            self.state_list = sorted(df['uf'].unique().tolist())
            
        num_states = len(self.state_list)
        state_to_idx = {uf: idx for idx, uf in enumerate(self.state_list)}
        
        # Convert date to week of year
        df['dt'] = pd.to_datetime(df['date'])
        df['week'] = df['dt'].dt.isocalendar().week.astype(float)
        
        # Fourier features for annual seasonality (52.8 week period)
        df['sin1'] = np.sin(2.0 * np.pi * df['week'] / 52.8)
        df['cos1'] = np.cos(2.0 * np.pi * df['week'] / 52.8)
        df['sin2'] = np.sin(4.0 * np.pi * df['week'] / 52.8)
        df['cos2'] = np.cos(4.0 * np.pi * df['week'] / 52.8)
        
        # Fourier features (used for both fixed and random effects)
        X_fourier = df[['sin1', 'cos1', 'sin2', 'cos2']].values
        
        # State indices
        state_idxs = df['uf'].map(state_to_idx).values
        
        # Target cases and population
        y = df['casos'].values
        pop = df['population'].values
        
        return {
            'X_fourier': torch.tensor(X_fourier, dtype=torch.float32),
            'state_idxs': torch.tensor(state_idxs, dtype=torch.long),
            'y': torch.tensor(y, dtype=torch.float32),
            'pop': torch.tensor(pop, dtype=torch.float32),
            'df_clean': df
        }

    def fit(self, df_train, num_epochs=1000):
        data = self._prepare_data(df_train)
        X_fourier = data['X_fourier']
        state_idxs = data['state_idxs']
        y = data['y']
        pop = data['pop']
        
        num_states = len(self.state_list)
        num_fourier_feats = X_fourier.shape[1]
        
        # Model parameters
        beta_fourier = nn.Parameter(torch.zeros(num_fourier_feats))
        intercept = nn.Parameter(torch.log(y.mean() / pop.mean()))
        
        # Random effects: (random intercept + random fourier slopes) per state
        u_rand = nn.Parameter(torch.zeros(num_states, 1 + num_fourier_feats))
        
        # Prior standard deviations
        log_sigma_fixed = nn.Parameter(torch.tensor(0.0))  # fixed fourier prior variance
        log_sigma_u = nn.Parameter(torch.zeros(1 + num_fourier_feats))  # random effects prior variances
        log_phi = nn.Parameter(torch.tensor(0.0))  # negative binomial dispersion
        
        optimizer = optim.Adam([
            beta_fourier, intercept, u_rand, log_sigma_fixed, log_sigma_u, log_phi
        ], lr=0.01)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Predict log-mean (no climate covariates)
            log_mu_fixed = intercept + torch.matmul(X_fourier, beta_fourier)
            u_intercept = u_rand[state_idxs, 0]
            u_fourier = torch.sum(u_rand[state_idxs, 1:] * X_fourier, dim=1)
            
            log_mu = torch.log(pop) + log_mu_fixed + u_intercept + u_fourier
            mu = torch.exp(log_mu).clamp(min=1e-5)
            
            # Negative Binomial NLL
            phi = torch.exp(log_phi)
            lgamma_y_phi = torch.lgamma(y + 1.0/phi)
            lgamma_phi = torch.lgamma(1.0/phi)
            lgamma_y = torch.lgamma(y + 1.0)
            
            nll = -(lgamma_y_phi - lgamma_phi - lgamma_y + y * torch.log(phi * mu) - (y + 1.0/phi) * torch.log(1.0 + phi * mu))
            loss_data = nll.mean()
            
            # Priors
            sigma_fixed = torch.exp(log_sigma_fixed)
            sigma_u = torch.exp(log_sigma_u)
            
            loss_prior_fixed = 0.5 * torch.sum(beta_fourier**2) / (sigma_fixed**2)
            loss_prior_u = 0.5 * torch.sum(u_rand**2 / (sigma_u**2))
            loss_prior_sigma = 0.1 * (log_sigma_fixed**2 + torch.sum(log_sigma_u**2))
            
            loss = loss_data + 1e-4 * (loss_prior_fixed + loss_prior_u + loss_prior_sigma)
            
            loss.backward()
            optimizer.step()
            
        self.params = {
            'beta_fourier': beta_fourier.detach().clone(),
            'intercept': intercept.detach().clone(),
            'u_rand': u_rand.detach().clone(),
            'phi': torch.exp(log_phi).detach().clone(),
            'sigma_fixed': torch.exp(log_sigma_fixed).detach().clone(),
            'sigma_u': torch.exp(log_sigma_u).detach().clone()
        }
        
    def predict(self, df_target):
        # We don't need history to shift climate since there are no covariates!
        # We can predict directly using target dates and population
        data = self._prepare_data(df_target)
        X_fourier_t = data['X_fourier']
        state_idxs_t = data['state_idxs']
        pop_t = data['pop']
        
        beta_fourier = self.params['beta_fourier']
        intercept = self.params['intercept']
        u_rand = self.params['u_rand']
        phi = self.params['phi'].item()
        
        # Predict target mean
        log_mu_fixed = intercept + torch.matmul(X_fourier_t, beta_fourier)
        u_intercept = u_rand[state_idxs_t, 0]
        u_fourier = torch.sum(u_rand[state_idxs_t, 1:] * X_fourier_t, dim=1)
        
        log_mu = torch.log(pop_t) + log_mu_fixed + u_intercept + u_fourier
        mu = torch.exp(log_mu).numpy()
        
        predictions = []
        n = 1.0 / phi
        
        for i in range(len(df_target)):
            row = df_target.iloc[i]
            mu_i = mu[i]
            p_i = 1.0 / (1.0 + phi * mu_i)
            
            samples = st.nbinom.rvs(n, p_i, size=5000)
            q_vals = np.percentile(samples, [q * 100 for q in self.quantiles])
            
            row_dict = {
                'uf': row['uf'],
                'date': row['date'],
                'casos': row['casos']
            }
            for q_i, q in enumerate(self.quantiles):
                row_dict[f'q_{q}'] = q_vals[q_i]
                
            predictions.append(row_dict)
            
        df_out = pd.DataFrame(predictions)
        
        df_final = pd.merge(df_target[['uf', 'date', 'casos']], df_out, on=['uf', 'date'], how='left')
        fill_cols = [f'q_{q}' for q in self.quantiles]
        df_final[fill_cols] = df_final[fill_cols].fillna(0.0)
        df_final['casos'] = df_final['casos_x']
        df_final = df_final.drop(columns=['casos_x', 'casos_y'])
        
        return df_final

class PyTorchNBGLMMThermal:
    """
    Negative Binomial Generalized Linear Mixed Model (NB-GLMM) in PyTorch.
    Fits fixed effects (Brière-transformed climate suitability and other climate covariates) 
    and random effects (state-specific intercepts and Fourier seasonal cycles) via 
    Maximum A Posteriori (MAP) estimation.
    """
    def __init__(self):
        self.quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        self.state_list = []
        self.device = torch.device('cpu')
        self.params = {}
        
    def _prepare_data(self, df):
        # Sort data
        df = df.sort_values(['uf', 'date']).copy()
        
        # Get unique states
        if len(self.state_list) == 0:
            self.state_list = sorted(df['uf'].unique().tolist())
            
        num_states = len(self.state_list)
        state_to_idx = {uf: idx for idx, uf in enumerate(self.state_list)}
        
        # Convert date to week of year
        df['dt'] = pd.to_datetime(df['date'])
        df['week'] = df['dt'].dt.isocalendar().week.astype(float)
        
        # Fourier features for annual seasonality (52.8 week period)
        df['sin1'] = np.sin(2.0 * np.pi * df['week'] / 52.8)
        df['cos1'] = np.cos(2.0 * np.pi * df['week'] / 52.8)
        df['sin2'] = np.sin(4.0 * np.pi * df['week'] / 52.8)
        df['cos2'] = np.cos(4.0 * np.pi * df['week'] / 52.8)
        
        # Brière suitability equation helper
        def briere_suitability(temp):
            t_min = 17.8
            t_max = 34.6
            val = temp * (temp - t_min) * np.sqrt(np.maximum(0.0, t_max - temp))
            return np.where((temp >= t_min) & (temp <= t_max), val, 0.0)
            
        # Compute suitability transformations
        df['ts_min'] = briere_suitability(df['temp_min'])
        df['ts_med'] = briere_suitability(df['temp_med'])
        
        # Shift transformed suitability indices and other climate features by their optimal lags
        df['ts_min_lag_11'] = df.groupby('uf')['ts_min'].shift(11)
        df['ts_med_lag_14'] = df.groupby('uf')['ts_med'].shift(14)
        df['rainy_days_lag_9'] = df.groupby('uf')['rainy_days'].shift(9)
        df['rel_humid_med_lag_4'] = df.groupby('uf')['rel_humid_med'].shift(4)
        
        lag_cols = [
            'ts_min_lag_11', 'ts_med_lag_14', 'rainy_days_lag_9', 'rel_humid_med_lag_4'
        ]
        
        # Drop rows with NaN (from shifts)
        df_clean = df.dropna(subset=lag_cols).copy()
        
        # Extract features
        X_fixed = df_clean[lag_cols].values
        
        # Scale fixed features
        if not hasattr(self, 'fixed_mean'):
            self.fixed_mean = X_fixed.mean(axis=0)
            self.fixed_std = X_fixed.std(axis=0)
            self.fixed_std[self.fixed_std == 0] = 1.0
            
        X_fixed_scaled = (X_fixed - self.fixed_mean) / self.fixed_std
        
        # Fourier features (used for both fixed and random effects)
        X_fourier = df_clean[['sin1', 'cos1', 'sin2', 'cos2']].values
        
        # State indices
        state_idxs = df_clean['uf'].map(state_to_idx).values
        
        # Target cases and population
        y = df_clean['casos'].values
        pop = df_clean['population'].values
        
        return {
            'X_fixed': torch.tensor(X_fixed_scaled, dtype=torch.float32),
            'X_fourier': torch.tensor(X_fourier, dtype=torch.float32),
            'state_idxs': torch.tensor(state_idxs, dtype=torch.long),
            'y': torch.tensor(y, dtype=torch.float32),
            'pop': torch.tensor(pop, dtype=torch.float32),
            'df_clean': df_clean,
            'lag_cols': lag_cols
        }

    def fit(self, df_train, num_epochs=1000):
        self.train_max_date = df_train['date'].max()
        df_train = df_train.copy()
        df_train['dt'] = pd.to_datetime(df_train['date'])
        df_train['week'] = df_train['dt'].dt.isocalendar().week.astype(int)
        df_train['temp_diff'] = df_train['temp_med'] - df_train['temp_min']
        self.temp_diff_normals = df_train.groupby(['uf', 'week'])['temp_diff'].mean().to_dict()
        self.temp_med_normals = df_train.groupby(['uf', 'week'])['temp_med'].mean().to_dict()
        self.temp_min_normals = df_train.groupby(['uf', 'week'])['temp_min'].mean().to_dict()
        self.rainy_days_normals = df_train.groupby(['uf', 'week'])['rainy_days'].mean().to_dict()
        self.rel_humid_med_normals = df_train.groupby(['uf', 'week'])['rel_humid_med'].mean().to_dict()
        
        # Apply optimal state-level hybrid mask for Zika (2016-2018) and COVID-19 (2019-2021) anomaly periods
        dt = pd.to_datetime(df_train['date'])
        is_zika = (dt >= pd.to_datetime('2016-10-01')) & (dt <= pd.to_datetime('2018-09-30'))
        is_covid = (dt >= pd.to_datetime('2019-10-01')) & (dt <= pd.to_datetime('2021-09-30'))
        is_anomaly = is_zika | is_covid
        
        states_to_mask = ['GO', 'MT', 'BA', 'PI', 'AP', 'PA', 'RO', 'MG', 'RJ', 'SP', 'RS', 'SC']
        mask_row = is_anomaly & df_train['uf'].isin(states_to_mask)
        df_train_filtered = df_train[~mask_row].copy()
        
        data = self._prepare_data(df_train_filtered)
        X_fixed = data['X_fixed']
        X_fourier = data['X_fourier']
        state_idxs = data['state_idxs']
        y = data['y']
        pop = data['pop']
        
        num_states = len(self.state_list)
        num_fixed_feats = X_fixed.shape[1]
        num_fourier_feats = X_fourier.shape[1]
        
        # Model parameters to optimize via MAP
        beta_fixed = nn.Parameter(torch.zeros(num_fixed_feats))
        beta_fourier = nn.Parameter(torch.zeros(num_fourier_feats))
        intercept = nn.Parameter(torch.log(y.mean() / pop.mean()))
        
        # Random effects
        u_rand = nn.Parameter(torch.zeros(num_states, 1 + num_fourier_feats))
        
        # Prior standard deviations
        log_sigma_fixed = nn.Parameter(torch.tensor(0.0))
        log_sigma_u = nn.Parameter(torch.zeros(1 + num_fourier_feats))
        log_phi = nn.Parameter(torch.tensor(0.0))
        
        optimizer = optim.Adam([
            beta_fixed, beta_fourier, intercept, u_rand, log_sigma_fixed, log_sigma_u, log_phi
        ], lr=0.01)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            log_mu_fixed = intercept + torch.matmul(X_fixed, beta_fixed) + torch.matmul(X_fourier, beta_fourier)
            u_intercept = u_rand[state_idxs, 0]
            u_fourier = torch.sum(u_rand[state_idxs, 1:] * X_fourier, dim=1)
            
            log_mu = torch.log(pop) + log_mu_fixed + u_intercept + u_fourier
            mu = torch.exp(log_mu).clamp(min=1e-5)
            
            phi = torch.exp(log_phi)
            lgamma_y_phi = torch.lgamma(y + 1.0/phi)
            lgamma_phi = torch.lgamma(1.0/phi)
            lgamma_y = torch.lgamma(y + 1.0)
            
            nll = -(lgamma_y_phi - lgamma_phi - lgamma_y + y * torch.log(phi * mu) - (y + 1.0/phi) * torch.log(1.0 + phi * mu))
            loss_data = nll.mean()
            
            sigma_fixed = torch.exp(log_sigma_fixed)
            sigma_u = torch.exp(log_sigma_u)
            
            loss_prior_fixed = 0.5 * torch.sum(beta_fixed**2) / (sigma_fixed**2) + 0.5 * torch.sum(beta_fourier**2) / (sigma_fixed**2)
            loss_prior_u = 0.5 * torch.sum(u_rand**2 / (sigma_u**2))
            loss_prior_sigma = 0.1 * (log_sigma_fixed**2 + torch.sum(log_sigma_u**2))
            
            loss = loss_data + 1e-4 * (loss_prior_fixed + loss_prior_u + loss_prior_sigma)
            
            loss.backward()
            optimizer.step()
            
        self.params = {
            'beta_fixed': beta_fixed.detach().clone(),
            'beta_fourier': beta_fourier.detach().clone(),
            'intercept': intercept.detach().clone(),
            'u_rand': u_rand.detach().clone(),
            'phi': torch.exp(log_phi).detach().clone(),
            'sigma_fixed': torch.exp(log_sigma_fixed).detach().clone(),
            'sigma_u': torch.exp(log_sigma_u).detach().clone()
        }
        
    def predict(self, df_target):
        df_full = pd.read_csv('data/processed/state_weekly_features.csv')
        df_full = df_full.sort_values(['uf', 'date']).reset_index(drop=True)
        
        target_dates = sorted(df_target['date'].unique())
        min_target_date = target_dates[0]
        
        df_history = df_full[df_full['date'] < min_target_date].copy()
        df_combined = pd.concat([df_history, df_target], ignore_index=True)
        
        df_combined = override_target_climate(df_combined, self.train_max_date)
        return self._predict_with_combined(df_combined, df_target)
        
    def _predict_with_combined(self, df_combined, df_target):
        target_dates = sorted(df_target['date'].unique())
        
        # Prepare data on the combined dataframe using the prepared suitability indices
        data = self._prepare_data(df_combined)
        df_clean = data['df_clean']
        lag_cols = data['lag_cols']
        
        target_mask = df_clean['date'].isin(target_dates)
        df_clean_target = df_clean[target_mask].copy()
        
        idx_target = np.where(target_mask)[0]
        X_fixed_t = data['X_fixed'][idx_target]
        X_fourier_t = data['X_fourier'][idx_target]
        state_idxs_t = data['state_idxs'][idx_target]
        pop_t = data['pop'][idx_target]
        
        beta_fixed = self.params['beta_fixed']
        beta_fourier = self.params['beta_fourier']
        intercept = self.params['intercept']
        u_rand = self.params['u_rand']
        phi = self.params['phi'].item()
        
        log_mu_fixed = intercept + torch.matmul(X_fixed_t, beta_fixed) + torch.matmul(X_fourier_t, beta_fourier)
        u_intercept = u_rand[state_idxs_t, 0]
        u_fourier = torch.sum(u_rand[state_idxs_t, 1:] * X_fourier_t, dim=1)
        
        log_mu = torch.log(pop_t) + log_mu_fixed + u_intercept + u_fourier
        mu = torch.exp(log_mu).numpy()
        
        predictions = []
        n = 1.0 / phi
        
        for i in range(len(df_clean_target)):
            row = df_clean_target.iloc[i]
            mu_i = mu[i]
            p_i = 1.0 / (1.0 + phi * mu_i)
            
            samples = st.nbinom.rvs(n, p_i, size=5000)
            q_vals = np.percentile(samples, [q * 100 for q in self.quantiles])
            
            row_dict = {
                'uf': row['uf'],
                'date': row['date'],
                'casos': row['casos']
            }
            for q_i, q in enumerate(self.quantiles):
                row_dict[f'q_{q}'] = q_vals[q_i]
                
            predictions.append(row_dict)
            
        df_out = pd.DataFrame(predictions)
        df_final = pd.merge(df_target[['uf', 'date', 'casos']], df_out, on=['uf', 'date'], how='left')
        
        fill_cols = [f'q_{q}' for q in self.quantiles]
        df_final[fill_cols] = df_final[fill_cols].fillna(0.0)
        df_final['casos'] = df_final['casos_x']
        df_final = df_final.drop(columns=['casos_x', 'casos_y'])
        
        return df_final

