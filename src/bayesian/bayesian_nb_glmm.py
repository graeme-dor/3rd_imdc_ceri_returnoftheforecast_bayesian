import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats as st

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

class PyTorchNBGLMMDataDriven:
    """
    Negative Binomial Generalized Linear Mixed Model (NB-GLMM) in PyTorch.
    Fits fixed effects (data-driven climate covariates) and random effects (state-specific intercepts
    and Fourier seasonal cycles) via Maximum A Posteriori (MAP) estimation.
    Uses custom lags for each covariate as determined by empirical data-driven analysis.
    """
    def __init__(self, feature_lags=None):
        self.quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        self.state_list = []
        self.device = torch.device('cpu')
        self.params = {}
        if feature_lags is None:
            self.feature_lags = {
                'temp_min': 11,
                'temp_med': 12,
                'rainy_days': 9,
                'rel_humid_med': 4
            }
        else:
            self.feature_lags = feature_lags
        
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
        
        # Lags for data-driven climate variables
        lag_cols = []
        for col, lag in self.feature_lags.items():
            lag_col = f'{col}_lag_{lag}'
            df[lag_col] = df.groupby('uf')[col].shift(lag)
            lag_cols.append(lag_col)
            
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
        target_dates = sorted(df_target['date'].unique())
        min_target_date = target_dates[0]
        
        df_history = df_full[df_full['date'] < min_target_date].copy()
        df_combined = pd.concat([df_history, df_target], ignore_index=True)
        
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

class PyTorchNBGLMMRegionalLags:
    """
    Negative Binomial Generalized Linear Mixed Model (NB-GLMM) in PyTorch.
    Fits fixed effects (regional-lag climate covariates) and random effects (state-specific intercepts
    and Fourier seasonal cycles) via Maximum A Posteriori (MAP) estimation.
    Uses custom regional lags for each covariate as determined by empirical regional correlation.
    """
    def __init__(self):
        import json
        import os
        self.quantiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        self.state_list = []
        self.device = torch.device('cpu')
        self.params = {}
        
        # State to region mapping
        self.state_to_region = {
            'AC': 'Norte', 'AM': 'Norte', 'AP': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
            'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
            'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MS': 'Centro-Oeste', 'MT': 'Centro-Oeste',
            'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
            'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
        }
        
        if os.path.exists('data/processed/regional_lags.json'):
            with open('data/processed/regional_lags.json', 'r') as f:
                self.regional_lags = json.load(f)
        else:
            self.regional_lags = {
                "Norte": {"temp_min": 12, "temp_med": 1, "rainy_days": 6, "rel_humid_med": 3},
                "Nordeste": {"temp_min": 12, "temp_med": 1, "rainy_days": 9, "rel_humid_med": 5},
                "Centro-Oeste": {"temp_min": 11, "temp_med": 12, "rainy_days": 8, "rel_humid_med": 5},
                "Sudeste": {"temp_min": 10, "temp_med": 10, "rainy_days": 9, "rel_humid_med": 5},
                "Sul": {"temp_min": 9, "temp_med": 10, "rainy_days": 9, "rel_humid_med": 12}
            }
            
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
        
        # Covariates we use
        cov_cols = ['temp_min', 'temp_med', 'rainy_days', 'rel_humid_med']
        lag_cols = [f'{col}_lag' for col in cov_cols]
        
        # Shift each covariate by its state-specific region-lag
        for col in cov_cols:
            df[f'{col}_lag'] = np.nan
            for uf, region in self.state_to_region.items():
                lag = self.regional_lags[region][col]
                mask = df['uf'] == uf
                df.loc[mask, f'{col}_lag'] = df.loc[mask].groupby('uf')[col].shift(lag)
                
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
        target_dates = sorted(df_target['date'].unique())
        min_target_date = target_dates[0]
        
        df_history = df_full[df_full['date'] < min_target_date].copy()
        df_combined = pd.concat([df_history, df_target], ignore_index=True)
        
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

class PyTorchNBGLMMInteractions:
    """
    Negative Binomial Generalized Linear Mixed Model (NB-GLMM) in PyTorch.
    Fits fixed effects (covariates + interaction terms) and random effects (state-specific intercepts
    and Fourier seasonal cycles) via Maximum A Posteriori (MAP) estimation.
    Includes data-driven optimal lags and pairwise environmental interaction features.
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
        
        # Shift data-driven climate variables by their optimal lags
        df['temp_min_lag_11'] = df.groupby('uf')['temp_min'].shift(11)
        df['temp_med_lag_12'] = df.groupby('uf')['temp_med'].shift(12)
        df['rainy_days_lag_9'] = df.groupby('uf')['rainy_days'].shift(9)
        df['rel_humid_med_lag_4'] = df.groupby('uf')['rel_humid_med'].shift(4)
        
        # Compute environmental interaction terms
        df['temp_rain_inter'] = df['temp_min_lag_11'] * df['rainy_days_lag_9']
        df['temp_humid_inter'] = df['temp_med_lag_12'] * df['rel_humid_med_lag_4']
        
        lag_cols = [
            'temp_min_lag_11', 'temp_med_lag_12', 'rainy_days_lag_9', 'rel_humid_med_lag_4',
            'temp_rain_inter', 'temp_humid_inter'
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
        target_dates = sorted(df_target['date'].unique())
        min_target_date = target_dates[0]
        
        df_history = df_full[df_full['date'] < min_target_date].copy()
        df_combined = pd.concat([df_history, df_target], ignore_index=True)
        
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
            t_min = 13.39
            t_max = 38.46
            val = temp * (temp - t_min) * np.sqrt(np.maximum(0.0, t_max - temp))
            return np.where((temp >= t_min) & (temp <= t_max), val, 0.0)
            
        # Compute suitability transformations
        df['ts_min'] = briere_suitability(df['temp_min'])
        df['ts_med'] = briere_suitability(df['temp_med'])
        
        # Shift transformed suitability indices and other climate features by their optimal lags
        df['ts_min_lag_11'] = df.groupby('uf')['ts_min'].shift(11)
        df['ts_med_lag_12'] = df.groupby('uf')['ts_med'].shift(12)
        df['rainy_days_lag_9'] = df.groupby('uf')['rainy_days'].shift(9)
        df['rel_humid_med_lag_4'] = df.groupby('uf')['rel_humid_med'].shift(4)
        
        lag_cols = [
            'ts_min_lag_11', 'ts_med_lag_12', 'rainy_days_lag_9', 'rel_humid_med_lag_4'
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
        # Apply optimal state-level hybrid mask for Zika (2016-2018) and COVID-19 (2019-2021) anomaly periods
        df_train = df_train.copy()
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
        target_dates = sorted(df_target['date'].unique())
        min_target_date = target_dates[0]
        
        df_history = df_full[df_full['date'] < min_target_date].copy()
        df_combined = pd.concat([df_history, df_target], ignore_index=True)
        
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

class PyTorchNBGLMMSpatialThermal:
    """
    Negative Binomial Generalized Linear Mixed Model (NB-GLMM) in PyTorch.
    Fits fixed effects (Brière-transformed temperature suitability only) 
    and random effects (state-specific spatial ICAR intercepts and independent 
    Fourier seasonal cycles) via MAP estimation.
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
        
        # Build undirected graph edges for spatial ICAR prior
        neighbors_dict = {
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
        
        edges_src = []
        edges_dst = []
        for src_uf in self.state_list:
            if src_uf in neighbors_dict:
                src_idx = state_to_idx[src_uf]
                for dest_uf in neighbors_dict[src_uf]:
                    if dest_uf in state_to_idx:
                        dest_idx = state_to_idx[dest_uf]
                        if src_idx < dest_idx:
                            edges_src.append(src_idx)
                            edges_dst.append(dest_idx)
                            
        self.edges_src = torch.tensor(edges_src, dtype=torch.long, device=self.device)
        self.edges_dst = torch.tensor(edges_dst, dtype=torch.long, device=self.device)
        
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
            t_min = 13.39
            t_max = 38.46
            val = temp * (temp - t_min) * np.sqrt(np.maximum(0.0, t_max - temp))
            return np.where((temp >= t_min) & (temp <= t_max), val, 0.0)
            
        # Compute suitability transformations
        df['ts_min'] = briere_suitability(df['temp_min'])
        df['ts_med'] = briere_suitability(df['temp_med'])
        
        # Shift transformed suitability indices by their optimal lags
        df['ts_min_lag_11'] = df.groupby('uf')['ts_min'].shift(11)
        df['ts_med_lag_12'] = df.groupby('uf')['ts_med'].shift(12)
        
        # We only keep temperature suitability features to maximize performance based on validation tests
        lag_cols = ['ts_min_lag_11', 'ts_med_lag_12']
        
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
        beta_fixed = nn.Parameter(torch.zeros(num_fixed_feats))
        beta_fourier = nn.Parameter(torch.zeros(num_fourier_feats))
        intercept = nn.Parameter(torch.log(y.mean() / pop.mean()))
        
        # Random effects: first column is spatial random intercept, rest are independent Fourier terms
        u_rand = nn.Parameter(torch.zeros(num_states, 1 + num_fourier_feats))
        
        # Prior standard deviations/scales to optimize
        log_sigma_fixed = nn.Parameter(torch.tensor(0.0))
        log_sigma_u_fourier = nn.Parameter(torch.zeros(num_fourier_feats))
        log_tau = nn.Parameter(torch.tensor(0.0)) # Spatial variance scale
        log_phi = nn.Parameter(torch.tensor(0.0)) # NB dispersion
        
        optimizer = optim.Adam([
            beta_fixed, beta_fourier, intercept, u_rand, log_sigma_fixed, log_sigma_u_fourier, log_tau, log_phi
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
            sigma_u_fourier = torch.exp(log_sigma_u_fourier)
            tau = torch.exp(log_tau)
            
            # Priors: L2 for fixed effects
            loss_prior_fixed = 0.5 * torch.sum(beta_fixed**2) / (sigma_fixed**2) + 0.5 * torch.sum(beta_fourier**2) / (sigma_fixed**2)
            
            # Spatial ICAR prior for intercepts (first column of u_rand)
            diffs = u_rand[self.edges_src, 0] - u_rand[self.edges_dst, 0]
            loss_prior_u_spatial = 0.5 * torch.sum(diffs**2) / (tau**2)
            
            # Sum-to-zero constraint to ensure spatial intercept is centered and identifiable
            loss_sum_zero = 0.5 * (torch.sum(u_rand[:, 0])**2) * 10.0
            
            # Independent Gaussian priors for Fourier random effects (remaining columns of u_rand)
            loss_prior_u_fourier = 0.5 * torch.sum(u_rand[:, 1:]**2 / (sigma_u_fourier**2))
            
            loss_prior_sigma = 0.1 * (log_sigma_fixed**2 + torch.sum(log_sigma_u_fourier**2) + log_tau**2)
            
            loss = loss_data + 1e-4 * (
                loss_prior_fixed + loss_prior_u_spatial + loss_prior_u_fourier + loss_prior_sigma + loss_sum_zero
            )
            
            loss.backward()
            optimizer.step()
            
        self.params = {
            'beta_fixed': beta_fixed.detach().clone(),
            'beta_fourier': beta_fourier.detach().clone(),
            'intercept': intercept.detach().clone(),
            'u_rand': u_rand.detach().clone(),
            'phi': torch.exp(log_phi).detach().clone(),
            'sigma_fixed': torch.exp(log_sigma_fixed).detach().clone(),
            'sigma_u_fourier': torch.exp(log_sigma_u_fourier).detach().clone(),
            'tau': torch.exp(log_tau).detach().clone()
        }
        
    def predict(self, df_target):
        df_full = pd.read_csv('data/processed/state_weekly_features.csv')
        target_dates = sorted(df_target['date'].unique())
        min_target_date = target_dates[0]
        
        df_history = df_full[df_full['date'] < min_target_date].copy()
        df_combined = pd.concat([df_history, df_target], ignore_index=True)
        
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

class PyTorchNBGLMMGravityThermal:
    """
    Negative Binomial Generalized Linear Mixed Model (NB-GLMM) in PyTorch.
    Fits fixed effects (Brière-transformed temperature suitability only) 
    and random effects (state-specific spatial ICAR intercepts on a Gravity Model Network 
    and independent Fourier seasonal cycles) via MAP estimation.
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
        
        # Capitals coordinates (latitude, longitude)
        coords = {
            'AC': (-9.974, -67.808), 'AL': (-9.666, -35.735), 'AM': (-3.100, -60.016),
            'AP': (0.034, -51.069), 'BA': (-12.971, -38.510), 'CE': (-3.731, -38.526),
            'DF': (-15.780, -47.930), 'GO': (-16.678, -49.253), 'MA': (-2.530, -44.302),
            'MG': (-19.920, -43.940), 'MS': (-20.442, -54.646), 'MT': (-15.601, -56.096),
            'PA': (-1.455, -48.503), 'PB': (-7.115, -34.863), 'PE': (-8.053, -34.881),
            'PI': (-5.089, -42.801), 'PR': (-25.427, -49.273), 'RJ': (-22.903, -43.209),
            'RN': (-5.795, -35.209), 'RO': (-8.761, -63.903), 'RR': (2.819, -60.673),
            'RS': (-30.034, -51.217), 'SC': (-27.596, -48.549), 'SE': (-10.911, -37.073),
            'SP': (-23.548, -46.636), 'TO': (-10.167, -48.331)
        }
        
        # Calculate state mean populations to use as gravity masses
        pop_dict = df.groupby('uf')['population'].mean().to_dict()
        
        # Calculate pairwise distances (Haversine)
        dist = np.zeros((num_states, num_states))
        for i in range(num_states):
            uf_i = self.state_list[i]
            lat1, lon1 = coords[uf_i]
            for j in range(num_states):
                if i == j:
                    dist[i, j] = 1e9 # Prevent self-loops
                    continue
                uf_j = self.state_list[j]
                lat2, lon2 = coords[uf_j]
                dlat = np.radians(lat2 - lat1)
                dlon = np.radians(lon2 - lon1)
                a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                dist[i, j] = 2 * np.arcsin(np.sqrt(a)) * 6371.0 # Distance in km
                
        # Compute gravity forces
        G = np.zeros((num_states, num_states))
        for i in range(num_states):
            uf_i = self.state_list[i]
            p_i = pop_dict.get(uf_i, 1.0e6)
            for j in range(num_states):
                if i == j:
                    G[i, j] = 0.0
                    continue
                uf_j = self.state_list[j]
                p_j = pop_dict.get(uf_j, 1.0e6)
                G[i, j] = (p_i * p_j) / (dist[i, j]**2)
                
        # Build undirected graph edges based on top K strongest gravity connections for each state
        edges_set = set()
        K = 3
        for i in range(num_states):
            top_k_idxs = np.argsort(G[i])[::-1][:K]
            for j in top_k_idxs:
                edge = tuple(sorted((i, j)))
                edges_set.add(edge)
                
        edges_src = [edge[0] for edge in edges_set]
        edges_dst = [edge[1] for edge in edges_set]
        
        self.edges_src = torch.tensor(edges_src, dtype=torch.long, device=self.device)
        self.edges_dst = torch.tensor(edges_dst, dtype=torch.long, device=self.device)
        
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
            t_min = 13.39
            t_max = 38.46
            val = temp * (temp - t_min) * np.sqrt(np.maximum(0.0, t_max - temp))
            return np.where((temp >= t_min) & (temp <= t_max), val, 0.0)
            
        # Compute suitability transformations
        df['ts_min'] = briere_suitability(df['temp_min'])
        df['ts_med'] = briere_suitability(df['temp_med'])
        
        # Shift transformed suitability indices by their optimal lags
        df['ts_min_lag_11'] = df.groupby('uf')['ts_min'].shift(11)
        df['ts_med_lag_12'] = df.groupby('uf')['ts_med'].shift(12)
        
        # We only keep temperature suitability features to maximize performance based on validation tests
        lag_cols = ['ts_min_lag_11', 'ts_med_lag_12']
        
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
        beta_fixed = nn.Parameter(torch.zeros(num_fixed_feats))
        beta_fourier = nn.Parameter(torch.zeros(num_fourier_feats))
        intercept = nn.Parameter(torch.log(y.mean() / pop.mean()))
        
        # Random effects: first column is spatial random intercept, rest are independent Fourier terms
        u_rand = nn.Parameter(torch.zeros(num_states, 1 + num_fourier_feats))
        
        # Prior standard deviations/scales to optimize
        log_sigma_fixed = nn.Parameter(torch.tensor(0.0))
        log_sigma_u_fourier = nn.Parameter(torch.zeros(num_fourier_feats))
        log_tau = nn.Parameter(torch.tensor(0.0)) # Spatial variance scale
        log_phi = nn.Parameter(torch.tensor(0.0)) # NB dispersion
        
        optimizer = optim.Adam([
            beta_fixed, beta_fourier, intercept, u_rand, log_sigma_fixed, log_sigma_u_fourier, log_tau, log_phi
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
            sigma_u_fourier = torch.exp(log_sigma_u_fourier)
            tau = torch.exp(log_tau)
            
            # Priors: L2 for fixed effects
            loss_prior_fixed = 0.5 * torch.sum(beta_fixed**2) / (sigma_fixed**2) + 0.5 * torch.sum(beta_fourier**2) / (sigma_fixed**2)
            
            # Spatial ICAR prior for intercepts (first column of u_rand)
            diffs = u_rand[self.edges_src, 0] - u_rand[self.edges_dst, 0]
            loss_prior_u_spatial = 0.5 * torch.sum(diffs**2) / (tau**2)
            
            # Sum-to-zero constraint to ensure spatial intercept is centered and identifiable
            loss_sum_zero = 0.5 * (torch.sum(u_rand[:, 0])**2) * 10.0
            
            # Independent Gaussian priors for Fourier random effects (remaining columns of u_rand)
            loss_prior_u_fourier = 0.5 * torch.sum(u_rand[:, 1:]**2 / (sigma_u_fourier**2))
            
            loss_prior_sigma = 0.1 * (log_sigma_fixed**2 + torch.sum(log_sigma_u_fourier**2) + log_tau**2)
            
            loss = loss_data + 1e-4 * (
                loss_prior_fixed + loss_prior_u_spatial + loss_prior_u_fourier + loss_prior_sigma + loss_sum_zero
            )
            
            loss.backward()
            optimizer.step()
            
        self.params = {
            'beta_fixed': beta_fixed.detach().clone(),
            'beta_fourier': beta_fourier.detach().clone(),
            'intercept': intercept.detach().clone(),
            'u_rand': u_rand.detach().clone(),
            'phi': torch.exp(log_phi).detach().clone(),
            'sigma_fixed': torch.exp(log_sigma_fixed).detach().clone(),
            'sigma_u_fourier': torch.exp(log_sigma_u_fourier).detach().clone(),
            'tau': torch.exp(log_tau).detach().clone()
        }
        
    def predict(self, df_target):
        df_full = pd.read_csv('data/processed/state_weekly_features.csv')
        target_dates = sorted(df_target['date'].unique())
        min_target_date = target_dates[0]
        
        df_history = df_full[df_full['date'] < min_target_date].copy()
        df_combined = pd.concat([df_history, df_target], ignore_index=True)
        
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


class PyTorchNBGLMMMobilityThermal:
    """
    Negative Binomial Generalized Linear Mixed Model (NB-GLMM) in PyTorch.
    Fits fixed effects (Brière-transformed temperature suitability only) 
    and random effects (state-specific spatial ICAR intercepts on our Empirical Mobility Network 
    and independent Fourier seasonal cycles) via MAP estimation.
    Has zero case leakage since it has no case-based covariates in fixed effects.
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
        
        # Load empirical mobility matrix
        df_mob = pd.read_csv('data/processed/state_mobility_matrix.csv', index_col=0)
        df_mob = df_mob.reindex(index=self.state_list, columns=self.state_list).fillna(0.0)
        W = df_mob.values
        
        # Build undirected graph edges based on top K strongest travel connections for each state
        edges_set = set()
        K = 3
        for i in range(num_states):
            # Combined undirected flow strength between state i and j
            flow_strength = W[i, :] + W[:, i]
            flow_strength[i] = 0.0 # Set self-loop to zero
            top_k_idxs = np.argsort(flow_strength)[::-1][:K]
            for j in top_k_idxs:
                edge = tuple(sorted((i, j)))
                edges_set.add(edge)
                
        edges_src = [edge[0] for edge in edges_set]
        edges_dst = [edge[1] for edge in edges_set]
        
        self.edges_src = torch.tensor(edges_src, dtype=torch.long, device=self.device)
        self.edges_dst = torch.tensor(edges_dst, dtype=torch.long, device=self.device)
        
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
            t_min = 13.39
            t_max = 38.46
            val = temp * (temp - t_min) * np.sqrt(np.maximum(0.0, t_max - temp))
            return np.where((temp >= t_min) & (temp <= t_max), val, 0.0)
            
        # Compute suitability transformations
        df['ts_min'] = briere_suitability(df['temp_min'])
        df['ts_med'] = briere_suitability(df['temp_med'])
        
        # Shift transformed suitability indices by their optimal lags
        df['ts_min_lag_11'] = df.groupby('uf')['ts_min'].shift(11)
        df['ts_med_lag_12'] = df.groupby('uf')['ts_med'].shift(12)
        
        lag_cols = ['ts_min_lag_11', 'ts_med_lag_12']
        
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
        # Apply optimal state-level hybrid mask for Zika (2016-2018) and COVID-19 (2019-2021) anomaly periods
        df_train = df_train.copy()
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
        
        # Random effects: spatial random intercept + independent Fourier seasonal terms
        u_rand = nn.Parameter(torch.zeros(num_states, 1 + num_fourier_feats))
        
        # Prior standard deviations/scales to optimize
        log_sigma_fixed = nn.Parameter(torch.tensor(0.0))
        log_sigma_u_fourier = nn.Parameter(torch.zeros(num_fourier_feats))
        log_tau = nn.Parameter(torch.tensor(0.0)) # Spatial variance scale
        log_phi = nn.Parameter(torch.tensor(0.0)) # NB dispersion
        
        optimizer = optim.Adam([
            beta_fixed, beta_fourier, intercept, u_rand, log_sigma_fixed, log_sigma_u_fourier, log_tau, log_phi
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
            sigma_u_fourier = torch.exp(log_sigma_u_fourier)
            tau = torch.exp(log_tau)
            
            # Priors: L2 for fixed effects
            loss_prior_fixed = 0.5 * torch.sum(beta_fixed**2) / (sigma_fixed**2) + 0.5 * torch.sum(beta_fourier**2) / (sigma_fixed**2)
            
            # Spatial ICAR prior for intercepts (first column of u_rand)
            diffs = u_rand[self.edges_src, 0] - u_rand[self.edges_dst, 0]
            loss_prior_u_spatial = 0.5 * torch.sum(diffs**2) / (tau**2)
            
            # Sum-to-zero constraint to ensure spatial intercept is centered and identifiable
            loss_sum_zero = 0.5 * (torch.sum(u_rand[:, 0])**2) * 10.0
            
            # Independent Gaussian priors for Fourier random effects (remaining columns of u_rand)
            loss_prior_u_fourier = 0.5 * torch.sum(u_rand[:, 1:]**2 / (sigma_u_fourier**2))
            
            loss_prior_sigma = 0.1 * (log_sigma_fixed**2 + torch.sum(log_sigma_u_fourier**2) + log_tau**2)
            
            loss = loss_data + 1e-4 * (
                loss_prior_fixed + loss_prior_u_spatial + loss_prior_u_fourier + loss_prior_sigma + loss_sum_zero
            )
            
            loss.backward()
            optimizer.step()
            
        self.params = {
            'beta_fixed': beta_fixed.detach().clone(),
            'beta_fourier': beta_fourier.detach().clone(),
            'intercept': intercept.detach().clone(),
            'u_rand': u_rand.detach().clone(),
            'phi': torch.exp(log_phi).detach().clone(),
            'sigma_fixed': torch.exp(log_sigma_fixed).detach().clone(),
            'sigma_u_fourier': torch.exp(log_sigma_u_fourier).detach().clone(),
            'tau': torch.exp(log_tau).detach().clone()
        }

    def predict(self, df_target):
        df_full = pd.read_csv('data/processed/state_weekly_features.csv')
        target_dates = sorted(df_target['date'].unique())
        min_target_date = target_dates[0]
        
        df_history = df_full[df_full['date'] < min_target_date].copy()
        df_combined = pd.concat([df_history, df_target], ignore_index=True)
        
        data = self._prepare_data(df_combined)
        df_clean = data['df_clean']
        
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
        df_final = df_final.sort_values(['uf', 'date']).reset_index(drop=True)
        df_final = df_final.drop(columns=['casos_x', 'casos_y'])
        
        return df_final
