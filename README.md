# 3rd IMDC Submission - Team CERI - Return of the Forecast

This repository contains the source code, data preprocessing, and predictions submitted by Team **CERI - Return of the Forecast** for the **3rd Infodengue–Mosqlimate Dengue Challenge (IMDC) 2026** (Dengue forecasting at the state level in Brazil).

---

## 1. Team and Contributors

**CERI - Return of the Forecast**

Carlin Foka<sup>1</sup>, Jenicca Poongavanan<sup>1</sup>, Monika Moir<sup>1</sup>, Graeme Dor<sup>1</sup>, Houriiyah Tegally<sup>1</sup>, Isabela Albuquerque<sup>2</sup>, Petar Veličković<sup>2,3</sup>

<sup>1</sup> Centre for Epidemic Response and Innovation (CERI), School of Data Science and Computational Thinking, Stellenbosch University, Stellenbosch, South Africa  
<sup>2</sup> Google DeepMind  
<sup>3</sup> University of Cambridge, Cambridge, United Kingdom

---

## 2. Repository Structure

A description of the contents and purpose of each directory and file in this repository:

*   `src/`: Core Python source files for preprocessing, model fitting, and formatting.
    *   `src/bayesian/bayesian_nb_glmm.py`: Implementation of the Negative Binomial Generalized Linear Mixed Model (NB-GLMM) in PyTorch. Fits parameters for fixed effects, random state-specific intercepts, and random seasonal cycles.
    *   `src/models.py`: Python wrapper registering the Bayesian Thermal Model.
    *   `src/preprocess_data.py`: Preprocessing script that aggregates municipality case data to the state level, computes population-weighted climate features, and generates the preprocessed dataset.
    *   `src/evaluate.py`: Retro-validation script to evaluate WIS, MAE, and RMSE across target seasons.
    *   `src/generate_submissions.py`: Generates standardized submission-ready CSV files containing the predictions and intervals for the retrospective validation targets.
*   `data/processed/`: Preprocessed datasets used for training and forecasting.
    *   `data/processed/state_weekly_features.csv`: Aggregated historical state-level case and climate features.
    *   `data/processed/state_forecasting_climate_processed.csv`: The forecasted climate dataset containing only the columns utilized in the model (`round`, `uf`, `date`, `temp_min`, `temp_med`, `rainy_days`, `rel_humid_med`).
*   `data/submissions/`: Standardized retrospective validation forecast outputs.
    *   `data/submissions/bayesian_nb_glmm_thermal/`: Submission CSVs containing probabilistic forecasts (median, 50%, 80%, 90%, and 95% prediction intervals) for the retrospective validation targets.
*   `requirements.txt`: Python package requirements and environment dependencies.

---

## 3. Libraries and Dependencies

All data processing, training, and forecast generation were performed in Python. The key libraries and dependencies are:

*   **PyTorch** (`torch`): Used to optimize model parameters (fixed and random effects) via Maximum A Posteriori (MAP) estimation.
*   **Pandas & NumPy** (`pandas`, `numpy`): Used for data manipulation, aggregation, and formatting.
*   **SciPy** (`scipy`): Used to draw negative binomial random samples for probabilistic interval and quantile estimations.

---

## 4. Data and Variables

The model utilizes the official challenge datasets, prepared as follows:

*   **Epidemiological Cases:** Dengue probable cases from Infodengue aggregated to the state (UF) level (excluding Espírito Santo).
*   **Demographic Offsets:** State populations (from DATASUS) used as log-scale offsets to model incidence rates.
*   **Population-Weighted Climate Features:** Mean temperature (`temp_med`), minimum temperature (`temp_min`), relative humidity (`rel_humid_med`), and rainy days aggregated using **population-weighted averages** to represent weather exposure in populated areas.
*   **Brière Temperature Suitability:** Non-linear vector suitability transformations applied to population-weighted temperatures based on literature-derived vector physiological thresholds:
    *   Lower temperature limit ($T_{min}$): $17.8^\circ\text{C}$
    *   Upper temperature limit ($T_{max}$): $34.6^\circ\text{C}$
*   **Optimal Feature Lags:** Shifted transformed suitability indices and other climate features by their optimal lags:
    *   Brière-transformed minimum temperature suitability: 11-week lag (`ts_min_lag_11`)
    *   Brière-transformed median temperature suitability: 14-week lag (`ts_med_lag_14`)
    *   Rainy days: 9-week lag (`rainy_days_lag_9`)
    *   Relative humidity median: 4-week lag (`rel_humid_med_lag_4`)

---

## 5. Model Training and Forecasting

The model is the **Bayesian Negative Binomial Generalized Linear Mixed Model (NB-GLMM) with Thermal Suitabilities and Climate Covariates**:

*   **Model Framework:** Implement a Negative Binomial Generalized Linear Mixed Model (NB-GLMM) in PyTorch.
*   **Fixed Effects:** Relies on Brière-transformed temperature suitability indices (`ts_min_lag_11`, `ts_med_lag_14`) and other climate covariates (`rainy_days_lag_9`, `rel_humid_med_lag_4`) along with Fourier seasonal components to capture global annual cycles.
*   **Random Effects:** Fits state-specific intercepts and seasonal Fourier cycles via MAP estimation to capture local baseline incidence, climate deviations, and regional variations in seasonality.
*   **State-Level Anomaly Masking:** The training dataset excludes Zika (2016–2018) and COVID-19 (2019–2021) anomaly periods only for the 12 states where anomalies historically degraded forecasting baselines (MG, RJ, SP, RS, SC, GO, MT, BA, PI, AP, PA, RO). The remaining 14 states retain their full history to maximize estimation stability.
*   **Forecast Generation:** Probabilistic forecast bands are generated by sampling from the fitted negative binomial distribution using the optimized dispersion parameter $\phi$.
