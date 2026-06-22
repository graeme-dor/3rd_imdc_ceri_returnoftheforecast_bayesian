import os
import pandas as pd
import numpy as np

def compute_wis(y_true, q_dict):
    """
    Computes the Weighted Interval Score (WIS) for a set of predictions.
    y_true: Series of actual cases
    q_dict: Dict of Series for each quantile {0.025: q_0.025, ..., 0.975: q_0.975}
    """
    # Absolute error of the median (q_0.5)
    ae = (y_true - q_dict[0.5]).abs()
    wis = ae.copy()
    
    intervals = [
        (0.25, 0.75, 0.50),  # 50% interval
        (0.10, 0.90, 0.20),  # 80% interval
        (0.05, 0.95, 0.10),  # 90% interval
        (0.025, 0.975, 0.05) # 95% interval
    ]
    
    for l_q, u_q, alpha in intervals:
        l = q_dict[l_q]
        u = q_dict[u_q]
        
        # Penalties for values outside the interval
        under_penalty = (l - y_true).clip(lower=0) * (2.0 / alpha)
        over_penalty = (y_true - u).clip(lower=0) * (2.0 / alpha)
        
        # Interval Score
        is_score = (u - l) + under_penalty + over_penalty
        wis += (alpha / 2.0) * is_score
        
    # Return average WIS
    return wis / (len(intervals) + 1)

def run_validation(model_class, model_name):
    print(f"\nEvaluating {model_name}...")
    df = pd.read_csv('data/processed/state_weekly_features.csv')
    
    results = []
    
    # 4 Validation rounds
    for i in range(1, 5):
        train_col = f'train_{i}'
        target_col = f'target_{i}'
        
        df_train = df[df[train_col] == True].copy()
        df_target = df[df[target_col] == True].copy()
        
        if len(df_target) == 0:
            print(f"Skipping Validation {i} (no target rows).")
            continue
            
        print(f"  Validation {i}: training on {len(df_train)} rows, evaluating on {len(df_target)} rows...")
        
        # Fit model
        model = model_class()
        model.fit(df_train)
        
        # Predict
        df_preds = model.predict(df_target)
        
        # Compute metrics
        y_true = df_preds['casos']
        q_dict = {q: df_preds[f'q_{q}'] for q in [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]}
        
        df_preds['wis'] = compute_wis(y_true, q_dict)
        df_preds['ae'] = (y_true - q_dict[0.5]).abs()
        df_preds['se'] = (y_true - q_dict[0.5]) ** 2
        
        # Compute coverage of 95% interval
        df_preds['cov_95'] = ((y_true >= q_dict[0.025]) & (y_true <= q_dict[0.975])).astype(float)
        
        # Average metrics
        mean_wis = df_preds['wis'].mean()
        mean_ae = df_preds['ae'].mean()
        rmse = np.sqrt(df_preds['se'].mean())
        cov_95 = df_preds['cov_95'].mean()
        
        print(f"    Validation {i} Results: WIS={mean_wis:.2f}, MAE={mean_ae:.2f}, RMSE={rmse:.2f}, 95% Cov={cov_95*100:.1f}%")
        
        results.append({
            'model': model_name,
            'val_round': i,
            'wis': mean_wis,
            'mae': mean_ae,
            'rmse': rmse,
            'cov_95': cov_95
        })
        
        # Save predictions to file for visualization/submission
        val_pred_dir = f'data/predictions/{model_name}'
        os.makedirs(val_pred_dir, exist_ok=True)
        df_preds.to_csv(os.path.join(val_pred_dir, f'val_preds_{i}.csv'), index=False)
        
    df_res = pd.DataFrame(results)
    
    # Save validation metrics summary
    summary_dir = 'data/metrics'
    os.makedirs(summary_dir, exist_ok=True)
    df_res.to_csv(os.path.join(summary_dir, f'{model_name}_summary.csv'), index=False)
    
    # Print overall average metrics
    print(f"\n--- {model_name} Overall Summary ---")
    print(df_res.groupby('model')[['wis', 'mae', 'rmse', 'cov_95']].mean())
    return df_res

if __name__ == '__main__':
    from models import HistoricalMedianModel, SARIMABaselineModel, GraphSpatioTemporalModel, CovariateModel, GraphLightGBMModel, STGCNModel, PyTorchNBGLMM, PyTorchNBGLMMNoCovariates, PyTorchNBGLMMDataDriven, PyTorchNBGLMMRegionalLags, PyTorchNBGLMMInteractions, BayesianThermalModel, BayesianSpatialThermalModel, BayesianGravityThermalModel, BayesianMobilityThermalModel
    
    df_res_baseline = run_validation(HistoricalMedianModel, 'baseline_historical_median')
    df_res_sarima = run_validation(SARIMABaselineModel, 'baseline_sarima')
    df_res_graph = run_validation(GraphSpatioTemporalModel, 'graph_spatiotemporal')
    df_res_lgb = run_validation(GraphLightGBMModel, 'graph_lightgbm')
    df_res_stgcn = run_validation(STGCNModel, 'graph_stgcn')
    df_res_cov = run_validation(CovariateModel, 'covariate_random_forest')
    df_res_bayesian = run_validation(PyTorchNBGLMM, 'bayesian_nb_glmm')
    df_res_bayesian_nocov = run_validation(PyTorchNBGLMMNoCovariates, 'bayesian_nb_glmm_no_covariates')
    df_res_bayesian_datadriven = run_validation(PyTorchNBGLMMDataDriven, 'bayesian_nb_glmm_datadriven')
    df_res_bayesian_regional = run_validation(PyTorchNBGLMMRegionalLags, 'bayesian_nb_glmm_regional_lags')
    df_res_bayesian_interactions = run_validation(PyTorchNBGLMMInteractions, 'bayesian_nb_glmm_interactions')
    df_res_bayesian_thermal = run_validation(BayesianThermalModel, 'bayesian_nb_glmm_thermal')
    df_res_bayesian_spatial_thermal = run_validation(BayesianSpatialThermalModel, 'bayesian_nb_glmm_spatial_thermal')
    df_res_bayesian_gravity_thermal = run_validation(BayesianGravityThermalModel, 'bayesian_nb_glmm_gravity_thermal')
    df_res_bayesian_mobility_thermal = run_validation(BayesianMobilityThermalModel, 'bayesian_nb_glmm_mobility_thermal')
    
    # Print comparison
    print("\n================ COMPARISON SUMMARY ================")
    df_all = pd.concat([
        df_res_baseline, df_res_sarima, df_res_graph, df_res_lgb, 
        df_res_stgcn, df_res_cov, df_res_bayesian, df_res_bayesian_nocov,
        df_res_bayesian_datadriven, df_res_bayesian_regional,
        df_res_bayesian_interactions, df_res_bayesian_thermal,
        df_res_bayesian_spatial_thermal, df_res_bayesian_gravity_thermal,
        df_res_bayesian_mobility_thermal
    ], ignore_index=True)
    comparison = df_all.groupby('model')[['wis', 'mae', 'rmse', 'cov_95']].mean()
    print(comparison)
    comparison.to_csv('data/metrics/comparison_summary.csv')


