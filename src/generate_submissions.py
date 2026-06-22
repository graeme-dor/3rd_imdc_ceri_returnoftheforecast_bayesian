import os
import pandas as pd
import numpy as np

def generate_submission_files(model_class, model_name):
    print(f"\nGenerating submission files for {model_name}...")
    df = pd.read_csv('data/processed/state_weekly_features.csv')
    
    # Map state two-letter codes (uf) to their integer codes (uf_code)
    # The uf_code column is already in state_weekly_features.csv
    uf_to_code = df.groupby('uf')['uf_code'].first().to_dict()
    
    output_dir = f'data/submissions/{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate for the 4 validation rounds
    for i in range(1, 5):
        train_col = f'train_{i}'
        target_col = f'target_{i}'
        
        df_train = df[df[train_col] == True].copy()
        df_target = df[df[target_col] == True].copy()
        
        if len(df_target) == 0:
            print(f"  Skipping Validation {i} (no target rows).")
            continue
            
        print(f"  Validation {i}: training and predicting...")
        
        # Fit model
        model = model_class()
        model.fit(df_train)
        
        # Predict
        df_preds = model.predict(df_target)
        
        # Format columns for Mosqlimate submission
        df_sub = pd.DataFrame()
        df_sub['date'] = df_preds['date']
        df_sub['adm_1'] = df_preds['uf'].map(uf_to_code).astype(int)
        
        # Point prediction (median)
        df_sub['pred'] = df_preds['q_0.5']
        
        # Intervals
        df_sub['lower_50'] = df_preds['q_0.25']
        df_sub['upper_50'] = df_preds['q_0.75']
        
        df_sub['lower_80'] = df_preds['q_0.1']
        df_sub['upper_80'] = df_preds['q_0.9']
        
        df_sub['lower_90'] = df_preds['q_0.05']
        df_sub['upper_90'] = df_preds['q_0.95']
        
        df_sub['lower_95'] = df_preds['q_0.025']
        df_sub['upper_95'] = df_preds['q_0.975']
        
        # Sort values
        df_sub = df_sub.sort_values(['adm_1', 'date']).reset_index(drop=True)
        
        # Save CSV
        out_path = os.path.join(output_dir, f'validation_round_{i}.csv')
        df_sub.to_csv(out_path, index=False)
        print(f"    Saved submission to {out_path} (shape: {df_sub.shape})")

if __name__ == '__main__':
    from models import HistoricalMedianModel, SARIMABaselineModel, GraphSpatioTemporalModel, CovariateModel, GraphLightGBMModel, STGCNModel, BayesianMobilityThermalModel, BayesianThermalModel
    
    generate_submission_files(HistoricalMedianModel, 'baseline_historical_median')
    generate_submission_files(SARIMABaselineModel, 'baseline_sarima')
    generate_submission_files(GraphSpatioTemporalModel, 'graph_spatiotemporal')
    generate_submission_files(GraphLightGBMModel, 'graph_lightgbm')
    generate_submission_files(STGCNModel, 'graph_stgcn')
    generate_submission_files(CovariateModel, 'covariate_random_forest')
    generate_submission_files(BayesianMobilityThermalModel, 'bayesian_nb_glmm_mobility_thermal')
    generate_submission_files(BayesianThermalModel, 'bayesian_nb_glmm_thermal')

