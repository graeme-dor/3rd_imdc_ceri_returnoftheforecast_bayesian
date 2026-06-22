import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.titlesize': 18
})

def generate_presentation_plots():
    print("Generating presentation plots...")
    os.makedirs('plots/presentation', exist_ok=True)
    
    # 1. Load comparison summary
    df_res_baseline = pd.read_csv('data/metrics/baseline_historical_median_summary.csv')
    df_res_sarima = pd.read_csv('data/metrics/baseline_sarima_summary.csv')
    df_res_graph = pd.read_csv('data/metrics/graph_spatiotemporal_summary.csv')
    df_res_lgb = pd.read_csv('data/metrics/graph_lightgbm_summary.csv')
    df_res_stgcn = pd.read_csv('data/metrics/graph_stgcn_summary.csv')
    df_res_cov = pd.read_csv('data/metrics/covariate_random_forest_summary.csv')
    
    df_all = pd.concat([df_res_baseline, df_res_sarima, df_res_graph, df_res_lgb, df_res_stgcn, df_res_cov], ignore_index=True)
    
    # Plot 1: Overall WIS and MAE comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.barplot(data=df_all, x='model', y='wis', errorbar=None, ax=ax1, palette='Set2')
    ax1.set_title("Overall Weighted Interval Score (WIS)\n(Lower is Better)")
    ax1.set_ylabel("Average WIS")
    ax1.set_xlabel("Model")
    # Clean up model names on x-axis
    ax1.set_xticklabels([n.get_text().replace('_', '\n').title() for n in ax1.get_xticklabels()])
    
    sns.barplot(data=df_all, x='model', y='mae', errorbar=None, ax=ax2, palette='Set2')
    ax2.set_title("Overall Mean Absolute Error (MAE)\n(Lower is Better)")
    ax2.set_ylabel("Average MAE")
    ax2.set_xlabel("Model")
    ax2.set_xticklabels([n.get_text().replace('_', '\n').title() for n in ax2.get_xticklabels()])
    
    plt.tight_layout()
    plt.savefig('plots/presentation/overall_comparison.png', dpi=150)
    plt.close()
    
    # 2. Horizon-wise error analysis
    print("  Calculating horizon-wise errors...")
    models = ['baseline_historical_median', 'baseline_sarima', 'graph_spatiotemporal', 'graph_lightgbm', 'graph_stgcn', 'covariate_random_forest']
    horizon_results = []
    
    # We will load target prediction files for Validation 1, 2, 3
    for m in models:
        for val_round in range(1, 4):
            p_path = f'data/predictions/{m}/val_preds_{val_round}.csv'
            if os.path.exists(p_path):
                df_preds = pd.read_csv(p_path)
                
                # For each state, the dates are ordered chronologically.
                # The forecasting horizon is from index 0 (week 16) to 51 (week 67).
                # Let's add a horizon column
                df_preds = df_preds.sort_values(['uf', 'date'])
                df_preds['horizon'] = df_preds.groupby('uf').cumcount() + 16
                
                # Group by horizon and calculate MAE
                for h, group in df_preds.groupby('horizon'):
                    mae_h = (group['casos'] - group['q_0.5']).abs().mean()
                    horizon_results.append({
                        'model': m,
                        'val_round': val_round,
                        'horizon': h,
                        'mae': mae_h
                    })
                    
    df_hor = pd.DataFrame(horizon_results)
    
    # Plot 2: Error as a function of forecasting horizon
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_hor, x='horizon', y='mae', hue='model', linewidth=2.5, marker='o', errorbar=None)
    plt.title("Forecasting Error (MAE) by Horizon Week (Weeks 16 to 67)")
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("Horizon (Weeks Ahead of EW25 Cut-off)")
    plt.legend(title="Model", frameon=True)
    plt.tight_layout()
    plt.savefig('plots/presentation/horizon_error.png', dpi=150)
    plt.close()
    
    # 3. State-level performance analysis
    print("  Calculating state-level performance...")
    state_results = []
    for m in models:
        for val_round in range(1, 4):
            p_path = f'data/predictions/{m}/val_preds_{val_round}.csv'
            if os.path.exists(p_path):
                df_preds = pd.read_csv(p_path)
                # Group by state and calculate MAE
                for uf, group in df_preds.groupby('uf'):
                    mae_s = (group['casos'] - group['q_0.5']).abs().mean()
                    state_results.append({
                        'model': m,
                        'val_round': val_round,
                        'uf': uf,
                        'mae': mae_s
                    })
                    
    df_state = pd.DataFrame(state_results)
    # Average across validation rounds
    df_state_avg = df_state.groupby(['model', 'uf'])['mae'].mean().reset_index()
    
    # Find which model is best (minimum MAE) for each state
    best_models = []
    for uf, group in df_state_avg.groupby('uf'):
        best_row = group.loc[group['mae'].idxmin()]
        best_models.append({
            'uf': uf,
            'best_model': best_row['model'],
            'mae': best_row['mae']
        })
    df_best = pd.DataFrame(best_models)
    
    plt.figure(figsize=(14, 6))
    sns.countplot(data=df_best, x='best_model', palette='Set2')
    plt.title("Number of States where Model Ranked Best (by MAE)")
    plt.ylabel("Count of States")
    plt.xlabel("Best Model")
    # Clean up names
    plt.gca().set_xticklabels([n.get_text().replace('_', '\n').title() for n in plt.gca().get_xticklabels()])
    plt.tight_layout()
    plt.savefig('plots/presentation/best_model_by_state.png', dpi=150)
    plt.close()
    
    print("Presentation plots generated successfully!")

if __name__ == '__main__':
    generate_presentation_plots()
