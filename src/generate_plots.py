import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.titlesize': 18
})

def generate_comparison_plots():
    print("Generating comparison plots...")
    os.makedirs('plots/validation_comparison', exist_ok=True)
    
    models = ['baseline_historical_median', 'baseline_sarima', 'graph_spatiotemporal', 'graph_lightgbm', 'graph_stgcn', 'covariate_random_forest']
    
    # We will plot a few representative states from different regions:
    # SP (Southeast), RJ (Southeast), BA (Northeast), AM (North), CE (Northeast), PR (South)
    test_states = ['SP', 'RJ', 'BA', 'AM', 'CE', 'PR']
    
    # Loop over validation rounds 1 to 3 (which have full 52-week seasons)
    for val_round in range(1, 4):
        print(f"  Processing Validation {val_round}...")
        
        # Load predictions for all models
        model_preds = {}
        for m in models:
            p_path = f'data/predictions/{m}/val_preds_{val_round}.csv'
            if os.path.exists(p_path):
                model_preds[m] = pd.read_csv(p_path)
                
        if not model_preds:
            print(f"    No predictions found for Validation {val_round}.")
            continue
            
        # Plot for each test state
        for uf in test_states:
            fig, axes = plt.subplots(6, 1, figsize=(14, 24), sharex=True, sharey=True)
            fig.suptitle(f"State {uf} - Validation Round {val_round} Predictions vs. Actuals", y=0.98)
            
            for idx, m in enumerate(models):
                ax = axes[idx]
                if m not in model_preds:
                    ax.set_title(f"{m} (Predictions Missing)")
                    continue
                    
                df_uf = model_preds[m][model_preds[m]['uf'] == uf].copy()
                if len(df_uf) == 0:
                    ax.set_title(f"{m} - No data for {uf}")
                    continue
                    
                df_uf = df_uf.sort_values('date').reset_index(drop=True)
                dates = df_uf['date']
                x = np.arange(len(dates))
                
                # Plot intervals (95%, 80%, 50%)
                ax.fill_between(x, df_uf['q_0.025'], df_uf['q_0.975'], color='#FFCDD2', alpha=0.4, label='95% Interval')
                ax.fill_between(x, df_uf['q_0.1'], df_uf['q_0.9'], color='#EF9A9A', alpha=0.5, label='80% Interval')
                ax.fill_between(x, df_uf['q_0.25'], df_uf['q_0.75'], color='#E57373', alpha=0.6, label='50% Interval')
                
                # Plot median and actuals
                ax.plot(x, df_uf['q_0.5'], color='#C62828', linewidth=2.0, label='Predicted Median')
                ax.plot(x, df_uf['casos'], color='black', linewidth=2.5, linestyle='--', label='Actual Cases')
                
                ax.set_title(f"Model: {m.replace('_', ' ').title()}", fontsize=14, loc='left')
                ax.set_ylabel('Weekly Cases')
                
                if idx == 0:
                    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
                    
            # X-axis formatting
            ax = axes[-1]
            tick_spacing = max(1, len(dates) // 10)
            ax.set_xticks(x[::tick_spacing])
            ax.set_xticklabels(dates[::tick_spacing], rotation=45, ha='right')
            ax.set_xlabel('Date (Epiweek Sunday)')
            
            plt.tight_layout()
            save_path = f'plots/validation_comparison/val_{val_round}_{uf}_comparison.png'
            plt.savefig(save_path, dpi=150)
            plt.close()
            
    print("Comparison plots generated successfully!")

if __name__ == '__main__':
    generate_comparison_plots()
