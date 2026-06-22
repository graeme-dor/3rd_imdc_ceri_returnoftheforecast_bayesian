import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Setup styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.titlesize': 18
})

def produce_decomposition():
    print("Loading data...")
    df = pd.read_csv('data/processed/state_weekly_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. National decomposition (sum of all states)
    print("Decomposing national weekly cases...")
    df_nat = df.groupby('date')['casos'].sum().sort_index()
    
    decomp_nat = seasonal_decompose(df_nat, model='additive', period=52)
    
    # Plot National
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Additive Seasonal Decomposition - National Dengue Cases (Brazil)", y=0.98)
    
    axes[0].plot(df_nat.index, decomp_nat.observed, color='royalblue', linewidth=1.5)
    axes[0].set_title("Observed Cases", loc='left')
    axes[0].set_ylabel("Cases")
    
    axes[1].plot(df_nat.index, decomp_nat.trend, color='darkorange', linewidth=2.0)
    axes[1].set_title("Trend", loc='left')
    axes[1].set_ylabel("Cases")
    
    axes[2].plot(df_nat.index, decomp_nat.seasonal, color='forestgreen', linewidth=1.5)
    axes[2].set_title("Seasonality (Period = 52 weeks)", loc='left')
    axes[2].set_ylabel("Cases")
    
    axes[3].scatter(df_nat.index, decomp_nat.resid, color='crimson', s=10, alpha=0.6)
    axes[3].axhline(0, color='black', linestyle='--', linewidth=1.0)
    axes[3].set_title("Residuals (Noise / Anomalies)", loc='left')
    axes[3].set_ylabel("Cases")
    axes[3].set_xlabel("Year")
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/national_decomposition.png', dpi=150)
    plt.close()
    print("Saved national decomposition to plots/national_decomposition.png")
    
    # 2. São Paulo (SP) decomposition
    print("Decomposing São Paulo (SP) weekly cases...")
    df_sp = df[df['uf'] == 'SP'].sort_values('date').set_index('date')['casos']
    
    decomp_sp = seasonal_decompose(df_sp, model='additive', period=52)
    
    # Plot SP
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Additive Seasonal Decomposition - São Paulo (SP) Dengue Cases", y=0.98)
    
    axes[0].plot(df_sp.index, decomp_sp.observed, color='royalblue', linewidth=1.5)
    axes[0].set_title("Observed Cases", loc='left')
    axes[0].set_ylabel("Cases")
    
    axes[1].plot(df_sp.index, decomp_sp.trend, color='darkorange', linewidth=2.0)
    axes[1].set_title("Trend", loc='left')
    axes[1].set_ylabel("Cases")
    
    axes[2].plot(df_sp.index, decomp_sp.seasonal, color='forestgreen', linewidth=1.5)
    axes[2].set_title("Seasonality (Period = 52 weeks)", loc='left')
    axes[2].set_ylabel("Cases")
    
    axes[3].scatter(df_sp.index, decomp_sp.resid, color='crimson', s=10, alpha=0.6)
    axes[3].axhline(0, color='black', linestyle='--', linewidth=1.0)
    axes[3].set_title("Residuals (Noise / Anomalies)", loc='left')
    axes[3].set_ylabel("Cases")
    axes[3].set_xlabel("Year")
    
    plt.tight_layout()
    plt.savefig('plots/SP_decomposition.png', dpi=150)
    plt.close()
    print("Saved SP decomposition to plots/SP_decomposition.png")

if __name__ == '__main__':
    produce_decomposition()
