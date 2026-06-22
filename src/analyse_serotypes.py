import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gisaid_parser import UF_TO_REGION

# Setup plot styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

def parse_official_serotypes(csv_path):
    """Parses Serotype_Progression.csv from comma-separated strings to wide DataFrame."""
    print(f"Loading official serotypes from: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    records = []
    for idx, row in df_raw.iterrows():
        variant = row['variant']
        months = [m.strip() for m in row['yearmth'].split(',')]
        pcts = [float(p.strip()) for p in row['percentage'].split(',')]
        for m, p in zip(months, pcts):
            records.append({'month': m, 'variant': variant, 'prop': p})
            
    df_long = pd.DataFrame(records)
    df_pivot = df_long.pivot(index='month', columns='variant', values='prop').fillna(0).reset_index()
    # Normalize proportions to sum to 1
    cols = ['DENV1', 'DENV2', 'DENV3', 'DENV4']
    for c in cols:
        if c not in df_pivot.columns:
            df_pivot[c] = 0.0
    row_sums = df_pivot[cols].sum(axis=1)
    df_pivot[cols] = df_pivot[cols].div(row_sums, axis=0).fillna(0)
    return df_pivot

def load_dengue_cases(csv_path):
    """Loads dengue cases aggregated by date and UF/Region."""
    print(f"Loading dengue cases from: {csv_path}")
    df = pd.read_csv(csv_path, usecols=['date', 'casos', 'uf'])
    
    # Map UF to Region
    df['Region'] = df['uf'].map(UF_TO_REGION)
    
    # Fill missing regions (e.g. ES is Sudeste, though excluded in forecast)
    df.loc[df['uf'] == 'ES', 'Region'] = 'Sudeste'
    
    # Convert date to month
    df['month'] = df['date'].apply(lambda x: x[:7] if isinstance(x, str) and len(x) >= 7 else None)
    return df

def analyze_cross_correlation(cases_series, serotype_series, max_lag=12):
    """Computes cross-correlation for lags in months. Positive lag means serotype leads cases."""
    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag < 0:
            # cases lead serotype
            c = cases_series.iloc[-lag:].corr(serotype_series.iloc[:lag])
        elif lag > 0:
            # serotype leads cases
            c = cases_series.iloc[:-lag].corr(serotype_series.iloc[lag:])
        else:
            c = cases_series.corr(serotype_series)
        corrs.append((lag, c))
    return pd.DataFrame(corrs, columns=['lag', 'correlation'])

def get_best_lag(cc):
    """Safely extracts the best lag row from cross-correlation results, handling all-NaN values."""
    if cc['correlation'].isna().all() or cc['correlation'].dropna().empty:
        return pd.Series({'lag': 0, 'correlation': 0.0})
    best_idx = cc['correlation'].abs().idxmax()
    return cc.loc[best_idx]

def plot_serotype_cases(time_series, case_col, prop_cols, title, save_path):
    """Generates a stacked area plot of serotypes with case volume line overlay and optional sequence volume sparkline."""
    time_series = time_series.sort_values('month').reset_index(drop=True)
    months = time_series['month']
    x_indices = np.arange(len(months))
    tick_spacing = max(1, len(months) // 12)
    
    has_seq_vol = 'total_seqs' in time_series.columns and time_series['total_seqs'].sum() > 0
    
    if has_seq_vol:
        fig, (ax1, ax_spark) = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True, 
                                            gridspec_kw={'height_ratios': [5, 1.2], 'hspace': 0.25})
    else:
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax_spark = None
        
    # Plot serotype proportions as stacked area
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Classic DENV1-4 colors
    ax1.stackplot(x_indices, 
                  [time_series[c] for c in prop_cols], 
                  labels=['DENV-1', 'DENV-2', 'DENV-3', 'DENV-4'], 
                  colors=colors, alpha=0.6)
    ax1.set_ylabel('Serotype Proportion', fontsize=14)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(0, len(months)-1)
    
    # Plot cases on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(x_indices, time_series[case_col], color='black', linewidth=2.5, label='Monthly Cases')
    ax2.set_ylabel('Dengue Cases', fontsize=14, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.grid(False) # Prevent overlapping grid lines
    
    # Add title and legends
    ax1.set_title(title, fontsize=16, pad=15)
    ax1.set_xticks(x_indices[::tick_spacing])
    ax1.set_xticklabels(months[::tick_spacing], rotation=45, ha='right')
    ax1.set_xlabel('Month', fontsize=14)
    ax1.tick_params(labelbottom=True) # Ensure bottom labels are shown on ax1
    
    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    
    # Plot sequence volume sparkline on ax_spark
    if has_seq_vol:
        ax_spark.plot(x_indices, time_series['total_seqs'], color='#4285F4', linewidth=1.2)
        ax_spark.fill_between(x_indices, time_series['total_seqs'], color='#E8F0FE', alpha=0.8)
        
        # Style ax_spark to look like a neat background bar/sparkline
        ax_spark.set_ylabel('Seqs/Mth', fontsize=10, color='#4285F4')
        ax_spark.tick_params(axis='y', labelsize=8, labelcolor='#4285F4')
        
        # Hide top/right/bottom spines, keep light left spine
        for spine in ['top', 'right', 'bottom']:
            ax_spark.spines[spine].set_visible(False)
        ax_spark.spines['left'].set_color('#4285F4')
        ax_spark.spines['left'].set_linewidth(0.5)
        
        # Hide x-axis labels on sparkline
        ax_spark.tick_params(labelbottom=False, bottom=False)
        ax_spark.set_xlim(0, len(months)-1)
        ax_spark.grid(False)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {save_path}")


def standardize_timeline(df, fill_cols, ffill_cols):
    """Aligns the DataFrame to a standard monthly grid from 2010-01 to 2026-04."""
    start_date = pd.to_datetime('2010-01-01')
    end_date = pd.to_datetime('2026-04-01')
    all_months = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m')
    
    grid = pd.DataFrame({'month': all_months})
    df_std = pd.merge(grid, df, on='month', how='left')
    
    # Fill cases/counts with 0
    for col in fill_cols:
        if col in df_std.columns:
            df_std[col] = df_std[col].fillna(0)
            
    # Forward fill and then backward fill proportions
    for col in ffill_cols:
        if col in df_std.columns:
            df_std[col] = df_std[col].ffill().bfill().fillna(0)
            
    return df_std

def run_analysis():
    # Create directories
    os.makedirs('plots', exist_ok=True)
    
    # Load cases
    df_cases = load_dengue_cases('data/data_imdc_2026/dengue.csv')
    
    # Aggregate National Cases
    print("Aggregating national monthly cases...")
    national_cases = df_cases.groupby('month')['casos'].sum().reset_index(name='cases')
    
    # Load Official National Serotypes
    official_se = parse_official_serotypes('data/serotype_progression/Serotype_Progression.csv')
    
    # Merge National official
    national_merged = pd.merge(national_cases, official_se, on='month', how='outer')
    national_merged = standardize_timeline(
        national_merged, 
        ['cases'], 
        ['DENV1', 'DENV2', 'DENV3', 'DENV4']
    )
    
    # Align and plot National Official
    print("\n--- NATIONAL ANALYSIS (OFFICIAL LAB SURVEILLANCE DATA) ---")
    plot_serotype_cases(
        national_merged, 
        'cases', 
        ['DENV1', 'DENV2', 'DENV3', 'DENV4'], 
        'Brazil: Monthly Dengue Cases vs. Official Serotype Proportions (2010-2026)', 
        'plots/national_official_serotypes.png'
    )
    
    # Calculate cross-correlations for National
    print("\nCross-correlations (Official data):")
    for denv in ['DENV1', 'DENV2', 'DENV3', 'DENV4']:
        cc = analyze_cross_correlation(national_merged['cases'], national_merged[denv], max_lag=12)
        best_row = get_best_lag(cc)
        print(f"  {denv}: Max correlation {best_row['correlation']:.3f} at lag {int(best_row['lag'])} months "
              f"({'serotype leads cases' if best_row['lag'] > 0 else 'cases lead serotype' if best_row['lag'] < 0 else 'coincident'})")
        
    # Load GISAID National data for comparison
    print("\nLoading GISAID national processed data...")
    gisaid_nat = pd.read_csv('data/serotype_progression/processed/gisaid_monthly_national.csv')
    # Rename columns to match DENV1, DENV2, etc.
    gisaid_nat = gisaid_nat.rename(columns={
        'DENV1_prop': 'gisaid_DENV1',
        'DENV2_prop': 'gisaid_DENV2',
        'DENV3_prop': 'gisaid_DENV3',
        'DENV4_prop': 'gisaid_DENV4'
    })
    
    national_gisaid_merged = pd.merge(national_cases, gisaid_nat, on='month', how='outer')
    national_gisaid_merged = standardize_timeline(
        national_gisaid_merged, 
        ['cases', 'total_seqs'], 
        ['gisaid_DENV1', 'gisaid_DENV2', 'gisaid_DENV3', 'gisaid_DENV4']
    )
    plot_serotype_cases(
        national_gisaid_merged, 
        'cases', 
        ['gisaid_DENV1', 'gisaid_DENV2', 'gisaid_DENV3', 'gisaid_DENV4'], 
        'Brazil: Monthly Dengue Cases vs. GISAID Serotype Proportions (2010-2026)', 
        'plots/national_gisaid_serotypes.png'
    )
    
    # Calculate cross-correlations for National GISAID
    print("\nCross-correlations (GISAID National data):")
    for denv in ['DENV1', 'DENV2', 'DENV3', 'DENV4']:
        g_col = f'gisaid_{denv}'
        cc = analyze_cross_correlation(national_gisaid_merged['cases'], national_gisaid_merged[g_col], max_lag=12)
        best_row = get_best_lag(cc)
        print(f"  {denv}: Max correlation {best_row['correlation']:.3f} at lag {int(best_row['lag'])} months")

    # REGIONAL ANALYSIS
    print("\n--- REGIONAL ANALYSIS (GISAID GENOMIC DATA) ---")
    gisaid_reg = pd.read_csv('data/serotype_progression/processed/gisaid_monthly_region.csv')
    
    # Aggregate cases by Region and Month
    regional_cases = df_cases.groupby(['month', 'Region'])['casos'].sum().reset_index(name='cases')
    
    regions = df_cases['Region'].dropna().unique()
    regional_stats = []
    
    for region in regions:
        print(f"\nAnalyzing region: {region}")
        reg_cases = regional_cases[regional_cases['Region'] == region].copy()
        reg_gisaid = gisaid_reg[gisaid_reg['Region'] == region].copy()
        
        reg_gisaid = reg_gisaid.rename(columns={
            'DENV1_prop': 'g_DENV1',
            'DENV2_prop': 'g_DENV2',
            'DENV3_prop': 'g_DENV3',
            'DENV4_prop': 'g_DENV4'
        })
        
        merged = pd.merge(reg_cases, reg_gisaid, on='month', how='outer')
        merged = standardize_timeline(
            merged, 
            ['cases', 'total_seqs'], 
            ['g_DENV1', 'g_DENV2', 'g_DENV3', 'g_DENV4']
        )
        merged['Region'] = region
            
        plot_serotype_cases(
            merged, 
            'cases', 
            ['g_DENV1', 'g_DENV2', 'g_DENV3', 'g_DENV4'], 
            f'Region {region}: Monthly Dengue Cases vs. GISAID Serotypes', 
            f'plots/regional_serotypes_{region}.png'
        )
        
        # Calculate cross-correlations
        for denv in ['DENV1', 'DENV2', 'DENV3', 'DENV4']:
            g_col = f'g_{denv}'
            cc = analyze_cross_correlation(merged['cases'], merged[g_col], max_lag=12)
            best_row = get_best_lag(cc)
            regional_stats.append({
                'Region': region,
                'Serotype': denv,
                'Max_Corr': best_row['correlation'],
                'Optimal_Lag_Months': int(best_row['lag']),
                'Direction': 'leads' if best_row['lag'] > 0 else 'lags' if best_row['lag'] < 0 else 'coincident'
            })
            print(f"  {denv}: Max corr {best_row['correlation']:.3f} at lag {int(best_row['lag'])} months")
            
    df_reg_stats = pd.DataFrame(regional_stats)
    print("\nRegional Correlation Summary:")
    print(df_reg_stats.to_string(index=False))
    df_reg_stats.to_csv('data/serotype_progression/processed/regional_serotype_correlations.csv', index=False)

    # SEROTYPE INVASION PREDICTIVE TEST
    print("\n--- SEROTYPE INVASION INDICATOR TEST ---")
    # Let's define an invasion event: 
    # A serotype proportion in a region is > 10% in month t, but was < 2% on average over the preceding 12 months.
    # We want to check if this invasion precedes an epidemic peak in the next 1-6 months.
    invasion_events = []
    
    for region in regions:
        reg_gisaid = gisaid_reg[gisaid_reg['Region'] == region].copy()
        reg_cases = regional_cases[regional_cases['Region'] == region].copy()
        
        # Merge to align timelines and standardize
        merged = pd.merge(reg_cases, reg_gisaid, on=['month', 'Region'], how='outer')
        merged = standardize_timeline(
            merged,
            ['cases', 'total_seqs', 'DENV1', 'DENV2', 'DENV3', 'DENV4'],
            ['DENV1_prop', 'DENV2_prop', 'DENV3_prop', 'DENV4_prop']
        )
        merged['Region'] = region
        if len(merged) < 24:
            continue
            
        # Define historical threshold for epidemic surge (e.g. cases > 80th percentile of region's historical cases)
        epidemic_threshold = merged['cases'].quantile(0.80)
        
        # Add rolling averages for each serotype proportion to test invasion
        for denv in ['DENV1', 'DENV2', 'DENV3', 'DENV4']:
            prop_col = f'{denv}_prop'
            
            # Identify invasion month
            for i in range(12, len(merged)):
                current_prop = merged.loc[i, prop_col]
                prev_12_props = merged.loc[i-12:i-1, prop_col]
                
                # Check condition
                if current_prop > 0.10 and prev_12_props.mean() < 0.02:
                    month = merged.loc[i, 'month']
                    
                    # Look ahead 1-6 months to see if an epidemic surge occurs
                    look_ahead = merged.loc[i+1:i+6]
                    success = (look_ahead['cases'] > epidemic_threshold).any()
                    peak_cases = look_ahead['cases'].max() if len(look_ahead) > 0 else np.nan
                    
                    invasion_events.append({
                        'Region': region,
                        'Month': month,
                        'Invading_Serotype': denv,
                        'Proportion_At_Invasion': current_prop,
                        'Prior_Mean_Proportion': prev_12_props.mean(),
                        'Followed_By_Surge_6m': success,
                        'Max_Cases_Following_6m': peak_cases,
                        'Regional_Epidemic_Threshold': epidemic_threshold
                    })
                    
    df_invasions = pd.DataFrame(invasion_events)
    print("\nDetected Serotype Invasion Events in GISAID regional data:")
    if len(df_invasions) > 0:
        print(df_invasions.to_string(index=False))
        df_invasions.to_csv('data/serotype_progression/processed/serotype_invasion_events.csv', index=False)
        tpr = df_invasions['Followed_By_Surge_6m'].mean()
        print(f"\nInvasion Predictive Power: {tpr*100:.1f}% of serotype invasion events were followed by an epidemic surge in the subsequent 6 months.")
    else:
        print("No serotype invasion events detected with the current strict threshold.")

if __name__ == '__main__':
    run_analysis()
