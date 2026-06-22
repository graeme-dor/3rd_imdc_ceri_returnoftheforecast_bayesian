import os
import gzip
import pandas as pd
import numpy as np

def preprocess():
    print("Starting preprocessing...")
    os.makedirs('data/processed', exist_ok=True)
    
    # 1. Load Geocode mapping
    print("Loading geocode mappings...")
    df_map = pd.read_csv('data/data_imdc_2026/map_regional_health.csv')
    # Keep geocode to state mappings
    geocode_to_state = df_map.set_index('geocode')[['uf', 'uf_code', 'uf_name']].to_dict('index')
    
    # 2. Aggregate cases (dengue) to state level
    print("Aggregating dengue cases to state level...")
    # Read dengue cases in chunks to save memory or fully if possible
    df_dengue = pd.read_csv('data/data_imdc_2026/dengue.csv')
    
    # Exclude ES (Espírito Santo)
    df_dengue = df_dengue[df_dengue['uf'] != 'ES'].copy()
    
    # Group by state and date, sum cases, and get the train/target flags
    agg_dict = {
        'casos': 'sum',
        'uf_code': 'first',
        'train_1': 'first',
        'target_1': 'first',
        'train_2': 'first',
        'target_2': 'first',
        'train_3': 'first',
        'target_3': 'first',
        'train_4': 'first',
        'target_4': 'first'
    }
    
    df_state_cases = df_dengue.groupby(['uf', 'date']).agg(agg_dict).reset_index()
    print(f"Aggregated cases shape: {df_state_cases.shape}")
    
    # 3. Load Population data and aggregate to state level
    print("Aggregating population to state level...")
    df_pop = pd.read_csv('data/data_imdc_2026/datasus_population_2001_2025.csv.gz')
    # Map geocode to state
    df_pop['uf'] = df_pop['geocode'].map(lambda x: geocode_to_state.get(x, {}).get('uf'))
    # Filter out missing states or ES
    df_pop = df_pop[(df_pop['uf'].notna()) & (df_pop['uf'] != 'ES')].copy()
    
    # Sum population by state and year
    df_state_pop = df_pop.groupby(['uf', 'year'])['population'].sum().reset_index()
    print(f"Aggregated population shape: {df_state_pop.shape}")
    
    # 4. Load climate reanalysis data and compute population-weighted averages
    print("Aggregating climate variables (population-weighted) to state level...")
    # To do this efficiently:
    # First, let's load the population dataset as a lookup dict: (geocode, year) -> population
    pop_lookup = df_pop.set_index(['geocode', 'year'])['population'].to_dict()
    
    # Read climate file in chunks to avoid memory issues
    climate_chunks = []
    chunk_size = 500000
    
    # Variables we want to aggregate:
    climate_vars = ['temp_min', 'temp_med', 'temp_max', 'precip_min', 'precip_med', 'precip_max', 'rel_humid_min', 'rel_humid_med', 'rel_humid_max', 'thermal_range', 'rainy_days']
    
    # Helper to parse year from date string 'YYYY-MM-DD'
    def get_year(date_str):
        try:
            return int(date_str[:4])
        except:
            return 2010

    print("Processing climate file chunks...")
    for chunk in pd.read_csv('data/data_imdc_2026/climate.csv.gz', chunksize=chunk_size):
        # Map geocode to state
        chunk['uf'] = chunk['geocode'].map(lambda x: geocode_to_state.get(x, {}).get('uf'))
        # Exclude ES and missing states
        chunk = chunk[(chunk['uf'].notna()) & (chunk['uf'] != 'ES')].copy()
        if len(chunk) == 0:
            continue
            
        # Get year for population matching (if year > 2025, clip to 2025)
        chunk['year'] = chunk['date'].apply(get_year).clip(upper=2025)
        
        # Get population for each row
        chunk['pop'] = chunk.set_index(['geocode', 'year']).index.map(pop_lookup.get)
        chunk['pop'] = chunk['pop'].fillna(1.0) # Fallback to 1 if missing
        
        # Multiply climate variables by population for weighted average
        for var in climate_vars:
            chunk[f'{var}_weighted'] = chunk[var] * chunk['pop']
            
        # Keep only required columns for grouping
        keep_cols = ['uf', 'date', 'pop'] + [f'{var}_weighted' for var in climate_vars]
        climate_chunks.append(chunk[keep_cols])
        
    print("Concatenating and grouping climate chunks...")
    df_climate_all = pd.concat(climate_chunks, ignore_index=True)
    
    # Group by state and date, sum weighted variables and population
    df_state_cli = df_climate_all.groupby(['uf', 'date']).sum().reset_index()
    
    # Divide by total population to get the weighted average
    for var in climate_vars:
        df_state_cli[var] = df_state_cli[f'{var}_weighted'] / df_state_cli['pop']
        df_state_cli = df_state_cli.drop(columns=[f'{var}_weighted'])
        
    df_state_cli = df_state_cli.drop(columns=['pop'])
    print(f"Aggregated climate shape: {df_state_cli.shape}")
    
    # 5. Load Ocean oscillations
    print("Loading ocean oscillations...")
    df_ocean = pd.read_csv('data/data_imdc_2026/ocean_climate_oscillations.csv.gz')
    # Ocean date starts on Monday, let's map it to the Sunday of that week or use a merge on date/epiweek.
    # Wait, the climate/dengue files have dates that are Sundays (e.g. 2010-01-03).
    # Let's map ocean date to the Sunday of that epiweek.
    # To do this safely, we can map date to epiweek, and then merge.
    # Let's check if ocean has epiweek. No, columns are: 'date', 'enso', 'iod', 'pdo'.
    # We can write a helper to map date to epiweek.
    def date_to_epiweek(date_str):
        # We can convert to pandas datetime and get the epiweek
        dt = pd.to_datetime(date_str)
        # Week number of the year (approx)
        # A simpler way is to find the closest Sunday or merge using a date offset
        return dt
    
    # Let's align ocean date (Monday) to Sunday (which is 6 days later, or 1 day earlier depending on definition).
    # Standard epiweek in dengue starts on Sunday. Let's find the Sunday of that week.
    df_ocean['date_dt'] = pd.to_datetime(df_ocean['date'])
    # Monday is weekday 0. Sunday of the same week is weekday 6 (which is date_dt - 1 day, or start of week).
    # In IBGE, Sunday is the first day of the week. So the Sunday of the week containing a Monday is Monday - 1 day.
    df_ocean['date_sunday'] = (df_ocean['date_dt'] - pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
    df_ocean = df_ocean.drop(columns=['date', 'date_dt']).rename(columns={'date_sunday': 'date'})
    print(f"Ocean oscillations shape: {df_ocean.shape}")
    
    # 6. Merge all datasets
    print("Merging datasets...")
    # Base is state cases
    df_merged = pd.merge(df_state_cases, df_state_cli, on=['uf', 'date'], how='left')
    
    # Merge population (need to extract year from date)
    df_merged['year'] = pd.to_datetime(df_merged['date']).dt.year
    df_merged = pd.merge(df_merged, df_state_pop, on=['uf', 'year'], how='left')
    df_merged = df_merged.drop(columns=['year'])
    
    # Merge ocean oscillations
    df_merged = pd.merge(df_merged, df_ocean, on='date', how='left')
    
    # Forward fill any missing climate/ocean values
    fill_cols = climate_vars + ['enso', 'iod', 'pdo', 'population']
    df_merged[fill_cols] = df_merged.groupby('uf')[fill_cols].ffill().bfill()
    
    # Sort
    df_merged = df_merged.sort_values(['uf', 'date']).reset_index(drop=True)
    
    # 7. Add human mobility importation risk covariate
    print("Calculating human mobility importation risk...")
    import sys
    if 'src' not in sys.path:
        sys.path.append('src')
    from preprocess_mobility import preprocess_mobility
    preprocess_mobility()
    
    # Load mobility matrix
    df_mob = pd.read_csv('data/processed/state_mobility_matrix.csv', index_col=0)
    states = sorted(df_merged['uf'].unique())
    df_mob = df_mob.reindex(index=states, columns=states).fillna(0.0)
    
    # Compute incidence (cases / population * 100,000)
    df_merged['inc'] = (df_merged['casos'] / df_merged['population']) * 100000.0
    
    # Pivot incidence
    df_inc = df_merged.pivot(index='date', columns='uf', values='inc').sort_index()
    
    # Compute risk: I_t @ W^T
    W = df_mob.values
    df_risk = pd.DataFrame(df_inc.values @ W.T, index=df_inc.index, columns=df_inc.columns)
    
    # Melt back
    df_risk_melted = df_risk.reset_index().melt(id_vars='date', value_name='mobility_import_risk', var_name='uf')
    
    # Merge back to df_merged
    df_merged = pd.merge(df_merged, df_risk_melted, on=['uf', 'date'], how='left')
    df_merged = df_merged.drop(columns=['inc'])
    
    # Save the aggregated dataset
    output_path = 'data/processed/state_weekly_features.csv'
    df_merged.to_csv(output_path, index=False)
    print(f"Saved aggregated state features to {output_path} (shape: {df_merged.shape})")
    
    # Quick check of null values
    print("Null values in merged dataset:")
    print(df_merged.isnull().sum())

if __name__ == '__main__':
    preprocess()
