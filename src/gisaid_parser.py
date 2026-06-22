import os
import pandas as pd
import datetime

# Define standard state and region mappings
STATE_TO_UF = {
    'Acre': 'AC',
    'Alagoas': 'AL',
    'Amapa': 'AP',
    'Amazonas': 'AM',
    'Bahia': 'BA',
    'Ceara': 'CE',
    'Distrito Federal': 'DF',
    'Espirito Santo': 'ES',
    'Federal District': 'DF',
    'Goias': 'GO',
    'Maranhao': 'MA',
    'Mato Grosso': 'MT',
    'Mato Grosso do Sul': 'MS',
    'Minas Gerais': 'MG',
    'Para': 'PA',
    'Paraiba': 'PB',
    'Parana': 'PR',
    'Pernambuco': 'PE',
    'Piaui': 'PI',
    'Rio Grande do Norte': 'RN',
    'Rio Grande do Sul': 'RS',
    'Rio de Janeiro': 'RJ',
    'Rondonia': 'RO',
    'Roraima': 'RR',
    'Santa Catarina': 'SC',
    'Sao Paulo': 'SP',
    'Sergipe': 'SE',
    'Tocantins': 'TO'
}

UF_TO_REGION = {
    'AC': 'Centro-Oeste' if False else 'Norte',  # Acre is Norte
    'AM': 'Norte', 'AP': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MS': 'Centro-Oeste', 'MT': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
}

def get_epiweek_sunday(date_str):
    """Maps YYYY-MM-DD to the Sunday of that epidemiological week."""
    try:
        if not isinstance(date_str, str) or len(date_str) != 10:
            return None
        dt = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
        offset = (dt.weekday() + 1) % 7
        sunday = dt - datetime.timedelta(days=offset)
        return sunday.strftime('%Y-%m-%d')
    except Exception:
        return None

def parse_gisaid_data(tsv_path, output_dir):
    print("Reading GISAID data...")
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Process location
    print("Parsing locations and states...")
    def extract_state_uf_region(location_str):
        if not isinstance(location_str, str):
            return None, None, None
        parts = [p.strip() for p in location_str.split('/')]
        if len(parts) >= 3 and parts[0] == 'South America' and parts[1] == 'Brazil':
            state = parts[2]
            uf = STATE_TO_UF.get(state)
            if uf:
                region = UF_TO_REGION.get(uf)
                return state, uf, region
        return None, None, None

    loc_parsed = df['Location'].apply(extract_state_uf_region)
    df['State'] = [x[0] for x in loc_parsed]
    df['UF'] = [x[1] for x in loc_parsed]
    df['Region'] = [x[2] for x in loc_parsed]
    
    # Filter rows with valid UF
    df = df[df['UF'].notna()].copy()
    
    # Process dates
    print("Parsing dates...")
    df['date'] = df['Collection date'].apply(get_epiweek_sunday)
    # Extract Month YYYY-MM
    df['month'] = df['Collection date'].apply(lambda x: x[:7] if isinstance(x, str) and len(x) >= 7 else None)
    
    # Filter rows with valid dates
    df = df[df['date'].notna()].copy()
    
    # Clean Serotype names
    df['Serotype'] = df['Serotype'].str.strip().str.upper()
    df = df[df['Serotype'].isin(['DENV1', 'DENV2', 'DENV3', 'DENV4'])].copy()
    
    print(f"Total valid DENV sequences parsed: {len(df)}")
    
    # Aggregate and Pivot function
    def aggregate_data(df_data, time_col, geo_col):
        # Count serotypes
        counts = df_data.groupby([time_col, geo_col, 'Serotype']).size().reset_index(name='count')
        
        # Pivot serotype counts
        pivoted = counts.pivot(index=[time_col, geo_col], columns='Serotype', values='count').fillna(0).reset_index()
        
        # Ensure all DENV serotype columns exist
        for serotype in ['DENV1', 'DENV2', 'DENV3', 'DENV4']:
            if serotype not in pivoted.columns:
                pivoted[serotype] = 0.0
                
        # Total counts
        pivoted['total_seqs'] = pivoted[['DENV1', 'DENV2', 'DENV3', 'DENV4']].sum(axis=1)
        
        # Compute proportions
        for serotype in ['DENV1', 'DENV2', 'DENV3', 'DENV4']:
            pivoted[f'{serotype}_prop'] = pivoted[serotype] / pivoted['total_seqs']
            
        return pivoted.sort_values([geo_col, time_col])

    # Perform aggregations
    print("Aggregating monthly state level...")
    monthly_state = aggregate_data(df, 'month', 'UF')
    
    print("Aggregating monthly regional level...")
    monthly_region = aggregate_data(df, 'month', 'Region')
    
    print("Aggregating weekly state level...")
    weekly_state = aggregate_data(df, 'date', 'UF')
    
    print("Aggregating weekly regional level...")
    weekly_region = aggregate_data(df, 'date', 'Region')
    
    # National aggregations
    print("Aggregating national level...")
    df['National'] = 'Brazil'
    monthly_national = aggregate_data(df, 'month', 'National')
    weekly_national = aggregate_data(df, 'date', 'National')
    
    # Save CSVs
    os.makedirs(output_dir, exist_ok=True)
    monthly_state.to_csv(os.path.join(output_dir, 'gisaid_monthly_state.csv'), index=False)
    monthly_region.to_csv(os.path.join(output_dir, 'gisaid_monthly_region.csv'), index=False)
    weekly_state.to_csv(os.path.join(output_dir, 'gisaid_weekly_state.csv'), index=False)
    weekly_region.to_csv(os.path.join(output_dir, 'gisaid_weekly_region.csv'), index=False)
    monthly_national.to_csv(os.path.join(output_dir, 'gisaid_monthly_national.csv'), index=False)
    weekly_national.to_csv(os.path.join(output_dir, 'gisaid_weekly_national.csv'), index=False)
    
    print("GISAID parsing completed successfully and files saved!")

if __name__ == '__main__':
    tsv_path = 'data/serotype_progression/gisaid_arbo_2026_05_06_15.tsv'
    output_dir = 'data/serotype_progression/processed'
    parse_gisaid_data(tsv_path, output_dir)
