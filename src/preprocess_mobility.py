import os
import pandas as pd
import numpy as np

# List of 26 states (excluding ES)
STATES = sorted([
    'AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 
    'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO'
])

def preprocess_mobility():
    print("Preprocessing human mobility networks for Brazil...")
    
    # 1. ANAC Flights
    print("Loading ANAC flights origin-destination flows...")
    df_flights = pd.read_excel('data/processed/LIG_AEREAS_2019-2020_fluxos_od.xlsx', sheet_name='BASE_FLUXO_OD')
    # Filter for our 26 states
    df_flights = df_flights[df_flights['UF_O'].isin(STATES) & df_flights['UF_D'].isin(STATES)].copy()
    
    # Aggregate passenger flow (VAR01 represents 2019 passengers)
    flight_flow = df_flights.groupby(['UF_D', 'UF_O'])['VAR01'].sum().unstack(fill_value=0.0)
    # Reindex to ensure 26x26 shape
    flight_flow = flight_flow.reindex(index=STATES, columns=STATES, fill_value=0.0)
    
    # 2. IBGE REGIC Urban Connections
    print("Loading REGIC inter-city connectivity network...")
    df_regic = pd.read_excel('data/processed/REGIC2018_Ligacoes_entre_Cidades.xlsx', sheet_name='REGIC2018_Ligacoes_entre_Cidade')
    # Filter for our 26 states
    df_regic = df_regic[df_regic['uf_ori'].isin(STATES) & df_regic['uf_dest'].isin(STATES)].copy()
    
    # Aggregate link counts
    regic_flow = df_regic.groupby(['uf_dest', 'uf_ori']).size().unstack(fill_value=0.0)
    # Reindex to ensure 26x26 shape
    regic_flow = regic_flow.reindex(index=STATES, columns=STATES, fill_value=0.0)
    
    # 3. Set diagonal to zero (exclude self-mobility for importation risk)
    np.fill_diagonal(flight_flow.values, 0.0)
    np.fill_diagonal(regic_flow.values, 0.0)
    
    # 4. Normalize matrices to put them on the same scale (total sum = 1.0)
    flight_sum = flight_flow.values.sum()
    regic_sum = regic_flow.values.sum()
    
    flight_norm = flight_flow / (flight_sum if flight_sum > 0 else 1.0)
    regic_norm = regic_flow / (regic_sum if regic_sum > 0 else 1.0)
    
    # Combine (50% flights, 50% regional connectivity)
    combined_flow = 0.5 * flight_norm + 0.5 * regic_norm
    
    # Row-normalize: inflow weights to any state (destination row) sum to 1.0
    row_sums = combined_flow.sum(axis=1)
    combined_norm = combined_flow.div(row_sums, axis=0).fillna(0.0)
    
    # Save the normalized 26x26 matrix
    output_path = 'data/processed/state_mobility_matrix.csv'
    combined_norm.to_csv(output_path)
    print(f"Saved normalized state-to-state mobility matrix to {output_path} (shape: {combined_norm.shape})")
    
if __name__ == '__main__':
    preprocess_mobility()
