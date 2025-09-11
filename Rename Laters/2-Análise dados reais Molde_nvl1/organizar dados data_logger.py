# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 16:44:54 2025

@author: ugims
"""


import pandas as pd

# Original file Path CSV
ficheiro_csv = r'C:/Users/ugims/inegi.up.pt/Teses & Estágios - Teses_Estágios - Miguel António Costa - Teses_Estágios - Miguel António Costa/3. Repositório do Miguel/Miguel_MoldeQuente/Session_6M2.csv'  # certifica-te de que está no mesmo diretório do script ou ajusta o caminho

# Read file, line by line
with open(ficheiro_csv, 'r', encoding='utf-8') as f:
    linhas = [linha.strip().split(',') for linha in f.readlines()]

# Find collumns (for alignment)
max_colunas = max(len(linha) for linha in linhas)

# Fill empty cells
linhas_alinhadas = [linha + [''] * (max_colunas - len(linha)) for linha in linhas]

# Convert to Dataframe
df_bruto = pd.DataFrame(linhas_alinhadas)

# Save to Excel
df_bruto.to_excel("Dados_MIGUEL_Bruto_3.xlsx", index=False, header=False)

print("File 'Dados_MIGUEL_Bruto.xlsx' created with all original data.")

