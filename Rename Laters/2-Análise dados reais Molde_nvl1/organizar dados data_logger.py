# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 16:44:54 2025

@author: ugims
"""


import pandas as pd

# Caminho para o ficheiro original CSV
ficheiro_csv = r'C:/Users/ugims/inegi.up.pt/Teses & Estágios - Teses_Estágios - Miguel António Costa - Teses_Estágios - Miguel António Costa/3. Repositório do Miguel/Miguel_MoldeQuente/Session_6M2.csv'  # certifica-te de que está no mesmo diretório do script ou ajusta o caminho

# Lê o conteúdo todo do ficheiro, linha a linha, como texto
with open(ficheiro_csv, 'r', encoding='utf-8') as f:
    linhas = [linha.strip().split(',') for linha in f.readlines()]

# Determina o número máximo de colunas encontradas (para alinhar tudo)
max_colunas = max(len(linha) for linha in linhas)

# Preenche com células vazias onde faltar informação
linhas_alinhadas = [linha + [''] * (max_colunas - len(linha)) for linha in linhas]

# Converte para DataFrame
df_bruto = pd.DataFrame(linhas_alinhadas)

# Guarda no Excel exatamente como estava
df_bruto.to_excel("Dados_MIGUEL_Bruto_3.xlsx", index=False, header=False)

print("Ficheiro 'Dados_MIGUEL_Bruto.xlsx' criado com todos os dados originais.")

