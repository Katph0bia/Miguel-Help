# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 12:52:03 2025

@author: ugims
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = r'C:/Users/ugims/inegi.up.pt/Teses & Estágios - Teses_Estágios - Miguel António Costa - Teses_Estágios - Miguel António Costa/3. Repositório do Miguel/Miguel_MoldeQuente/Dados_MIGUEL_Bruto_2.xlsx'
sheet_name = "Sheet1"

# Skip metadata rows and read actual data
df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=27)

# Convert relevant columns to numeric (handle non-numeric data gracefully)
channels = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ambient C']
df[channels] = df[channels].apply(pd.to_numeric, errors='coerce')

# Plot all temperature channels
plt.figure(figsize=(14, 7))

for col in channels:
    plt.plot(df['Record'], df[col], label=col)

# Graph formatting
plt.title("Temperature Evolution of All Channels Over Time", fontsize=14)
plt.xlabel("Record Number", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.grid(True)
plt.legend(title="Channels", loc='upper right')
plt.tight_layout()

# Save and show the graph
plt.savefig("all_channels_temperature_graph.png", dpi=300)
plt.show()