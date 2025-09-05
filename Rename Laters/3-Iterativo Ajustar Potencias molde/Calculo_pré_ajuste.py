# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 18:34:23 2025

@author: ugims
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata
import matplotlib.patches as patches

# ==========================
# CONFIGURAÇÃO E LEITURA
# ==========================
file_path = r"C:/Users/ugims/inegi.up.pt/Teses & Estágios - Teses_Estágios - Miguel António Costa - Teses_Estágios - Miguel António Costa/3. Repositório do Miguel/Matriz seleção/prot2_uniforme.csv"

# Dimensões do molde (### ALTERADO)
MOLD_X = 900.0  # mm
MOLD_Z = 420.0  # mm

# Lê o ficheiro, ignorando as primeiras 8 linhas e forçando separador vírgula
df = pd.read_csv(
    file_path,
    skiprows=8,
    sep=',',
    skip_blank_lines=True,
    engine='python',
    encoding='windows-1252'
)

# Limpar nomes das colunas e espaços à volta dos dados
df.columns = [col.strip() for col in df.columns]
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Converter colunas para tipo numérico
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remover linhas inválidas
df.dropna(inplace=True)

# ==========================
# REESCALA PARA 900 x 420 mm (### ALTERADO)
# ==========================
# Considera as colunas originais 'X (mm)' e 'Z (mm)' como coordenadas medidas e
# reescala linearmente para caber exatamente em [0, MOLD_X] e [0, MOLD_Z]
x_raw = df['X (mm)']
z_raw = df['Z (mm)']

x_min_raw, x_max_raw = x_raw.min(), x_raw.max()
z_min_raw, z_max_raw = z_raw.min(), z_raw.max()

# Evitar divisão por zero
if x_max_raw == x_min_raw:
    raise ValueError("Intervalo de X no ficheiro é nulo; não é possível reescalar.")
if z_max_raw == z_min_raw:
    raise ValueError("Intervalo de Z no ficheiro é nulo; não é possível reescalar.")

# Reescala linear
df['X (mm)'] = (x_raw - x_min_raw) / (x_max_raw - x_min_raw) * MOLD_X
df['Z (mm)'] = (z_raw - z_min_raw) / (z_max_raw - z_min_raw) * MOLD_Z

# Atalhos
x = df['X (mm)']
y = df['Z (mm)']  # Largura (eixo Z do molde)
temperatura = df['Value (Celsius)']

# Mostrar amostra
print(df.head())

# (Opcional) Guardar ficheiro limpo
df.to_csv("dados_superficie_organizados.csv", sep=';', index=False)

# ==========================
# GRÁFICO 3D DE DISPERSÃO
# ==========================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, temperatura, c=temperatura, cmap='plasma')

ax.set_xlabel('X (mm) - Comprimento (0–900)')
ax.set_ylabel('Z (mm) - Largura (0–420)')
ax.set_zlabel('Temperatura (°C)')
ax.set_title('Distribuição de Temperatura na Superfície Superior do Molde (900 × 420 mm)')

cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Temperatura (°C)')

plt.tight_layout()
plt.show()

# ==========================
# MAPA 2D INTERPOLADO
# ==========================
xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), temperatura, (xi, yi), method='cubic')

fig2 = plt.figure(figsize=(10, 8))
contour = plt.contourf(xi, yi, zi, levels=100, cmap='plasma')
plt.xlabel('X (mm) - Comprimento (0–900)')
plt.ylabel('Z (mm) - Largura (0–420)')
plt.title('Mapa 2D de Temperatura na Superfície Superior (900 × 420 mm)')
cbar2 = plt.colorbar(contour)
cbar2.set_label('Temperatura (°C)')
plt.tight_layout()
plt.show()

# ==========================
# DIVISÃO EM ZONAS 3 × 2 (média por zona) — Mantido (opera na escala 900 × 420)
# ==========================
x_min, x_max = 0.0, MOLD_X
z_min, z_max = 0.0, MOLD_Z

x_div = np.linspace(x_min, x_max, 4)  # 3 zonas em X → 4 limites
z_div = np.linspace(z_min, z_max, 3)  # 2 zonas em Z → 3 limites

def identificar_zona(xp, zp):
    for i in range(3):  # X
        for j in range(2):  # Z
            if x_div[i] <= xp < x_div[i+1] and z_div[j] <= zp < z_div[j+1]:
                return j * 3 + i + 1
    # incluir ponto no limite superior extremo
    if np.isclose(xp, x_max) and np.isclose(zp, z_max):
        return 6
    return np.nan

df['Zona'] = df.apply(lambda row: identificar_zona(row['X (mm)'], row['Z (mm)']), axis=1)

medias_por_zona = df.groupby('Zona')['Value (Celsius)'].mean().reset_index()
print("\nTemperatura média por zona (°C):")
for _, row in medias_por_zona.iterrows():
    print(f"Zona {int(row['Zona'])}: {row['Value (Celsius)']:.2f} °C")

zona_temp_dict = dict(zip(medias_por_zona['Zona'], medias_por_zona['Value (Celsius)']))

# Mapa 3×2 pintado por médias
fig, ax = plt.subplots(figsize=(10, 8))
for i in range(3):  # X
    for j in range(2):  # Z
        zona_id = j * 3 + i + 1
        x_start = x_div[i]
        z_start = z_div[j]
        largura = x_div[i+1] - x_div[i]
        altura = z_div[j+1] - z_div[j]
        media_temp = zona_temp_dict.get(zona_id, np.nan)

        cor = plt.cm.plasma(
            (media_temp - medias_por_zona['Value (Celsius)'].min()) /
            (medias_por_zona['Value (Celsius)'].max() - medias_por_zona['Value (Celsius)'].min())
        )
        rect = patches.Rectangle((x_start, z_start), largura, altura, facecolor=cor, edgecolor='white', linewidth=1)
        ax.add_patch(rect)

        ax.text(
            x_start + largura / 2, z_start + altura / 2,
            f"{media_temp:.1f}°C\nZona {zona_id}",
            color='white', ha='center', va='center', fontsize=7, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', edgecolor='white', linewidth=0.8, alpha=0.6)
        )

ax.set_xlim(x_min, x_max)
ax.set_ylim(z_min, z_max)
ax.set_xlabel('X (mm) - Comprimento (0–900)')
ax.set_ylabel('Z (mm) - Largura (0–420)')
ax.set_title('Zonas de Aquecimento 3×2 com Temperatura Média (Cores Uniformes)')
sm = plt.cm.ScalarMappable(
    cmap='plasma',
    norm=plt.Normalize(vmin=medias_por_zona['Value (Celsius)'].min(),
                       vmax=medias_por_zona['Value (Celsius)'].max())
)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Temperatura Média (°C)')
plt.grid(False)
plt.tight_layout()
plt.show()

# ==========================
# SUBDIVISÃO 15 × 4 (60 subzonas) — alinhada às 15 zonas longitudinais reais
# ==========================
import matplotlib.patches as patches

# Centros reais (mm) das 15 resistências (do teu desenho)
centros = [
    29.60, 88.80, 148.00, 207.20, 266.40,
    331.60, 390.80, 450.00, 509.20, 568.40,
    633.60, 692.80, 752.00, 811.20, 870.40
]
centros = sorted(centros)

# Limites irregulares das 15 zonas por pontos médios entre centros
limites = [0.0]
for i in range(1, len(centros)):
    limites.append(0.5 * (centros[i-1] + centros[i]))
limites.append(900.0)
x_edges = np.array(limites, dtype=float)   # 16 bordas → 15 colunas
z_edges = np.linspace(z_min, z_max, 5)     # 5 bordas → 4 linhas

ncols = len(x_edges) - 1  # 15
nrows = len(z_edges) - 1  # 4

def identificar_zona_15x4(xp, zp, xed, zed):
    # coluna (X)
    i = None
    for ii in range(len(xed) - 1):
        if xed[ii] <= xp < xed[ii + 1]:
            i = ii
            break
    if i is None and np.isclose(xp, xed[-1]):
        i = len(xed) - 2  # incluir extremo

    # linha (Z)
    j = None
    for jj in range(len(zed) - 1):
        if zed[jj] <= zp < zed[jj + 1]:
            j = jj
            break
    if j is None and np.isclose(zp, zed[-1]):
        j = len(zed) - 2  # incluir extremo

    if i is None or j is None:
        return np.nan
    return j * (len(xed) - 1) + i + 1  # numeração linha a linha

# Atribuir subzona 15×4 a cada ponto
df['Zona_15x4'] = df.apply(lambda r: identificar_zona_15x4(r['X (mm)'], r['Z (mm)'], x_edges, z_edges), axis=1)

# Estatísticas por subzona
estat_15x4 = df.groupby('Zona_15x4')['Value (Celsius)'].agg(['min', 'max', 'mean']).reset_index()
estat_15x4['Delta (max - min)'] = estat_15x4['max'] - estat_15x4['min']
estat_dict_15x4 = estat_15x4.set_index('Zona_15x4').to_dict(orient='index')

# GRÁFICO das 60 subzonas
fig, ax = plt.subplots(figsize=(16, 9))

for i in range(ncols):         # 15 colunas
    for j in range(nrows):     # 4 linhas
        zona_id = j * ncols + i + 1
        x_start = x_edges[i]
        z_start = z_edges[j]
        largura = x_edges[i + 1] - x_edges[i]
        altura  = z_edges[j + 1] - z_edges[j]

        stats = estat_dict_15x4.get(zona_id)
        if stats:
            media_temp = stats['mean']
            delta_temp = stats['Delta (max - min)']
            cor = plt.cm.plasma(
                (media_temp - estat_15x4['mean'].min()) /
                (estat_15x4['mean'].max() - estat_15x4['mean'].min() + 1e-12)
            )
        else:
            media_temp = np.nan
            delta_temp = np.nan
            cor = 'gray'

        rect = patches.Rectangle((x_start, z_start), largura, altura,
                                 facecolor=cor, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)

        ax.text(
            x_start + largura / 2, z_start + altura / 2,
            f"{media_temp:.1f}°C\nΔT={delta_temp:.1f}°C\nZona {zona_id}",
            color='white', ha='center', va='center', fontsize=6.2, weight='bold',
            bbox=dict(boxstyle='round,pad=0.28', facecolor='black', edgecolor='white', linewidth=0.8, alpha=0.6)
        )

# Configuração final
ax.set_xlim(x_min, x_max)
ax.set_ylim(z_min, z_max)
ax.set_xlabel('X (mm) - Comprimento (0–900)')
ax.set_ylabel('Z (mm) - Largura (0–420)')
ax.set_title('Subdivisão 15×4: Temperatura Média e ΔT por Subzona (60 zonas)')

sm = plt.cm.ScalarMappable(
    cmap='plasma',
    norm=plt.Normalize(vmin=estat_15x4['mean'].min(), vmax=estat_15x4['mean'].max())
)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Temperatura Média (°C)')

plt.grid(False)
plt.tight_layout()

# (Opcional) retângulo de contorno entre duas zonas (exemplo)
zona_inf_esq = 12
zona_sup_dir = 25

def get_indices(zona_id, nx):
    i = (zona_id - 1) % nx
    j = (zona_id - 1) // nx
    return i, j

i_min, j_min = get_indices(zona_inf_esq, ncols)
i_max, j_max = get_indices(zona_sup_dir, ncols)

x_start = x_edges[i_min]
z_start = z_edges[j_min]
largura = x_edges[i_max + 1] - x_start
altura  = z_edges[j_max + 1] - z_start

rect_contorno = patches.Rectangle(
    (x_start, z_start), largura, altura,
    linewidth=2.5, edgecolor='blue', facecolor='none', linestyle='solid'
)
ax.add_patch(rect_contorno)

plt.grid(False)
plt.tight_layout()
plt.show()

# ==========================
# AJUSTE DE POTÊNCIA — 15 ZONAS PARA A TEMPERATURA ALVO
# ==========================
K = 1.5         # Ganho de correção térmica (ajusta a agressividade do ajuste)
P_MIN = 10      # Potência mínima permitida (W)
P_MAX = 150     # Potência máxima permitida (W)

# --- Centros das resistências em mm (do desenho) ---
centros = [
    29.60, 88.80, 148.00, 207.20, 266.40,
    331.60, 390.80, 450.00, 509.20, 568.40,
    633.60, 692.80, 752.00, 811.20, 870.40
]
centros = sorted(centros)

# --- Limites irregulares das 15 zonas por pontos médios entre centros ---
limites = [0.0]
for i in range(1, len(centros)):
    limites.append(0.5 * (centros[i-1] + centros[i]))
limites.append(900.0)
limites = np.array(limites)  # 16 bordas → 15 zonas

# Função para mapear X -> zona [1..15] usando limites irregulares
def identificar_zona_x_custom(xp, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= xp < edges[i + 1]:
            return i + 1
    if np.isclose(xp, edges[-1]):
        return len(edges) - 1
    return np.nan

# Atribuir zona a cada ponto e calcular T_média por zona
df['Zona_X'] = df['X (mm)'].apply(lambda v: identificar_zona_x_custom(v, limites))
medias_por_zona_x = df.groupby('Zona_X')['Value (Celsius)'].mean().reset_index()
zona_temp = medias_por_zona_x.copy().rename(columns={'Value (Celsius)': 'T_média'})
zona_temp = zona_temp.sort_values('Zona_X').reset_index(drop=True)

# --- INPUT: Temperatura alvo ---
while True:
    try:
        T_SET = float(input("\nInsere a TEMPERATURA DE SUPERFÍCIE ALVO (°C): "))
        break
    except ValueError:
        print("Valor inválido. Insere um número (°C).")

# --- INPUT: Potências apenas das zonas 1..8 (o resto é simétrico) ---
print("\nInsere a potência aplicada em cada zona (1 a 8) — o outro lado é simétrico em torno da zona 8:")
potencias_parciais = {}
for i in range(1, 9):  # 1..8
    while True:
        try:
            p = float(input(f"  Zona {i}: "))
            potencias_parciais[i] = p
            break
        except ValueError:
            print("Valor inválido. Insere um número válido.")

# Reconstruir vetor completo 1..15 (centro = zona 8)
potencias_aplicadas = [
    potencias_parciais[1],  # 1
    potencias_parciais[2],  # 2
    potencias_parciais[3],  # 3
    potencias_parciais[4],  # 4
    potencias_parciais[5],  # 5
    potencias_parciais[6],  # 6
    potencias_parciais[7],  # 7
    potencias_parciais[8],  # 8 (centro)
    potencias_parciais[7],  # 9
    potencias_parciais[6],  # 10
    potencias_parciais[5],  # 11
    potencias_parciais[4],  # 12
    potencias_parciais[3],  # 13
    potencias_parciais[2],  # 14
    potencias_parciais[1],  # 15
]
zona_temp['Potência Aplicada (W)'] = potencias_aplicadas

# --- AJUSTE PARA O ALVO ---
# Erro por zona versus alvo (positivo = está acima do alvo)
denom = max(T_SET, 1e-6)  # evitar divisão por zero
zona_temp['Erro vs alvo (°C)'] = zona_temp['T_média'] - T_SET

# Fator de ajuste proporcional (se T_média > alvo → reduz potência; se < alvo → aumenta)
zona_temp['Fator ajuste'] = 1 - K * (zona_temp['Erro vs alvo (°C)'] / denom)

# Nova potência (com limites)
zona_temp['Nova Potência (W)'] = (zona_temp['Potência Aplicada (W)'] * zona_temp['Fator ajuste']).clip(P_MIN, P_MAX)

# --- RESULTADOS ---
print("\n=== Ajuste por Zona para Alvo = {:.2f} °C ===".format(T_SET))
for _, row in zona_temp.iterrows():
    print(f"Zona {int(row['Zona_X']):2d}: Tmédia = {row['T_média']:.2f} °C | "
          f"Erro = {row['Erro vs alvo (°C)']:+.2f} °C | "
          f"Fator = {row['Fator ajuste']:.3f} | "
          f"P_old = {row['Potência Aplicada (W)']:.1f} W → P_new = {row['Nova Potência (W)']:.1f} W")

# --- GRÁFICO: comparação Potência Aplicada vs Nova Potência + Erro vs Alvo ---
fig, ax1 = plt.subplots(figsize=(12, 5.8))
ax1.bar(zona_temp['Zona_X'], zona_temp['Nova Potência (W)'], label='Nova Potência', alpha=0.85)
ax1.plot(zona_temp['Zona_X'], zona_temp['Potência Aplicada (W)'],
         linestyle='--', marker='o', label='Potência Aplicada')

ax1.set_xlabel('Zona Longitudinal (X) — 15 zonas alinhadas aos centros reais')
ax1.set_ylabel('Potência (W)')
ax1.set_title(f'Ajuste de Potência por Zona — Alvo = {T_SET:.1f} °C (K = {K})')
ax1.legend()
ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
ax1.set_xticks(range(1, 16))

# Eixo secundário: erro vs alvo
ax2 = ax1.twinx()
ax2.plot(zona_temp['Zona_X'], zona_temp['Erro vs alvo (°C)'],
         marker='s', linestyle='-', label='Erro vs alvo (°C)')
ax2.set_ylabel('Erro vs alvo (°C)')
ax2.tick_params(axis='y')

plt.tight_layout()
plt.show()

# --- OVERLAY: desenhar os 15 limites no mapa 2D (para conferência visual) ---
fig = plt.figure(figsize=(10, 8))
contour = plt.contourf(xi, yi, zi, levels=100, cmap='plasma')
for e in limites:
    plt.axvline(e, linewidth=0.9, alpha=0.75)  # linhas dos limites de zona
# opcional: marcar os centros
for c in centros:
    plt.axvline(c, linewidth=0.8, linestyle=':', alpha=0.6)
plt.xlabel('X (mm) - Comprimento (0–900)')
plt.ylabel('Z (mm) - Largura (0–420)')
plt.title('Mapa 2D com limites das 15 zonas (linhas) e centros (tracejado)')
cbar2 = plt.colorbar(contour)
cbar2.set_label('Temperatura (°C)')
plt.tight_layout()
plt.show()
