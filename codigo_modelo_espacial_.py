# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:47:32 2026

@author: USUARIO
"""

# -*- coding: utf-8 -*-
"""
codigo_modelo_espacial.py
=========================
Script de econometría espacial para el TFG.

Contenido:
  0. Instalación de librerías (ejecutar una vez)
  1. Construcción de la matriz W (contigüidad reina + distancia inversa)
  2. Test de Moran's I global por año  →  H1
  3. Moran's I local (LISA)            →  clusters regionales
  4. Modelo SAR panel                  →  H1 + H2
  5. Modelo SDM panel                  →  robustez
  6. Comparación FE vs SAR vs SDM      →  H2
  7. Efectos directos, indirectos y totales

Librerías necesarias:
  pip install libpysal esda spreg geopandas

Autor: Nuria Adame Fuentes — TFG Business Analytics, UAM 2025/26
"""

# ══════════════════════════════════════════════════════════════════
# 0.  INSTALACIÓN (descomenta y ejecuta una vez si no las tienes)
# ══════════════════════════════════════════════════════════════════
# import subprocess
# subprocess.run(["pip", "install", "libpysal", "esda", "spreg", "geopandas"])

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Espaciales
import libpysal
from libpysal.weights import Queen, KNN
import esda
from esda.moran import Moran, Moran_Local
import spreg

# Visualización
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ══════════════════════════════════════════════════════════════════
# 1.  RUTAS  ← ajusta si ejecutas en local
# ══════════════════════════════════════════════════════════════════
FILE_PANEL = r"C:\Users\USUARIO\Desktop\4º\TFG--\data\data2\CODIGOS BUENOS\Outputs Codigos\panel_final_anual_v4.xlsx"

# ══════════════════════════════════════════════════════════════════
# 2.  CARGA DEL PANEL
# ══════════════════════════════════════════════════════════════════

panel = pd.read_excel(FILE_PANEL)
print(f"Panel cargado → {panel.shape[0]} obs | {panel['COUNTRY'].nunique()} países")

# Variables del modelo (mismas que en FE para comparabilidad)
DEPVAR   = "INSTABILITY_INDEX"
XVARS    = ["Inflation_CPI", "GDP_Growth", "Unemployment",
            "Gov_debt_GDP", "Energy_Imports", "Rule_of_Law", "VolPS"]

# Filtrar periodo y eliminar NaN en variables del modelo
df_model = panel[["COUNTRY", "YEAR", DEPVAR] + XVARS].dropna().copy()
print(f"Muestra modelo → {len(df_model)} obs | años: {sorted(df_model['YEAR'].unique())}")

# Lista de países en el modelo (orden fijo → crítico para W)
PAISES = sorted(df_model["COUNTRY"].unique())
N = len(PAISES)
print(f"Países ({N}): {PAISES}")

# ══════════════════════════════════════════════════════════════════
# 3.  COORDENADAS DE CAPITALES
#     Usadas para:
#       (a) matriz de distancia inversa
#       (b) verificar contigüidad
# ══════════════════════════════════════════════════════════════════
CAPITALES = {
    "Austria":       (48.2092, 16.3728),
    "Belgium":       (50.8503, 4.3517),
    "Bulgaria":      (42.6977, 23.3219),
    "Croatia":       (45.8150, 15.9819),
    "Cyprus":        (35.1676, 33.3736),
    "Czech Republic":(50.0755, 14.4378),
    "Denmark":       (55.6761, 12.5683),
    "Estonia":       (59.4370, 24.7536),
    "Finland":       (60.1699, 24.9384),
    "France":        (48.8566,  2.3522),
    "Germany":       (52.5200, 13.4050),
    "Greece":        (37.9838, 23.7275),
    "Hungary":       (47.4979, 19.0402),
    "Ireland":       (53.3498, -6.2603),
    "Italy":         (41.9028, 12.4964),
    "Latvia":        (56.9460, 24.1059),
    "Lithuania":     (54.6872, 25.2797),
    "Luxembourg":    (49.6117,  6.1319),
    "Malta":         (35.9042, 14.5189),
    "Netherlands":   (52.3702,  4.8952),
    "Poland":        (52.2297, 21.0122),
    "Portugal":      (38.7169, -9.1399),
    "Romania":       (44.4268, 26.1025),
    "Russia":        (55.7558, 37.6173),
    "Slovenia":      (46.0569, 14.5058),
    "Spain":         (40.4168, -3.7038),
    "Sweden":        (59.3293, 18.0686),
    "United States": (38.9072,-77.0369),
}

# Filtrar solo los países que están en el modelo
coords = pd.DataFrame(
    [(p, CAPITALES[p][0], CAPITALES[p][1]) for p in PAISES if p in CAPITALES],
    columns=["COUNTRY", "lat", "lon"]
).set_index("COUNTRY").loc[PAISES]

print(f"\nCoordenadas disponibles: {len(coords)}/{N} países")

# ══════════════════════════════════════════════════════════════════
# 4.  MATRICES DE PESOS ESPACIALES
# ══════════════════════════════════════════════════════════════════

# ── 4a. Contigüidad: definida manualmente (frontera terrestre) ────
# Fuente: geografía política europea
# Nota: Cyprus, Malta, Ireland, UK son islas → vecinos más cercanos
CONTIG = {
    "Austria":       ["Germany","Czech Republic","Slovakia","Hungary","Slovenia","Italy","Switzerland","Liechtenstein"],
    "Belgium":       ["France","Germany","Luxembourg","Netherlands"],
    "Bulgaria":      ["Romania","Greece","Serbia","North Macedonia","Turkey"],
    "Croatia":       ["Slovenia","Hungary","Serbia","Bosnia and Herzegovina","Montenegro"],
    "Cyprus":        ["Greece"],          # isla → vecino más cercano en el modelo
    "Czech Republic":["Germany","Poland","Slovakia","Austria"],
    "Denmark":       ["Germany","Sweden"],
    "Estonia":       ["Latvia","Finland","Russia"],
    "Finland":       ["Sweden","Norway","Estonia","Russia"],
    "France":        ["Belgium","Luxembourg","Germany","Switzerland","Italy","Spain","Monaco","Andorra"],
    "Germany":       ["France","Belgium","Luxembourg","Netherlands","Denmark","Poland","Czech Republic","Austria","Switzerland"],
    "Greece":        ["Bulgaria","Albania","North Macedonia","Turkey","Cyprus"],
    "Hungary":       ["Austria","Slovakia","Ukraine","Romania","Serbia","Croatia","Slovenia"],
    "Ireland":       ["United Kingdom"],  # isla → UK
    "Italy":         ["France","Switzerland","Austria","Slovenia","San Marino","Vatican"],
    "Latvia":        ["Estonia","Lithuania","Russia","Belarus"],
    "Lithuania":     ["Latvia","Poland","Belarus","Russia"],
    "Luxembourg":    ["Belgium","France","Germany"],
    "Malta":         ["Italy"],           # isla → Italia
    "Netherlands":   ["Belgium","Germany"],
    "Poland":        ["Germany","Czech Republic","Slovakia","Ukraine","Belarus","Lithuania","Russia"],
    "Portugal":      ["Spain"],
    "Romania":       ["Hungary","Serbia","Bulgaria","Moldova","Ukraine"],
    "Russia":        ["Estonia","Latvia","Lithuania","Poland","Finland","Norway","Belarus","Ukraine"],
    "Slovenia":      ["Italy","Austria","Hungary","Croatia"],
    "Spain":         ["France","Portugal","Andorra","Morocco"],
    "Sweden":        ["Norway","Finland","Denmark"],
    "United States": ["Canada","Mexico"],  # no frontera con Europa → se asigna 0 vecinos
}

# Construir matriz de adyacencia N×N
W_contig_arr = np.zeros((N, N))
for i, pais_i in enumerate(PAISES):
    vecinos = CONTIG.get(pais_i, [])
    for j, pais_j in enumerate(PAISES):
        if pais_j in vecinos:
            W_contig_arr[i, j] = 1

# Asegurar simetría
W_contig_arr = np.maximum(W_contig_arr, W_contig_arr.T)

# USA y Cyprus/Malta/Ireland sin vecinos en el panel → asignar KNN=1
for i, pais in enumerate(PAISES):
    if W_contig_arr[i, :].sum() == 0:
        # Calcular distancia a todos los demás y asignar al más cercano
        lat_i, lon_i = coords.loc[pais, ["lat", "lon"]]
        dists = []
        for j, otro in enumerate(PAISES):
            if j != i:
                lat_j, lon_j = coords.loc[otro, ["lat", "lon"]]
                d = np.sqrt((lat_i - lat_j)**2 + (lon_i - lon_j)**2)
                dists.append((j, d))
        dists.sort(key=lambda x: x[1])
        j_near = dists[0][0]
        W_contig_arr[i, j_near] = 1
        W_contig_arr[j_near, i] = 1
        print(f"  {pais} (isla/lejano) → vecino asignado: {PAISES[j_near]}")

# Estandarización por filas (row-standardized)
row_sums = W_contig_arr.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # evitar división por cero
W_contig_std = W_contig_arr / row_sums

# Convertir a objeto W de libpysal
neighbors_c = {i: list(np.where(W_contig_arr[i] > 0)[0]) for i in range(N)}
weights_c   = {i: [1.0/len(v)]*len(v) if len(v) > 0 else [] for i, v in neighbors_c.items()}
W_contig = libpysal.weights.W(neighbors_c, weights_c, id_order=list(range(N)))
W_contig.transform = "R"

print(f"\nW contigüidad → {W_contig.n} unidades | "
      f"conectividad media: {np.mean([len(v) for v in neighbors_c.values()]):.2f} vecinos")

# ── 4b. Distancia inversa entre capitales ────────────────────────
from scipy.spatial.distance import cdist

coords_arr = coords[["lat","lon"]].values
dist_matrix = cdist(coords_arr, coords_arr, metric="euclidean")

# Inversa de la distancia (0 en diagonal)
with np.errstate(divide="ignore"):
    W_dist_arr = np.where(dist_matrix > 0, 1.0 / dist_matrix, 0)

# Row-standardize
row_sums_d = W_dist_arr.sum(axis=1, keepdims=True)
row_sums_d[row_sums_d == 0] = 1
W_dist_std = W_dist_arr / row_sums_d

neighbors_d = {i: [j for j in range(N) if j != i] for i in range(N)}
weights_d   = {i: list(W_dist_std[i, [j for j in range(N) if j != i]]) for i in range(N)}
W_dist = libpysal.weights.W(neighbors_d, weights_d, id_order=list(range(N)))
W_dist.transform = "R"

print(f"W distancia   → {W_dist.n} unidades | inversa distancia, row-standardized")

# ══════════════════════════════════════════════════════════════════
# 5.  TEST DE MORAN'S I GLOBAL  →  H1
#     Se calcula por año para ver evolución temporal
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  MORAN'S I GLOBAL — por año (W contigüidad)")
print("═"*60)
print(f"  {'Año':<6} {'Moran I':>9} {'E[I]':>8} {'p-valor':>9} {'Signif.'}")
print("  " + "-"*50)

moran_results = []
for year in sorted(df_model["YEAR"].unique()):
    df_y = df_model[df_model["YEAR"] == year].copy()
    df_y = df_y.set_index("COUNTRY").reindex(PAISES)

    y_vec = df_y[DEPVAR].values

    # Si hay NaN en algún país ese año, imputar con la media
    if np.isnan(y_vec).any():
        y_vec = np.where(np.isnan(y_vec), np.nanmean(y_vec), y_vec)

    mi = Moran(y_vec, W_contig, permutations=999)
    sig = "***" if mi.p_sim < 0.01 else ("**" if mi.p_sim < 0.05 else ("*" if mi.p_sim < 0.1 else ""))
    print(f"  {year:<6} {mi.I:>9.4f} {mi.EI:>8.4f} {mi.p_sim:>9.4f}  {sig}")
    moran_results.append({"year": year, "I": mi.I, "EI": mi.EI,
                           "p_sim": mi.p_sim, "sig": sig})

moran_df = pd.DataFrame(moran_results)

# Gráfico Moran's I por año
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["red" if p < 0.1 else "steelblue" for p in moran_df["p_sim"]]
ax.bar(moran_df["year"], moran_df["I"], color=colors, edgecolor="white", width=0.6)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axhline(moran_df["EI"].mean(), color="gray", linewidth=0.8,
           linestyle=":", label="E[I] teórico")
ax.set_xlabel("Año")
ax.set_ylabel("Moran's I")
ax.set_title("Autocorrelación espacial global de la inestabilidad política\n"
             "(Moran's I, W contigüidad, permutaciones=999)\n"
             "Rojo = significativo al 10%")
ax.legend()
plt.tight_layout()
plt.savefig("moran_i_por_año.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → Gráfico guardado: moran_i_por_año.png")

# Moran's I sobre panel completo (todas las obs)
print("\n" + "═"*60)
print("  MORAN'S I — panel completo (promedio ponderado)")
print("═"*60)
# Calcular sobre los residuos del FE (más correcto que sobre y bruto)
# Aquí usamos y bruto como aproximación; en el SAR se hace correctamente
y_all = df_model.groupby("COUNTRY")[DEPVAR].mean().reindex(PAISES).values
y_all = np.where(np.isnan(y_all), np.nanmean(y_all), y_all)
mi_all = Moran(y_all, W_contig, permutations=999)
print(f"  Moran's I = {mi_all.I:.4f}  (p = {mi_all.p_sim:.4f})")

# ══════════════════════════════════════════════════════════════════
# 6.  MORAN'S I LOCAL (LISA) — año con mayor señal
# ══════════════════════════════════════════════════════════════════
# Tomar el año con Moran's I más alto
best_year = moran_df.loc[moran_df["I"].idxmax(), "year"]
df_best   = df_model[df_model["YEAR"] == best_year].set_index("COUNTRY").reindex(PAISES)
y_best    = df_best[DEPVAR].values
y_best    = np.where(np.isnan(y_best), np.nanmean(y_best), y_best)

lisa = Moran_Local(y_best, W_contig, permutations=999)

print(f"\n  LISA ({best_year}) — clusters significativos (p<0.05):")
for i, pais in enumerate(PAISES):
    if lisa.p_sim[i] < 0.05:
        tipo = {1: "HH (alto-alto)", 2: "LH (bajo-alto)",
                3: "LL (bajo-bajo)", 4: "HL (alto-bajo)"}.get(lisa.q[i], "ns")
        print(f"    {pais:<20} {tipo}  (p={lisa.p_sim[i]:.3f})")

# ══════════════════════════════════════════════════════════════════
# 7.  MODELO SAR PANEL  →  H1 (ρ) + H2 (vs FE)
#
#     spreg.Panel_RE_Lag es el SAR de panel más estable en spreg.
#     Alternativa: GM_Lag (GMM) si quieres efectos fijos.
#     Aquí usamos ML con efectos aleatorios + dummies de año como Xs.
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  MODELO SAR — Spatial Autoregressive Panel")
print("═"*60)

# Preparar matrices para spreg
# spreg necesita arrays numpy ordenados por (año, país) — stacked
years_sorted  = sorted(df_model["YEAR"].unique())
T = len(years_sorted)

# Reindexar: para cada año, todos los países en el mismo orden (PAISES)
rows = []
for yr in years_sorted:
    df_yr = df_model[df_model["YEAR"] == yr].set_index("COUNTRY").reindex(PAISES)
    rows.append(df_yr)
df_stack = pd.concat(rows).reset_index()

# Imputar NaN con media del país (los pocos que quedan)
for col in [DEPVAR] + XVARS:
    if df_stack[col].isna().any():
        df_stack[col] = df_stack.groupby("COUNTRY")[col].transform(
            lambda x: x.fillna(x.mean())
        )

y_sar = df_stack[DEPVAR].values.reshape(-1, 1)
X_sar = df_stack[XVARS].values

# Dummies de año (quitar una para evitar multicolinealidad)
year_dummies = pd.get_dummies(df_stack["YEAR"], drop_first=True).astype(float).values
X_sar_full   = np.hstack([X_sar, year_dummies])

print(f"  Stacked panel → y: {y_sar.shape} | X: {X_sar_full.shape}")
print(f"  T={T} años × N={N} países = {T*N} obs")

# SAR con spreg (Panel_RE_Lag)
try:
    sar = spreg.Panel_RE_Lag(
        y        = y_sar,
        x        = X_sar_full,
        w        = W_contig,
        name_y   = DEPVAR,
        name_x   = XVARS + [f"D_{y}" for y in years_sorted[1:]],
        name_w   = "W_contig",
        name_ds  = "panel_TFG"
    )

    print(f"\n  ρ (spatial lag) = {sar.rho:.4f}  "
          f"(z = {sar.z_stat[0][0]:.3f}, p = {sar.z_stat[0][1]:.4f})")

    sig_rho = ("***" if sar.z_stat[0][1] < 0.01 else
               ("**"  if sar.z_stat[0][1] < 0.05 else
                ("*"   if sar.z_stat[0][1] < 0.1  else "ns")))
    print(f"  Significatividad ρ: {sig_rho}")

    # Tabla de coeficientes
    print(f"\n  {'Variable':<28} {'Coef.':>8} {'Std.Err.':>9} {'z':>7} {'p':>7} {'Sig.'}")
    print("  " + "-"*65)
    betas = sar.betas.flatten()
    # spreg devuelve [constante, X..., lambda/rho al final]
    var_names = ["Constante"] + XVARS + [f"D_{y}" for y in years_sorted[1:]]
    for k, name in enumerate(var_names):
        if k < len(betas) - 1:  # excluir el último (es rho)
            z_s  = sar.z_stat[k+1][0] if k+1 < len(sar.z_stat) else np.nan
            p_s  = sar.z_stat[k+1][1] if k+1 < len(sar.z_stat) else np.nan
            sig  = ("***" if p_s < 0.01 else ("**" if p_s < 0.05 else
                    ("*" if p_s < 0.1 else ""))) if not np.isnan(p_s) else ""
            print(f"  {name:<28} {betas[k]:>8.4f} {'—':>9} {z_s:>7.3f} {p_s:>7.4f} {sig}")

    print(f"\n  Log-likelihood : {sar.logll:.4f}")
    print(f"  AIC            : {sar.aic:.4f}")
    print(f"  Schwarz (BIC)  : {sar.schwarz:.4f}")

    sar_ok = True

except Exception as e:
    print(f"  ⚠️  SAR Panel_RE_Lag falló: {e}")
    print("  → Intentando con GM_Lag (GMM)...")

    try:
        # Alternativa GMM
        sar = spreg.GM_Lag(
            y      = y_sar,
            x      = X_sar_full,
            w      = W_contig,
            name_y = DEPVAR,
            name_x = XVARS + [f"D_{y}" for y in years_sorted[1:]],
        )
        print(f"\n  ρ (spatial lag) = {sar.rho:.4f}")
        print(sar.summary)
        sar_ok = True
    except Exception as e2:
        print(f"  ❌ GM_Lag también falló: {e2}")
        sar_ok = False

# ══════════════════════════════════════════════════════════════════
# 8.  MODELO SDM (robustez) — Spatial Durbin
#     SDM = SAR + WX (lag espacial de las X)
#     spreg no tiene SDM directo → lo aproximamos añadiendo WX como regresores
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  MODELO SDM (robustez) — Spatial Durbin")
print("═"*60)

# Calcular WX para las variables principales
W_full = np.tile(W_contig_std, (T, 1))  # repetir W para cada periodo
# Más correcto: calcular WX año a año
WX_list = []
for yr in years_sorted:
    df_yr   = df_model[df_model["YEAR"] == yr].set_index("COUNTRY").reindex(PAISES)
    X_yr    = df_yr[XVARS].fillna(df_yr[XVARS].mean()).values
    WX_yr   = W_contig_std @ X_yr
    WX_list.append(WX_yr)
WX_stack = np.vstack(WX_list)

# SDM = X + WX como regresores en SAR
X_sdm = np.hstack([X_sar, WX_stack, year_dummies])
wx_names = [f"W_{v}" for v in XVARS]

try:
    sdm = spreg.Panel_RE_Lag(
        y      = y_sar,
        x      = X_sdm,
        w      = W_contig,
        name_y = DEPVAR,
        name_x = XVARS + wx_names + [f"D_{y}" for y in years_sorted[1:]],
        name_w = "W_contig_SDM",
        name_ds= "panel_TFG"
    )

    print(f"\n  ρ SDM = {sdm.rho:.4f}  "
          f"(z = {sdm.z_stat[0][0]:.3f}, p = {sdm.z_stat[0][1]:.4f})")
    print(f"  Log-likelihood : {sdm.logll:.4f}")
    print(f"  AIC            : {sdm.aic:.4f}")
    print(f"  Schwarz (BIC)  : {sdm.schwarz:.4f}")
    sdm_ok = True

except Exception as e:
    print(f"  ⚠️  SDM falló: {e}")
    sdm_ok = False

# ══════════════════════════════════════════════════════════════════
# 9.  COMPARACIÓN FE vs SAR vs SDM  →  H2
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  COMPARACIÓN DE MODELOS  →  H2")
print("═"*60)
print(f"  {'Modelo':<20} {'AIC':>10} {'BIC':>10} {'Log-lik':>10} {'ρ':>8}")
print("  " + "-"*58)

# FE (AIC aproximado desde R²)
# AIC del FE no lo da linearmodels directamente → se omite o se calcula manual
print(f"  {'FE base (M0)':<20} {'—':>10} {'—':>10} {'—':>10} {'—':>8}")

if sar_ok:
    print(f"  {'SAR':<20} {sar.aic:>10.2f} {sar.schwarz:>10.2f} "
          f"{sar.logll:>10.2f} {sar.rho:>8.4f}")
if sdm_ok:
    print(f"  {'SDM':<20} {sdm.aic:>10.2f} {sdm.schwarz:>10.2f} "
          f"{sdm.logll:>10.2f} {sdm.rho:>8.4f}")

print("\n  Regla: menor AIC/BIC = mejor ajuste")
print("  H2 se confirma si SAR/SDM < AIC del modelo no espacial equivalente")

# ══════════════════════════════════════════════════════════════════
# 10.  EFECTOS DIRECTOS, INDIRECTOS Y TOTALES  (SAR)
#      En SAR: efecto total = β / (1-ρ)
#              efecto directo ≈ β  (simplificación para TFG)
#              efecto indirecto = efecto total - directo
# ══════════════════════════════════════════════════════════════════
if sar_ok:
    print("\n" + "═"*60)
    print("  EFECTOS DIRECTOS / INDIRECTOS / TOTALES (SAR)")
    print("═"*60)
    print(f"  ρ = {sar.rho:.4f}  →  multiplicador = 1/(1-ρ) = {1/(1-sar.rho):.4f}")
    print(f"\n  {'Variable':<25} {'Directo':>10} {'Indirecto':>11} {'Total':>10}")
    print("  " + "-"*58)

    multiplier = 1 / (1 - sar.rho)
    betas_x    = sar.betas[1:len(XVARS)+1].flatten()  # solo las X, no dummies ni cte

    for k, var in enumerate(XVARS):
        beta     = betas_x[k]
        total    = beta * multiplier
        directo  = beta
        indir    = total - directo
        print(f"  {var:<25} {directo:>10.4f} {indir:>11.4f} {total:>10.4f}")

# ══════════════════════════════════════════════════════════════════
# 11.  GUARDAR RESULTADOS
# ══════════════════════════════════════════════════════════════════
moran_df.to_excel("resultados_moran_i.xlsx", index=False)
print(f"\n✅ Moran's I guardado: resultados_moran_i.xlsx")
print(f"✅ Gráfico Moran's I: moran_i_por_año.png")
