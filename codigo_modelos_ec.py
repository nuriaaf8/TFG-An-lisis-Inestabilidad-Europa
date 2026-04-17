# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:18:21 2026

@author: USUARIO
"""

# -*- coding: utf-8 -*-
"""
codigo_modelos_ec.py
=====================
Script de estimación econométrica.
Carga el panel construido por codigo_panel_v4.py y estima:

  M0 — FE base        : efectos fijos país + año, sin rezago ni espacial
  M1 — FE dinámico    : añade L1_INSTABILITY_INDEX
  M2 — FE interacciones: añade interacciones post-2022 (H3 amplificada)
  M3 — FE completo    : M1 + M2 juntos (especificación principal)

Errores estándar: cluster por país en todos los modelos.

Hipótesis a probar:
  H3 — Inflation_CPI, GDP_Growth              → coeficientes significativos
  H4 — Rule_of_Law                            → coeficiente negativo
  H5 — VolPS (volatilidad Political_Stability)→ coeficiente positivo
  (H1/H2 se contrastan en el script espacial --> "codigo_modelo_espacial.py")

Autor: Nuria Adame Fuentes — TFG Business Analytics, UAM 2025/26
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# 0.  RUTA  ← ajusta si ejecutas en local
# ══════════════════════════════════════════════════════════════════
FILE_PANEL = r"C:\Users\USUARIO\Desktop\4º\TFG--\data\data2\CODIGOS BUENOS\Outputs Codigos\panel_final_anual_v4.xlsx"

# ══════════════════════════════════════════════════════════════════
# 1.  CARGA DEL PANEL
# ══════════════════════════════════════════════════════════════════
panel = pd.read_excel(FILE_PANEL)

print(f"Panel cargado → shape: {panel.shape}")
print(f"Países: {panel['COUNTRY'].nunique()} | Años: {sorted(panel['YEAR'].unique())}")

# ══════════════════════════════════════════════════════════════════
# 2.  VARIABLES DEL MODELO
# ══════════════════════════════════════════════════════════════════
DEPVAR = "INSTABILITY_INDEX"

# Variables base (M0 y todos los modelos)
XVARS_BASE = [
    "Inflation_CPI",    # H3 — canal inflacionario
    "GDP_Growth",       # H3 — canal crecimiento
    "Unemployment",     # control macro
    "Gov_debt_GDP",     # vulnerabilidad fiscal
    "Energy_Imports",   # vulnerabilidad energética
    "Rule_of_Law",      # H4 — amortiguación institucional
    "VolPS",            # H5 — polarización (volatilidad Political Stability)
]

# Variables adicionales por modelo
VAR_LAG          = "L1_INSTABILITY_INDEX"          # M1
VARS_INTERACT    = ["Inflation_post22",             # M2 / M3
                    "Energy_post22",
                    "Inflation_x_RoL"]


# Todas las variables que necesitamos para poder dropear NaN correctamente
ALL_VARS = [DEPVAR] + XVARS_BASE + [VAR_LAG] + VARS_INTERACT

# ══════════════════════════════════════════════════════════════════
# 3.  PREPARACIÓN DEL PANEL
# ══════════════════════════════════════════════════════════════════

# 3a. Quedarse solo con columnas relevantes
cols_keep = ["COUNTRY", "YEAR"] + ALL_VARS
panel = panel[[c for c in cols_keep if c in panel.columns]].copy()

# 3b. Recalcular interacciones por si no estuvieran en el CSV
if "Inflation_post22" not in panel.columns:
    panel["D_UKR"] = (panel["YEAR"] >= 2022).astype(int)
    panel["Inflation_post22"] = panel["Inflation_CPI"] * panel["D_UKR"]
if "Energy_post22" not in panel.columns:
    panel["D_UKR"] = (panel["YEAR"] >= 2022).astype(int)
    panel["Energy_post22"] = panel["Energy_Imports"] * panel["D_UKR"]
if "Inflation_x_RoL" not in panel.columns:
    panel["Inflation_x_RoL"] = panel["Inflation_CPI"] * panel["Rule_of_Law"]

# 3c. Índice múltiple COUNTRY–YEAR (requerido por linearmodels)
panel = panel.set_index(["COUNTRY", "YEAR"])

# ══════════════════════════════════════════════════════════════════
# 4.  FUNCIÓN AUXILIAR: ESTIMAR Y MOSTRAR RESULTADOS
# ══════════════════════════════════════════════════════════════════
def estimar_fe(data, depvar, xvars, nombre_modelo, descripcion):
    """
    Estima un modelo PanelOLS con efectos fijos país + año
    y errores cluster por país (cluster_entity=True).
    Devuelve el objeto resultado de linearmodels.
    """
    # Submuestra limpia: solo filas sin NaN en las variables del modelo
    vars_modelo = [depvar] + xvars
    df = data[vars_modelo].dropna().copy()

    n_obs     = len(df)
    n_paises  = df.index.get_level_values("COUNTRY").nunique()
    n_anos    = df.index.get_level_values("YEAR").nunique()
    paises    = sorted(df.index.get_level_values("COUNTRY").unique())
    anos      = sorted(df.index.get_level_values("YEAR").unique())

    print("\n" + "═" * 65)
    print(f"  {nombre_modelo}  —  {descripcion}")
    print("═" * 65)
    print(f"  Observaciones : {n_obs}")
    print(f"  Países        : {n_paises}  → {paises}")
    print(f"  Años          : {anos}")
    print(f"  Variables     : {xvars}")

    # Fórmula: depvar ~ xvars + EntityEffects + TimeEffects
    formula = depvar + " ~ " + " + ".join(xvars) + " + EntityEffects + TimeEffects"

    modelo = PanelOLS.from_formula(formula, data=df)
    resultado = modelo.fit(cov_type="clustered", cluster_entity=True)

    # Tabla de resultados limpia
    res_df = pd.DataFrame({
        "Coef."   : resultado.params.round(4),
        "Std.Err." : resultado.std_errors.round(4),
        "t-stat"  : resultado.tstats.round(3),
        "p-value" : resultado.pvalues.round(4),
        "Signif." : resultado.pvalues.apply(
            lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        )
    })
    print("\n" + res_df.to_string())
    print(f"\n  R² within  : {resultado.rsquared:.4f}")
    print(f"  R² between : {resultado.rsquared_between:.4f}")
    print(f"  R² overall : {resultado.rsquared_overall:.4f}")
    print(f"  F-stat     : {resultado.f_statistic.stat:.3f}  "
          f"(p = {resultado.f_statistic.pval:.4f})")

    return resultado


# ══════════════════════════════════════════════════════════════════
# 5.  ESTIMACIÓN DE LOS MODELOS
# ══════════════════════════════════════════════════════════════════

# ── M0: FE base ──────────────────────────────────────────────────
res_M0 = estimar_fe(
    data        = panel,
    depvar      = DEPVAR,
    xvars       = XVARS_BASE,
    nombre_modelo = "M0 — FE BASE",
    descripcion = "Efectos fijos país + año | sin rezago ni interacciones"
)

# ── M1: FE dinámico (añade rezago dependiente) ───────────────────
res_M1 = estimar_fe(
    data        = panel,
    depvar      = DEPVAR,
    xvars       = XVARS_BASE + [VAR_LAG],
    nombre_modelo = "M1 — FE DINÁMICO",
    descripcion = "M0 + L1_INSTABILITY_INDEX (persistencia)"
)

# ── M2: FE con interacciones post-2022 ───────────────────────────
res_M2 = estimar_fe(
    data        = panel,
    depvar      = DEPVAR,
    xvars       = XVARS_BASE + VARS_INTERACT,
    nombre_modelo = "M2 — FE INTERACCIONES",
    descripcion = "M0 + interacciones post-2022 (H3 amplificada)"
)

# ── M3: FE completo (especificación principal) ───────────────────
res_M3 = estimar_fe(
    data        = panel,
    depvar      = DEPVAR,
    xvars       = XVARS_BASE + [VAR_LAG] + VARS_INTERACT,
    nombre_modelo = "M3 — FE COMPLETO (especificación principal)",
    descripcion = "M0 + rezago + interacciones post-2022"
)

# ══════════════════════════════════════════════════════════════════
# 6.  TABLA COMPARATIVA DE COEFICIENTES
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "═" * 65)
print("  TABLA COMPARATIVA — coeficientes (errores cluster país)")
print("═" * 65)

modelos = {"M0": res_M0, "M1": res_M1, "M2": res_M2, "M3": res_M3}
todas_vars = XVARS_BASE + [VAR_LAG] + VARS_INTERACT

filas = []
for var in todas_vars:
    fila = {"Variable": var}
    for nombre, res in modelos.items():
        if var in res.params.index:
            coef = res.params[var]
            pval = res.pvalues[var]
            sig  = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
            fila[nombre] = f"{coef:.4f}{sig}"
        else:
            fila[nombre] = "—"
    filas.append(fila)

# Añadir métricas de ajuste
for metrica, attr in [("R² within", "rsquared"),
                       ("N obs",     None)]:
    fila = {"Variable": metrica}
    for nombre, res in modelos.items():
        if metrica == "N obs":
            fila[nombre] = str(res.nobs)
        else:
            fila[nombre] = f"{getattr(res, attr):.4f}"
    filas.append(fila)

tabla = pd.DataFrame(filas).set_index("Variable")
print(tabla.to_string())
print("\nNota: *** p<0.01  ** p<0.05  * p<0.1  | Errores cluster por país")

# ══════════════════════════════════════════════════════════════════
# 7.  GUARDAR TABLA COMPARATIVA EN EXCEL
# ══════════════════════════════════════════════════════════════════
tabla.to_excel("resultados_modelos_FE.xlsx")
print("\n Tabla comparativa guardada en: resultados_modelos_FE.xlsx")


# ══════════════════════════════════════════════════════════════════
# 8.  AIC / BIC DE LOS MODELOS FE  →  comparación con SAR/SDM (H2)
#
#     linearmodels no expone AIC directamente.
#     Fórmula estándar OLS:
#       loglik  = -N/2 * (1 + log(2π) + log(RSS/N))
#       AIC     = -2*loglik + 2*k
#       BIC     = -2*loglik + k*log(N)
#     k = betas + efectos fijos absorbidos (N_países-1 + N_años-1)
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "═" * 65)
print("  AIC / BIC — modelos FE  (para comparación con SAR/SDM → H2)")
print("═" * 65)
print(f"  {'Modelo':<22} {'N':>5} {'k':>5} {'LogLik':>10} {'AIC':>10} {'BIC':>10}")
print("  " + "-"*62)

N_PAISES_FE = 27
N_ANOS_FE   = 5   # 2019-2023

def calc_aic_bic(resultado, n_paises, n_anos):
    resid   = resultado.resids.values
    n       = len(resid)
    rss     = np.sum(resid**2)
    sigma2  = rss / n
    loglik  = -n/2 * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    k_betas = len(resultado.params)
    k_fe    = (n_paises - 1) + (n_anos - 1)
    k       = k_betas + k_fe
    aic     = -2 * loglik + 2 * k
    bic     = -2 * loglik + k * np.log(n)
    return loglik, aic, bic, k, n

aic_resultados = {}
for nombre, res in modelos.items():
    loglik, aic, bic, k, n = calc_aic_bic(res, N_PAISES_FE, N_ANOS_FE)
    aic_resultados[nombre] = {"LogLik": loglik, "AIC": aic, "BIC": bic}
    print(f"  {nombre:<22} {n:>5} {k:>5} {loglik:>10.2f} {aic:>10.2f} {bic:>10.2f}")

print("\n  Referencia modelos espaciales (script espacial):")
print(f"  {'SAR':<22} {'135':>5} {'—':>5} {'-190.99':>10} {'407.99':>10} {'445.76':>10}")
print(f"  {'SDM':<22} {'135':>5} {'—':>5} {'-185.80':>10} {'411.61':>10} {'469.71':>10}")
print("\n  → H2 se confirma si AIC(SAR) < AIC(FE base M0)")

# Añadir AIC/BIC a la tabla Excel
for metrica in ["LogLik", "AIC", "BIC"]:
    fila = {"Variable": metrica}
    for nombre in modelos.keys():
        fila[nombre] = f"{aic_resultados[nombre][metrica]:.2f}"
    filas.append(fila)

tabla_final = pd.DataFrame(filas).set_index("Variable")
tabla_final.to_excel("resultados_modelos_FE.xlsx")
print("\n✅ Tabla actualizada con AIC/BIC: resultados_modelos_FE.xlsx")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# crear grupo institucional: alto/bajo según la mediana
panel["RoL_group"] = pd.qcut(
    panel["Rule_of_Law"],
    q=2,
    labels=["Bajo Rule_of_Law", "Alto Rule_of_Law"]
)

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=panel,
    x="Inflation_CPI",
    y="INSTABILITY_INDEX",
    hue="RoL_group"
)

sns.regplot(
    data=panel[panel["RoL_group"] == "Bajo Rule_of_Law"],
    x="Inflation_CPI",
    y="INSTABILITY_INDEX",
    scatter=False
)

sns.regplot(
    data=panel[panel["RoL_group"] == "Alto Rule_of_Law"],
    x="Inflation_CPI",
    y="INSTABILITY_INDEX",
    scatter=False
)

plt.xlabel("Inflation_CPI")
plt.ylabel("INSTABILITY_INDEX")
plt.title("Inflation and political instability by institutional quality")
plt.show()
