# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:55:46 2026

@author: USUARIO
"""


# -*- coding: utf-8 -*-
"""
codigo_panel_v4.py
=================
Construcción del panel anual definitivo.

Fuentes:
  - ACLED mensual        → acled_country_month_eda.xlsx
  - Macro + WGI          → macro_institucional_panel.xlsx
  - The Global Economy   → eda_variables_the_global_economy_tfg.xlsx

Output:
  - panel_final_anual_v4.xlsx  /  panel_final_anual_v4.csv

Panel resultante:
  - 27 países × 2018–2023 = 134 observaciones
  - Variable dependiente principal  : INSTABILITY_INDEX  (z-score continuo)
  - Variable dependiente alternativa: INSTABILITY_BINARY (binaria anual)

Hipótesis que cubre:
  H1 — Difusión espacial     → variable dependiente + matriz W (paso posterior)
  H2 — Mejora espacial       → se contrasta en el modelo (paso posterior)
  H3 — Canal macro           → Inflation_CPI, GDP_Growth
  H4 — Amortiguación inst.   → Rule_of_Law  (WGI)
  H5 — Polarización política → VolPS  (volatilidad anual de Political_Stability WGI)
  H6 — ML predictivo         → panel completo entra en RF/XGBoost (paso posterior)

Autor: Nuria Adame Fuentes  —  TFG Business Analytics, UAM 2025/26
"""

import os
import pandas as pd
import numpy as np

print("Directorio de trabajo:", os.getcwd())

# ══════════════════════════════════════════════════════════════════
# 0.  RUTAS  
# ══════════════════════════════════════════════════════════════════
FILE_ACLED = r"C:\Users\USUARIO\Desktop\4º\TFG--\data\data2\AECLED\acled_country_month_eda.xlsx"
FILE_MACRO = r"C:\Users\USUARIO\Desktop\4º\TFG--\data\data2\WorldBank (Macroeconómicas)\Merge_WDI_WGI\macro_institucional_panel.xlsx"
FILE_TGE   = r"C:\Users\USUARIO\Desktop\4º\TFG--\data\data2\TheGlobalEconomy\eda_variables_the_global_economy_tfg.xlsx"

OUTFILE_XLSX = r"panel_final_anual_v4.xlsx"
OUTFILE_CSV  = r"panel_final_anual_v4.csv"

# ══════════════════════════════════════════════════════════════════
# 1.  CARGA DE LAS TRES BASES de DATOS
# ══════════════════════════════════════════════════════════════════
acled = pd.read_excel(FILE_ACLED)
macro = pd.read_excel(FILE_MACRO)
tge   = pd.read_excel(FILE_TGE)

print(f"\nShapes iniciales → ACLED: {acled.shape} | MACRO: {macro.shape} | TGE: {tge.shape}")

# ══════════════════════════════════════════════════════════════════
# 2.  HOMOGENEIZAR NOMBRES de PAÍSES (antes de merge)
# ══════════════════════════════════════════════════════════════════
macro["Country Name"] = macro["Country Name"].replace({
    "Russian Federation": "Russia",
    "Czechia":            "Czech Republic",
    "Slovak Republic":    "Slovakia",
})

tge["Country"] = tge["Country"].replace({
    "USA":                "United States",
    "Russian Federation": "Russia",
    "Czechia":            "Czech Republic",
    "Slovak Republic":    "Slovakia",
})

# ACLED ya tiene los nombres correctos; por si acaso:
acled["COUNTRY"] = acled["COUNTRY"].replace({
    "USA":                "United States",
    "Russian Federation": "Russia",
    "Czechia":            "Czech Republic",
    "Slovak Republic":    "Slovakia",
})

# ══════════════════════════════════════════════════════════════════
# 3.  ACLED: MENSUAL → ANUAL
#
#   IMPORTANTE: se agrega DESDE el xlsx mensual original para
#   preservar la escala real del INSTABILITY_INDEX (−3.6 → +2.9).
#   NO se reconstruye desde el panel ya mergeado (eso aplasta la escala).
#
#   - INSTABILITY_INDEX    : media anual  (dependiente principal)
#   - INSTABILITY_0_100    : media anual  (versión reescalada 0-100)
#   - INSTABILITY_DUMMY_P75: media anual  (proporción de meses "altos")
#   - Eventos/fatalidades  : suma anual   (para ML y contexto)
# ══════════════════════════════════════════════════════════════════
acled_annual = (
    acled
    .groupby(["COUNTRY", "YEAR"], as_index=False)
    .agg(
        INSTABILITY_INDEX     = ("INSTABILITY_INDEX",        "mean"),
        INSTABILITY_0_100     = ("INSTABILITY_0_100",        "mean"),
        INSTABILITY_DUMMY_P75 = ("INSTABILITY_DUMMY_P75",    "mean"),
        EVENTS_PROTESTS       = ("EVENTS_PROTESTS",          "sum"),
        EVENTS_RIOTS          = ("EVENTS_RIOTS",             "sum"),
        EVENTS_VIOLENCE       = ("EVENTS_VIOLENCE_CIVILIANS","sum"),
        EVENTS_EXPLOSIONS     = ("EVENTS_EXPLOSIONS_REMOTE", "sum"),
        EVENTS_TOTAL          = ("EVENTS_TOTAL",             "sum"),
        FATALITIES_TOTAL      = ("FATALITIES_TOTAL",         "sum"),
    )
    .rename(columns={"COUNTRY": "COUNTRY", "YEAR": "YEAR"})
)

# Binaria anual: 1 si ≥50 % de los meses del año fueron "episodio alto"
acled_annual["INSTABILITY_BINARY"] = (
    acled_annual["INSTABILITY_DUMMY_P75"] >= 0.5
).astype(int)

print(f"ACLED anual → shape: {acled_annual.shape} | "
      f"INSTABILITY_INDEX rango: "
      f"{acled_annual['INSTABILITY_INDEX'].min():.3f} → "
      f"{acled_annual['INSTABILITY_INDEX'].max():.3f}")

# ══════════════════════════════════════════════════════════════════
# 4.  MACRO + WGI: selección de variables
# ══════════════════════════════════════════════════════════════════
macro_vars = [
    "Country Name", "Country Code", "Year",
    "GDP_Growth",             # H3 — canal macro
    "Inflation_CPI",          # H3 — canal macro (principal)
    "Unemployment",           # control
    "GDP_pc_const2015",       # control nivel de desarrollo
    "Debt_GDP",               # vulnerabilidad fiscal (muchos NaN → usar Gov_debt_GDP de TGE)
    "Energy_Imports",         # dependencia energética
    "Gini",                   # desigualdad
    "Rule_of_Law",            # H4 — amortiguación institucional
    "Political_Stability",    # base para H5 (volatilidad)
    "Control_of_Corruption",
    "Government_Effectiveness",
]
macro_sel = (
    macro[macro_vars]
    .rename(columns={"Country Name": "COUNTRY", "Year": "YEAR"})
)

# ── H5: volatilidad anual de Political_Stability (proxy de polarización) ──
# Cambio absoluto año a año dentro de cada país.
# Los primeros años por país tendrán NaN.
macro_sel = macro_sel.sort_values(["COUNTRY", "YEAR"])
macro_sel["VolPS"] = (
    macro_sel
    .groupby("COUNTRY")["Political_Stability"]
    .diff()
    .abs()
)

# ══════════════════════════════════════════════════════════════════
# 5.  THE GLOBAL ECONOMY: selección de variables
# ══════════════════════════════════════════════════════════════════
tge_vars = [
    "Country", "Year",
    "Government debt as percent of GDP",               # deuda pública (cubre Debt_GDP)
    "Military spending, percent of GDP",               # gasto defensa post-2022
    "Net energy imports as percent of total energy use",
    "Non-performing loans as percent of all bank loans",
    "Banking system z-scores",
    "Economic freedom, overall index (0-100)",
    "Property rights index (0-100)",
    "External debt, percent of GDP",
    "Current account balance as percent of GDP",
]
tge_sel = (
    tge[tge_vars]
    .rename(columns={
        "Country":                                              "COUNTRY",
        "Year":                                                 "YEAR",
        "Government debt as percent of GDP":                    "Gov_debt_GDP",
        "Military spending, percent of GDP":                    "Military_spending_GDP",
        "Net energy imports as percent of total energy use":    "Energy_imports_net",
        "Non-performing loans as percent of all bank loans":    "NPL",
        "Banking system z-scores":                             "Bank_Z",
        "Economic freedom, overall index (0-100)":             "Econ_freedom",
        "Property rights index (0-100)":                       "Property_rights",
        "External debt, percent of GDP":                       "External_debt_GDP",
        "Current account balance as percent of GDP":           "CA_GDP",
    })
)

# ══════════════════════════════════════════════════════════════════
# 6.  MERGE DE LAS TRES BASES
#
#   ACLED inner MACRO  → solo entran países con datos macro
#   result  left  TGE  → no perdemos observaciones si TGE tiene huecos
# ══════════════════════════════════════════════════════════════════
panel = (
    acled_annual
    .merge(macro_sel, on=["COUNTRY", "YEAR"], how="inner")
    .merge(tge_sel,   on=["COUNTRY", "YEAR"], how="left")
)

print(f"Tras merge → shape: {panel.shape}")

# ══════════════════════════════════════════════════════════════════
# 7.  FILTRO PERIODO 2018–2023
#
# ══════════════════════════════════════════════════════════════════
panel = panel[(panel["YEAR"] >= 2018) & (panel["YEAR"] <= 2023)].copy()
panel = panel.sort_values(["COUNTRY", "YEAR"]).reset_index(drop=True)

print(f"Tras filtro 2018-2023 → shape: {panel.shape} | países: {panel['COUNTRY'].nunique()}")

# ══════════════════════════════════════════════════════════════════
# 8.  VARIABLES DEL MODELO ECONOMÉTRICO
# ══════════════════════════════════════════════════════════════════

# ── Dummies temporales de shocks ──────────────────────────────────
panel["D_COVID"]   = panel["YEAR"].isin([2020, 2021]).astype(int)
panel["D_UKR"]     = (panel["YEAR"] >= 2022).astype(int)

# ── Interacciones post-2022 (H3 amplificada por shock geopolítico) ─
panel["Inflation_post22"]  = panel["Inflation_CPI"]    * panel["D_UKR"]
panel["Energy_post22"]     = panel["Energy_Imports"]   * panel["D_UKR"]
panel["GovDebt_post22"]    = panel["Gov_debt_GDP"]     * panel["D_UKR"]

# ── Interacción H4: inflación × calidad institucional ─────────────
panel["Inflation_x_RoL"]   = panel["Inflation_CPI"]   * panel["Rule_of_Law"]

# ── Rezago de la dependiente (modelo dinámico, paso 3) ────────────
panel["L1_INSTABILITY_INDEX"] = (
    panel.groupby("COUNTRY")["INSTABILITY_INDEX"].shift(1)
)

# ── Log-transformaciones ──────────────────────────────────────────
panel["log_GDP_pc"]      = np.log(panel["GDP_pc_const2015"].clip(lower=1))
panel["log_EVENTS_TOTAL"]= np.log1p(panel["EVENTS_TOTAL"])

# ══════════════════════════════════════════════════════════════════
# 9.  DIAGNÓSTICO FINAL
# ══════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("PANEL FINAL v4 — DIAGNÓSTICO")
print("═" * 60)
print(f"Shape          : {panel.shape[0]} filas × {panel.shape[1]} columnas")
print(f"Países         : {panel['COUNTRY'].nunique()}")
print(f"Años           : {sorted(panel['YEAR'].unique())}")
print(f"\nPaíses incluidos:\n  {sorted(panel['COUNTRY'].unique())}")

print("\n--- Dependiente principal (INSTABILITY_INDEX) ---")
print(panel["INSTABILITY_INDEX"].describe().round(3))
print(f"Missing: {panel['INSTABILITY_INDEX'].isna().sum()}")

print("\n--- Dependiente alternativa (INSTABILITY_BINARY) ---")
print(panel["INSTABILITY_BINARY"].value_counts(dropna=False))

print("\n--- Missing L1_INSTABILITY_INDEX ---")
print(f"{panel['L1_INSTABILITY_INDEX'].isna().sum()} NaN "
      f"(primer año por país — estructural)")

print("\n--- Missings por variable ---")
# Solo las variables que tienen 
miss = panel.isna().sum().sort_values(ascending=False)
print(miss[miss > 0].to_string())

print("\n--- Muestra Austria ---")
cols_muestra = [
    "COUNTRY","YEAR","INSTABILITY_INDEX","INSTABILITY_BINARY",
    "Inflation_CPI","GDP_Growth","Rule_of_Law","VolPS",
    "D_COVID","D_UKR","L1_INSTABILITY_INDEX"
]
print(panel[panel["COUNTRY"] == "Austria"][cols_muestra].to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# 10.  EXPORT
#      sep=";"  decimal="."  → evita el problema de coma decimal
#      que corrompía los floats en versiones anteriores
# ══════════════════════════════════════════════════════════════════
panel.to_excel(OUTFILE_XLSX, index=False)
panel.to_csv(OUTFILE_CSV, index=False, sep=";", decimal=".", encoding="utf-8-sig")

print("Paneles exportados: ")
print(f"   {OUTFILE_XLSX}")
print(f"   {OUTFILE_CSV}")