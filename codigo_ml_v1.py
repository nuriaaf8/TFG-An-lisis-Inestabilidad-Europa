# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:47:47 2026

@author: USUARIO
"""

# -*- coding: utf-8 -*-
"""
codigo_ml_v1.py
============
Script de Machine Learning v1.Versión exploratoria

Contenido:
  1. Preparación de features y targets
  2. REGRESIÓN: predecir INSTABILITY_INDEX (continuo)
     - Baseline: OLS
     - Random Forest Regressor
     - XGBoost Regressor
     - Métricas: RMSE, MAE, R²
  3. CLASIFICACIÓN: predecir INSTABILITY_BINARY (alto/bajo)
     - Baseline: Logit
     - Random Forest Classifier
     - XGBoost Classifier
     - Métricas: AUC, Accuracy, Precision, Recall, F1
  4. Comparación de modelos (tabla resumen)
  5. Importancia de variables (RF + XGBoost)
  6. SHAP values (interpretabilidad)
  7. Partial Dependence Plots (variables clave)

Validación: Leave-One-Year-Out cross-validation
  → más honesta que random split en datos de panel
  → evita data leakage temporal

Librerías necesarias:
  pip install xgboost shap scikit-learn

Autor: Nuria Adame Fuentes — TFG Business Analytics, UAM 2025/26
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 150

# ══════════════════════════════════════════════════════════════════
# 0.  RUTA  
# ══════════════════════════════════════════════════════════════════
FILE_PANEL = r"C:\Users\USUARIO\Desktop\4º\TFG--\data\data2\CODIGOS BUENOS\Outputs Codigos\panel_final_anual_v4.xlsx"

# ══════════════════════════════════════════════════════════════════
# 1.  CARGA Y PREPARACIÓN
# ══════════════════════════════════════════════════════════════════
panel = pd.read_excel(FILE_PANEL)
print(f"Panel cargado → {panel.shape[0]} obs | {panel['COUNTRY'].nunique()} países")

# Features: todas las variables disponibles sin missings excesivos
# Se usan MÁS variables que en el FE — el ML puede manejar más inputs
FEATURES = [
    # Macro core
    "Inflation_CPI", "GDP_Growth", "Unemployment", "Gov_debt_GDP",
    "Energy_Imports",
    # Institucional
    "Rule_of_Law", 
    #"Control_of_Corruption", "Government_Effectiveness",
    "Political_Stability",
    # Proxy polarización
    "VolPS",
    # Shocks temporales
    "D_COVID", "D_UKR",
    # Interacciones
    "Inflation_post22", "Energy_post22", "Inflation_x_RoL",
    # Rezago dependiente
    #"L1_INSTABILITY_INDEX",
    # TGE adicionales
    "Military_spending_GDP", "Econ_freedom", "Property_rights",
]

TARGET_REG  = "INSTABILITY_INDEX"    # regresión
TARGET_CLF  = "INSTABILITY_BINARY"   # clasificación

# Recalcular variables si no están
panel["D_COVID"]          = panel["YEAR"].isin([2020, 2021]).astype(int)
panel["D_UKR"]            = (panel["YEAR"] >= 2022).astype(int)
panel["Inflation_post22"] = panel["Inflation_CPI"] * panel["D_UKR"]
panel["Energy_post22"]    = panel["Energy_Imports"] * panel["D_UKR"]
panel["Inflation_x_RoL"]  = panel["Inflation_CPI"] * panel["Rule_of_Law"]
panel["L1_INSTABILITY_INDEX"] = panel.groupby("COUNTRY")["INSTABILITY_INDEX"].shift(1)

# Quedarse con features disponibles en el panel
FEATURES = [f for f in FEATURES if f in panel.columns]

# Submuestra: solo filas sin NaN en features ni targets
cols_needed = ["COUNTRY", "YEAR"] + FEATURES + [TARGET_REG, TARGET_CLF]
df = panel[cols_needed].dropna().copy()

print(f"Muestra ML → {len(df)} obs | {df['COUNTRY'].nunique()} países | "
      f"años: {sorted(df['YEAR'].unique())}")
print(f"Features utilizados ({len(FEATURES)}): {FEATURES}")
print(f"\nDistribución target clasificación:")
print(df[TARGET_CLF].value_counts())

X = df[FEATURES].values
y_reg = df[TARGET_REG].values
y_clf = df[TARGET_CLF].values
years = df["YEAR"].values

# ══════════════════════════════════════════════════════════════════
# 2.  VALIDACIÓN: LEAVE-ONE-YEAR-OUT (LOYO)
#
#     Para cada año de test: entrena en el resto, predice en ese año.
#     Más rigurosa que random split en panel — evita leakage temporal.
#     Con T=5 años → 5 folds.
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  VALIDACIÓN: LEAVE-ONE-YEAR-OUT (LOYO)")
print("═"*65)

years_unique = sorted(np.unique(years))

# Contenedores de resultados
results_reg = {m: {"rmse":[], "mae":[], "r2":[]}
               for m in ["OLS", "RF_reg", "XGB_reg"]}
results_clf = {m: {"auc":[], "acc":[], "prec":[], "rec":[], "f1":[]}
               for m in ["Logit", "RF_clf", "XGB_clf"]}

for test_year in years_unique:
    train_mask = years != test_year
    test_mask  = years == test_year

    X_train, X_test = X[train_mask], X[test_mask]
    yr_train, yr_test = y_reg[train_mask], y_reg[test_mask]
    yc_train, yc_test = y_clf[train_mask], y_clf[test_mask]

    # Escalar (necesario para OLS/Logit, indiferente para árboles)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── REGRESIÓN ────────────────────────────────────────────────
    # OLS baseline
    ols = LinearRegression().fit(X_train_sc, yr_train)
    yp  = ols.predict(X_test_sc)
    results_reg["OLS"]["rmse"].append(np.sqrt(mean_squared_error(yr_test, yp)))
    results_reg["OLS"]["mae"].append(mean_absolute_error(yr_test, yp))
    results_reg["OLS"]["r2"].append(r2_score(yr_test, yp))

    # Random Forest
    rf_r = RandomForestRegressor(n_estimators=300, max_depth=4,
                                  min_samples_leaf=3, random_state=42)
    rf_r.fit(X_train, yr_train)
    yp = rf_r.predict(X_test)
    results_reg["RF_reg"]["rmse"].append(np.sqrt(mean_squared_error(yr_test, yp)))
    results_reg["RF_reg"]["mae"].append(mean_absolute_error(yr_test, yp))
    results_reg["RF_reg"]["r2"].append(r2_score(yr_test, yp))

    # XGBoost
    xgb_r = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbosity=0)
    xgb_r.fit(X_train, yr_train)
    yp = xgb_r.predict(X_test)
    results_reg["XGB_reg"]["rmse"].append(np.sqrt(mean_squared_error(yr_test, yp)))
    results_reg["XGB_reg"]["mae"].append(mean_absolute_error(yr_test, yp))
    results_reg["XGB_reg"]["r2"].append(r2_score(yr_test, yp))

    # ── CLASIFICACIÓN ────────────────────────────────────────────
    # Solo si hay ambas clases en train y test
    if len(np.unique(yc_train)) < 2 or len(np.unique(yc_test)) < 2:
        continue

    # Logit baseline
    logit = LogisticRegression(max_iter=1000, random_state=42, class_weight = "balanced")
    logit.fit(X_train_sc, yc_train)
    yp_prob = logit.predict_proba(X_test_sc)[:, 1]
    yp_bin  = logit.predict(X_test_sc)
    results_clf["Logit"]["auc"].append(roc_auc_score(yc_test, yp_prob))
    results_clf["Logit"]["acc"].append(accuracy_score(yc_test, yp_bin))
    results_clf["Logit"]["prec"].append(precision_score(yc_test, yp_bin, zero_division=0))
    results_clf["Logit"]["rec"].append(recall_score(yc_test, yp_bin, zero_division=0))
    results_clf["Logit"]["f1"].append(f1_score(yc_test, yp_bin, zero_division=0))

    # Random Forest
    rf_c = RandomForestClassifier(n_estimators=300, max_depth=6,
                                   min_samples_leaf=3,class_weight= "balanced", random_state=42)
    rf_c.fit(X_train, yc_train)
    yp_prob = rf_c.predict_proba(X_test)[:, 1]
    yp_bin  = rf_c.predict(X_test)
    results_clf["RF_clf"]["auc"].append(roc_auc_score(yc_test, yp_prob))
    results_clf["RF_clf"]["acc"].append(accuracy_score(yc_test, yp_bin))
    results_clf["RF_clf"]["prec"].append(precision_score(yc_test, yp_bin, zero_division=0))
    results_clf["RF_clf"]["rec"].append(recall_score(yc_test, yp_bin, zero_division=0))
    results_clf["RF_clf"]["f1"].append(f1_score(yc_test, yp_bin, zero_division=0))

    # XGBoost
    xgb_c = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                use_label_encoder=False, eval_metric="logloss",
                                random_state=42, verbosity=0)
    xgb_c.fit(X_train, yc_train)
    yp_prob = xgb_c.predict_proba(X_test)[:, 1]
    yp_bin  = xgb_c.predict(X_test)
    results_clf["XGB_clf"]["auc"].append(roc_auc_score(yc_test, yp_prob))
    results_clf["XGB_clf"]["acc"].append(accuracy_score(yc_test, yp_bin))
    results_clf["XGB_clf"]["prec"].append(precision_score(yc_test, yp_bin, zero_division=0))
    results_clf["XGB_clf"]["rec"].append(recall_score(yc_test, yp_bin, zero_division=0))
    results_clf["XGB_clf"]["f1"].append(f1_score(yc_test, yp_bin, zero_division=0))

# ══════════════════════════════════════════════════════════════════
# 3.  TABLA DE RESULTADOS
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  REGRESIÓN — métricas LOYO (media sobre folds)")
print("═"*65)
print(f"  {'Modelo':<15} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("  " + "-"*42)
for modelo, vals in results_reg.items():
    print(f"  {modelo:<15} "
          f"{np.mean(vals['rmse']):>8.4f} "
          f"{np.mean(vals['mae']):>8.4f} "
          f"{np.mean(vals['r2']):>8.4f}")

print("\n" + "═"*65)
print("  CLASIFICACIÓN — métricas LOYO (media sobre folds)")
print("═"*65)
print(f"  {'Modelo':<15} {'AUC':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("  " + "-"*50)
for modelo, vals in results_clf.items():
    if vals["auc"]:
        print(f"  {modelo:<15} "
              f"{np.mean(vals['auc']):>7.4f} "
              f"{np.mean(vals['acc']):>7.4f} "
              f"{np.mean(vals['prec']):>7.4f} "
              f"{np.mean(vals['rec']):>7.4f} "
              f"{np.mean(vals['f1']):>7.4f}")

# ══════════════════════════════════════════════════════════════════
# 4.  REENTRENAR EN MUESTRA COMPLETA (para SHAP e importancia)
# ══════════════════════════════════════════════════════════════════
scaler_full = StandardScaler()
X_sc_full   = scaler_full.fit_transform(X)

# Regresión
rf_r_full  = RandomForestRegressor(n_estimators=300, max_depth=4,
                                    min_samples_leaf=3, random_state=42)
rf_r_full.fit(X, y_reg)

xgb_r_full = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, verbosity=0)
xgb_r_full.fit(X, y_reg)

# Clasificación
rf_c_full  = RandomForestClassifier(n_estimators=300, max_depth=4,
                                     min_samples_leaf=3, random_state=42)
rf_c_full.fit(X, y_clf)

xgb_c_full = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 use_label_encoder=False, eval_metric="logloss",
                                 random_state=42, verbosity=0)
xgb_c_full.fit(X, y_clf)

# ══════════════════════════════════════════════════════════════════
# 5.  IMPORTANCIA DE VARIABLES
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  IMPORTANCIA DE VARIABLES — Random Forest (regresión)")
print("═"*65)
imp_rf = pd.Series(rf_r_full.feature_importances_, index=FEATURES).sort_values(ascending=False)
for var, imp in imp_rf.items():
    bar = "█" * int(imp * 100)
    print(f"  {var:<30} {imp:.4f}  {bar}")

print("\n" + "═"*65)
print("  IMPORTANCIA DE VARIABLES — XGBoost (regresión)")
print("═"*65)
imp_xgb = pd.Series(xgb_r_full.feature_importances_, index=FEATURES).sort_values(ascending=False)
for var, imp in imp_xgb.items():
    bar = "█" * int(imp * 100)
    print(f"  {var:<30} {imp:.4f}  {bar}")

# Gráfico comparativo importancia
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
imp_rf.head(12).sort_values().plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("Random Forest\nImportancia de variables (regresión)")
axes[0].set_xlabel("Importancia")

imp_xgb.head(12).sort_values().plot(kind="barh", ax=axes[1], color="darkorange")
axes[1].set_title("XGBoost\nImportancia de variables (regresión)")
axes[1].set_xlabel("Importancia")

plt.tight_layout()
plt.savefig("importancia_variables.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → Gráfico guardado: importancia_variables.png")

# ══════════════════════════════════════════════════════════════════
# 6.  SHAP VALUES — interpretabilidad XGBoost (regresión)
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  SHAP VALUES — XGBoost regresión")
print("═"*65)

explainer   = shap.TreeExplainer(xgb_r_full)
shap_values = explainer.shap_values(X)

# Summary plot (beeswarm)
plt.figure()
shap.summary_plot(shap_values, X, feature_names=FEATURES,
                  show=False, plot_size=(10, 6))
plt.tight_layout()
plt.savefig("shap_summary_xgb_reg.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Gráfico SHAP summary guardado: shap_summary_xgb_reg.png")

# Bar plot SHAP (importancia media absoluta)
plt.figure()
shap.summary_plot(shap_values, X, feature_names=FEATURES,
                  plot_type="bar", show=False, plot_size=(10, 6))
plt.tight_layout()
plt.savefig("shap_bar_xgb_reg.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Gráfico SHAP bar guardado: shap_bar_xgb_reg.png")

# Top 5 variables por SHAP
shap_mean = pd.Series(np.abs(shap_values).mean(axis=0),
                      index=FEATURES).sort_values(ascending=False)
print("\n  Top 10 variables por SHAP (|valor medio|):")
for var, val in shap_mean.head(10).items():
    print(f"    {var:<30} {val:.4f}")

# ══════════════════════════════════════════════════════════════════
# 7.  SHAP — CLASIFICACIÓN (XGBoost)
# ══════════════════════════════════════════════════════════════════
explainer_clf   = shap.TreeExplainer(xgb_c_full)
shap_values_clf = explainer_clf.shap_values(X)

plt.figure()
shap.summary_plot(shap_values_clf, X, feature_names=FEATURES,
                  show=False, plot_size=(10, 6))
plt.tight_layout()
plt.savefig("shap_summary_xgb_clf.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → Gráfico SHAP clasificación guardado: shap_summary_xgb_clf.png")

# ══════════════════════════════════════════════════════════════════
# 8.  GUARDAR RESULTADOS EN EXCEL
# ══════════════════════════════════════════════════════════════════
with pd.ExcelWriter("resultados_ml.xlsx") as writer:

    # Regresión
    reg_rows = []
    for modelo, vals in results_reg.items():
        reg_rows.append({
            "Modelo": modelo,
            "RMSE":   round(np.mean(vals["rmse"]), 4),
            "MAE":    round(np.mean(vals["mae"]),  4),
            "R²":     round(np.mean(vals["r2"]),   4),
        })
    pd.DataFrame(reg_rows).to_excel(writer, sheet_name="Regresion", index=False)

    # Clasificación
    clf_rows = []
    for modelo, vals in results_clf.items():
        if vals["auc"]:
            clf_rows.append({
                "Modelo":    modelo,
                "AUC":       round(np.mean(vals["auc"]),  4),
                "Accuracy":  round(np.mean(vals["acc"]),  4),
                "Precision": round(np.mean(vals["prec"]), 4),
                "Recall":    round(np.mean(vals["rec"]),  4),
                "F1":        round(np.mean(vals["f1"]),   4),
            })
    pd.DataFrame(clf_rows).to_excel(writer, sheet_name="Clasificacion", index=False)

    # Importancia RF
    imp_rf.reset_index().rename(columns={"index": "Variable",
                                          0: "Importancia_RF"}).to_excel(
        writer, sheet_name="Importancia_RF", index=False)

    # Importancia XGB
    imp_xgb.reset_index().rename(columns={"index": "Variable",
                                            0: "Importancia_XGB"}).to_excel(
        writer, sheet_name="Importancia_XGB", index=False)

    # SHAP medias
    shap_mean.reset_index().rename(columns={"index": "Variable",
                                             0: "SHAP_mean_abs"}).to_excel(
        writer, sheet_name="SHAP_XGB_reg", index=False)

print("\n Resultados ML guardados: resultados_ml.xlsx")
print("Gráficos: importancia_variables.png | shap_summary_xgb_reg.png | "
      "shap_bar_xgb_reg.png | shap_summary_xgb_clf.png")
