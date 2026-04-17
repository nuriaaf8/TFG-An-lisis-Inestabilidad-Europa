# -*- coding: utf-8 -*-
"""
codigo_ml_v2.py
===============
Script ML v2.
Variables seleccionadas mediante stepwise-backward combinadas con variables teoricas.

Mejoras respecto a v1:
  - Features reducidos de 19 a 17 (menos ruido)
  - Incluye L1_INSTABILITY_INDEX (rezago dependiente)
  - class_weight=balanced en clasificacion
  - scale_pos_weight en XGBoost clasificacion
  - Comparacion directa v1 vs v2

Autor: Nuria Adame Fuentes - TFG Business Analytics, UAM 2025/26
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score)
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
# 1.  CARGA Y PREPARACION
# ══════════════════════════════════════════════════════════════════
panel = pd.read_excel(FILE_PANEL)
print(f"Panel cargado -> {panel.shape[0]} obs | {panel['COUNTRY'].nunique()} paises")

panel["D_COVID"]          = panel["YEAR"].isin([2020, 2021]).astype(int)
panel["D_UKR"]            = (panel["YEAR"] >= 2022).astype(int)
panel["Inflation_post22"] = panel["Inflation_CPI"] * panel["D_UKR"]
panel["Energy_post22"]    = panel["Energy_Imports"] * panel["D_UKR"]
panel["Inflation_x_RoL"]  = panel["Inflation_CPI"]  * panel["Rule_of_Law"]

# Features organizados por grupo con justificacion formal
FEATURES_STEPWISE  = ["Political_Stability", "Control_of_Corruption"]
FEATURES_TEORICAS  = ["Inflation_CPI", "GDP_Growth", "Unemployment",
                       "Gov_debt_GDP", "Energy_Imports", "Rule_of_Law", "VolPS",
                       "L1_INSTABILITY_INDEX"]  # rezago: señal temporal clave
FEATURES_SHOCKS    = ["D_COVID", "D_UKR"]
FEATURES_INTERACT  = ["Inflation_post22", "Energy_post22", "Inflation_x_RoL"]
FEATURES_TGE       = ["Military_spending_GDP", "Econ_freedom", "Property_rights"]

FEATURES = (FEATURES_STEPWISE + FEATURES_TEORICAS +
            FEATURES_SHOCKS + FEATURES_INTERACT + FEATURES_TGE)
FEATURES = [f for f in FEATURES if f in panel.columns]

TARGET_REG = "INSTABILITY_INDEX"
TARGET_CLF = "INSTABILITY_BINARY"

cols_needed = ["COUNTRY", "YEAR"] + FEATURES + [TARGET_REG, TARGET_CLF]
df = panel[cols_needed].dropna().copy()

print(f"Muestra ML v2 -> {len(df)} obs | {df['COUNTRY'].nunique()} paises | "
      f"anos: {sorted(df['YEAR'].unique())}")
print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"\nDistribucion target clasificacion:")
print(df[TARGET_CLF].value_counts())

X     = df[FEATURES].values
y_reg = df[TARGET_REG].values
y_clf = df[TARGET_CLF].values
years = df["YEAR"].values

# ══════════════════════════════════════════════════════════════════
# 2.  VALIDACION LOYO
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  VALIDACION: LEAVE-ONE-YEAR-OUT (LOYO)")
print("="*65)

years_unique = sorted(np.unique(years))

results_reg = {m: {"rmse":[], "mae":[], "r2":[]}
               for m in ["OLS", "RF_reg", "XGB_reg"]}
results_clf = {m: {"auc":[], "acc":[], "prec":[], "rec":[], "f1":[]}
               for m in ["Logit", "RF_clf", "XGB_clf"]}

for test_year in years_unique:
    train_mask = years != test_year
    test_mask  = years == test_year

    X_train, X_test   = X[train_mask], X[test_mask]
    yr_train, yr_test = y_reg[train_mask], y_reg[test_mask]
    yc_train, yc_test = y_clf[train_mask], y_clf[test_mask]

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Regresion
    ols = LinearRegression().fit(X_train_sc, yr_train)
    yp  = ols.predict(X_test_sc)
    results_reg["OLS"]["rmse"].append(np.sqrt(mean_squared_error(yr_test, yp)))
    results_reg["OLS"]["mae"].append(mean_absolute_error(yr_test, yp))
    results_reg["OLS"]["r2"].append(r2_score(yr_test, yp))

    rf_r = RandomForestRegressor(n_estimators=300, max_depth=4,
                                  min_samples_leaf=3, random_state=42)
    rf_r.fit(X_train, yr_train)
    yp = rf_r.predict(X_test)
    results_reg["RF_reg"]["rmse"].append(np.sqrt(mean_squared_error(yr_test, yp)))
    results_reg["RF_reg"]["mae"].append(mean_absolute_error(yr_test, yp))
    results_reg["RF_reg"]["r2"].append(r2_score(yr_test, yp))

    xgb_r = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbosity=0)
    xgb_r.fit(X_train, yr_train)
    yp = xgb_r.predict(X_test)
    results_reg["XGB_reg"]["rmse"].append(np.sqrt(mean_squared_error(yr_test, yp)))
    results_reg["XGB_reg"]["mae"].append(mean_absolute_error(yr_test, yp))
    results_reg["XGB_reg"]["r2"].append(r2_score(yr_test, yp))

    # Clasificacion
    if len(np.unique(yc_train)) < 2 or len(np.unique(yc_test)) < 2:
        continue

    spm = len(yc_train[yc_train==0]) / max(1, len(yc_train[yc_train==1]))

    logit = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    logit.fit(X_train_sc, yc_train)
    yp_prob = logit.predict_proba(X_test_sc)[:, 1]
    yp_bin  = logit.predict(X_test_sc)
    results_clf["Logit"]["auc"].append(roc_auc_score(yc_test, yp_prob))
    results_clf["Logit"]["acc"].append(accuracy_score(yc_test, yp_bin))
    results_clf["Logit"]["prec"].append(precision_score(yc_test, yp_bin, zero_division=0))
    results_clf["Logit"]["rec"].append(recall_score(yc_test, yp_bin, zero_division=0))
    results_clf["Logit"]["f1"].append(f1_score(yc_test, yp_bin, zero_division=0))

    rf_c = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                   class_weight="balanced", random_state=42)
    rf_c.fit(X_train, yc_train)
    yp_prob = rf_c.predict_proba(X_test)[:, 1]
    yp_bin  = rf_c.predict(X_test)
    results_clf["RF_clf"]["auc"].append(roc_auc_score(yc_test, yp_prob))
    results_clf["RF_clf"]["acc"].append(accuracy_score(yc_test, yp_bin))
    results_clf["RF_clf"]["prec"].append(precision_score(yc_test, yp_bin, zero_division=0))
    results_clf["RF_clf"]["rec"].append(recall_score(yc_test, yp_bin, zero_division=0))
    results_clf["RF_clf"]["f1"].append(f1_score(yc_test, yp_bin, zero_division=0))

    xgb_c = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                scale_pos_weight=spm, eval_metric="logloss",
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
# 3.  RESULTADOS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  REGRESION - metricas LOYO (media sobre folds)")
print("="*65)
print(f"  {'Modelo':<15} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print("  " + "-"*42)
for modelo, vals in results_reg.items():
    print(f"  {modelo:<15} {np.mean(vals['rmse']):>8.4f} "
          f"{np.mean(vals['mae']):>8.4f} {np.mean(vals['r2']):>8.4f}")

print("\n" + "="*65)
print("  CLASIFICACION - metricas LOYO (media sobre folds)")
print("="*65)
print(f"  {'Modelo':<15} {'AUC':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("  " + "-"*50)
for modelo, vals in results_clf.items():
    if vals["auc"]:
        print(f"  {modelo:<15} {np.mean(vals['auc']):>7.4f} "
              f"{np.mean(vals['acc']):>7.4f} {np.mean(vals['prec']):>7.4f} "
              f"{np.mean(vals['rec']):>7.4f} {np.mean(vals['f1']):>7.4f}")

# # ══════════════════════════════════════════════════════════════════
# # 4.  COMPARACION v1 vs v2
# # ══════════════════════════════════════════════════════════════════
# print("\n" + "="*65)
# print("  COMPARACION ML v1 vs v2")
# print("="*65)

# v1_rmse = {"OLS": 1.2753, "RF_reg": 1.497, "XGB_reg": 1.479}
# v1_auc  = {"Logit": 0.4529, "RF_clf": 0.691, "XGB_clf": 0.6928}

# print(f"\n  REGRESION (RMSE - menor es mejor):")
# print(f"  {'Modelo':<15} {'v1':>8} {'v2':>8} {'Mejora?'}")
# print("  " + "-"*40)
# for m, vals in results_reg.items():
#     v2 = np.mean(vals["rmse"])
#     v1 = v1_rmse.get(m, np.nan)
#     print(f"  {m:<15} {v1:>8.4f} {v2:>8.4f}  {'SI' if v2 < v1 else 'NO'}")

# print(f"\n  CLASIFICACION (AUC - mayor es mejor):")
# print(f"  {'Modelo':<15} {'v1':>8} {'v2':>8} {'Mejora?'}")
# print("  " + "-"*40)
# for m, vals in results_clf.items():
#     if vals["auc"]:
#         v2 = np.mean(vals["auc"])
#         v1 = v1_auc.get(m, np.nan)
#         print(f"  {m:<15} {v1:>8.4f} {v2:>8.4f}  {'SI' if v2 > v1 else 'NO'}")

# ══════════════════════════════════════════════════════════════════
# 5.  REENTRENAR EN MUESTRA COMPLETA
# ══════════════════════════════════════════════════════════════════
rf_r_full = RandomForestRegressor(n_estimators=300, max_depth=4,
                                   min_samples_leaf=3, random_state=42)
rf_r_full.fit(X, y_reg)

xgb_r_full = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, verbosity=0)
xgb_r_full.fit(X, y_reg)

spm_full = len(y_clf[y_clf==0]) / max(1, len(y_clf[y_clf==1]))
xgb_c_full = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=spm_full, eval_metric="logloss",
                                 random_state=42, verbosity=0)
xgb_c_full.fit(X, y_clf)

# ══════════════════════════════════════════════════════════════════
# 6.  IMPORTANCIA Y SHAP
# ══════════════════════════════════════════════════════════════════
imp_rf  = pd.Series(rf_r_full.feature_importances_, index=FEATURES).sort_values(ascending=False)
imp_xgb = pd.Series(xgb_r_full.feature_importances_, index=FEATURES).sort_values(ascending=False)

print("\n" + "="*65)
print("  IMPORTANCIA - Random Forest (regresion)")
print("="*65)
for var, imp in imp_rf.items():
    print(f"  {var:<30} {imp:.4f}  {'█' * int(imp*100)}")

print("\n" + "="*65)
print("  IMPORTANCIA - XGBoost (regresion)")
print("="*65)
for var, imp in imp_xgb.items():
    print(f"  {var:<30} {imp:.4f}  {'█' * int(imp*100)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
imp_rf.head(12).sort_values().plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("Random Forest v2\nImportancia de variables (regresion)")
axes[0].set_xlabel("Importancia")
imp_xgb.head(12).sort_values().plot(kind="barh", ax=axes[1], color="darkorange")
axes[1].set_title("XGBoost v2\nImportancia de variables (regresion)")
axes[1].set_xlabel("Importancia")
plt.tight_layout()
plt.savefig("importancia_variables_v2.png", dpi=150, bbox_inches="tight")
plt.close()

explainer   = shap.TreeExplainer(xgb_r_full)
shap_values = explainer.shap_values(X)

plt.figure()
shap.summary_plot(shap_values, X, feature_names=FEATURES, show=False, plot_size=(10,6))
plt.tight_layout()
plt.savefig("shap_summary_v2_reg.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(shap_values, X, feature_names=FEATURES,
                  plot_type="bar", show=False, plot_size=(10,6))
plt.tight_layout()
plt.savefig("shap_bar_v2_reg.png", dpi=150, bbox_inches="tight")
plt.close()

shap_mean = pd.Series(np.abs(shap_values).mean(axis=0),
                      index=FEATURES).sort_values(ascending=False)
print("\n  Top 10 variables por SHAP:")
for var, val in shap_mean.head(10).items():
    print(f"    {var:<30} {val:.4f}")

explainer_clf   = shap.TreeExplainer(xgb_c_full)
shap_values_clf = explainer_clf.shap_values(X)
plt.figure()
shap.summary_plot(shap_values_clf, X, feature_names=FEATURES, show=False, plot_size=(10,6))
plt.tight_layout()
plt.savefig("shap_summary_v2_clf.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nGraficos guardados: importancia_variables_v2.png | shap_summary_v2_reg.png | "
      "shap_bar_v2_reg.png | shap_summary_v2_clf.png")

# ══════════════════════════════════════════════════════════════════
# 7.  EXPORT EXCEL
# ══════════════════════════════════════════════════════════════════
with pd.ExcelWriter("resultados_ml_v2.xlsx") as writer:
    reg_rows = []
    for m, vals in results_reg.items():
        reg_rows.append({"Modelo":m, "RMSE":round(np.mean(vals["rmse"]),4),
                         "MAE":round(np.mean(vals["mae"]),4),
                         "R2":round(np.mean(vals["r2"]),4),
                         "RMSE_v1":v1_rmse.get(m)})
    pd.DataFrame(reg_rows).to_excel(writer, sheet_name="Regresion", index=False)

    clf_rows = []
    for m, vals in results_clf.items():
        if vals["auc"]:
            clf_rows.append({"Modelo":m, "AUC":round(np.mean(vals["auc"]),4),
                             "Accuracy":round(np.mean(vals["acc"]),4),
                             "Precision":round(np.mean(vals["prec"]),4),
                             "Recall":round(np.mean(vals["rec"]),4),
                             "F1":round(np.mean(vals["f1"]),4),
                             "AUC_v1":v1_auc.get(m)})
    pd.DataFrame(clf_rows).to_excel(writer, sheet_name="Clasificacion", index=False)

    imp_rf.reset_index().rename(columns={"index":"Variable",0:"Importancia_RF"}).to_excel(
        writer, sheet_name="Importancia_RF", index=False)
    imp_xgb.reset_index().rename(columns={"index":"Variable",0:"Importancia_XGB"}).to_excel(
        writer, sheet_name="Importancia_XGB", index=False)
    shap_mean.reset_index().rename(columns={"index":"Variable",0:"SHAP_mean_abs"}).to_excel(
        writer, sheet_name="SHAP_XGB_reg", index=False)

print("\nResultados guardados: resultados_ml_v2.xlsx")
print("\n-> Analisis cuantitativo completado.")
