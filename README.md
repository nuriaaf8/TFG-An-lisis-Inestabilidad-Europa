### Análisis Espacial de la Inestabilidad Política y Económica en Europa

Este repositorio contiene el código desarrollado para el Trabajo de Fin de Grado (TFG):

“Análisis Cuantitativo de la Difusión Espacial de la Inestabilidad Política y Económica en Europa”  
mediante técnicas de econometría espacial y machine learning.

---

## Descripción del proyecto

Este trabajo analiza la difusión espacial de la inestabilidad política entre países europeos, integrando distintas fuentes de datos:

- Datos de inestabilidad política (ACLED)
- Indicadores macroeconómicos (World Bank)
- Calidad institucional (WGI)
- Variables estructurales (The Global Economy)

Los objetivos principales son:

- Identificar factores asociados a la inestabilidad  
- Analizar la existencia de spillovers geográficos  
- Evaluar la capacidad predictiva como indicador de riesgo país  

---

## Metodología

El análisis combina tres enfoques complementarios:

### Modelos econométricos
- Modelos de efectos fijos (FE)
- Modelos dinámicos
- Especificaciones con interacciones (post-2022)

### Econometría espacial
- Moran’s I (global y local)
- Modelo SAR (Spatial Autoregressive)
- Modelo SDM (Spatial Durbin Model)

### Machine Learning
- Random Forest
- XGBoost
- Validación: Leave-One-Year-Out (LOYO)

---

## Estructura del repositorio
├── data/
│ ├── raw/
│ ├── processed/
│
├── scripts/
│ ├── build_panel_v4.py
│ ├── codigo_modelos_ec.py
│ ├── codigo_modelo_espacial.py
│ ├── codigo_ml_v1.py
│ ├── codigo_ml_v2.py
│ ├── codigo_seleccion_var.py
│
├── results/
│ ├── tablas/
│ ├── figuras/
│
└── README.md


## Resultados principales

- La calidad institucional (Rule of Law) se asocia de forma consistente con menores niveles de inestabilidad.  
- Existen spillovers espaciales, aunque de intensidad moderada y no persistente.  
- Las variables macroeconómicas muestran baja significatividad en modelos FE, debido a la limitada variación temporal.  
- Los modelos de machine learning mejoran la capacidad predictiva en clasificación (AUC ≈ 0,73), aunque su carácter es exploratorio.  

---

## Limitaciones

- Panel corto (T = 5)  
- Posible endogeneidad  
- Estructura espacial simplificada (contigüidad)  
- Tamaño muestral reducido en ML  
- Los resultados deben interpretarse como asociaciones, no como relaciones causales  

---

## Requisitos

Python 3.x

Principales librerías:
- pandas
- numpy
- matplotlib
- seaborn
- linearmodels
- scikit-learn
- xgboost
- pysal

Instalación:

```bash
pip install -r requirements.txt

python codigo_modelos_ec.py
python codigo_modelo_espacial.py
python codigo_ml_v1.py
python codigo_ml_v2.py


