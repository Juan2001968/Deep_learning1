# Avazu CTR Deep Learning Mini-Project

Proyecto de clasificación binaria de Click-Through Rate (CTR) sobre el dataset Avazu (~40M registros de impresiones publicitarias móviles), implementado con **scikit-learn** y **PySpark**, con explicabilidad local mediante **LIME**.

## Objetivo

Predecir si un usuario hará clic en un anuncio móvil (`click = 1`) o no (`click = 0`) utilizando redes neuronales multicapa (MLP) y comparar dos entornos de ejecución:

- **scikit-learn**: pipeline local sobre una muestra estratificada de 1M de registros
- **PySpark**: pipeline distribuido sobre el dataset completo (~40M registros)

## Estructura del Repositorio

```
DEEP_LEARNING1-MAIN/
├── book/
│   ├── _config.yml               # Configuración de Jupyter Book
│   ├── _toc.yml                  # Tabla de contenidos
│   ├── intro.md                  # Página de introducción
│   └── docs/
│       ├── 01_project_overview_and_eda.ipynb    # EDA completo
│       ├── 02_sklearn_mlp_pipeline.ipynb        # Pipeline scikit-learn
│       ├── 03_pyspark_mlp_pipeline.ipynb        # Pipeline PySpark
│       └── 04_lime_and_framework_comparison.ipynb # LIME + comparación
├── configs/
│   └── project_settings.toml     # Configuración centralizada
├── data/
│   ├── processed/                # Muestras procesadas (Parquet)
│   └── raw/                      # Dataset comprimido (train.gz)
├── models/                       # Modelos entrenados guardados
├── reports/figures/               # Figuras generadas (PNG)
├── src/ctr_mlp/                  # Módulos Python reutilizables
│   ├── config.py                 # Rutas, colores, estilo visual
│   ├── data_io.py                # Carga y muestreo de datos
│   ├── eda.py                    # Visualizaciones EDA
│   ├── evaluation.py             # Métricas y gráficos de evaluación
│   ├── explainability.py         # Funciones LIME
│   ├── feature_engineering.py    # Ingeniería de features temporales
│   ├── sklearn_workflow.py       # Pipeline scikit-learn completo
│   ├── spark_workflow.py         # Pipeline PySpark completo
│   └── utils.py                  # Utilidades auxiliares
├── pyproject.toml
└── README.md
```

## Notebooks

| Notebook | Contenido |
|----------|-----------|
| **01 — EDA** | Inspección del dataset, valores faltantes, distribución del target, cardinalidad, feature engineering temporal, 15+ visualizaciones con estilo oscuro profesional |
| **02 — scikit-learn** | Muestreo 1M registros, Target Encoding + OneHot, MLPClassifier con GridSearchCV (18 combinaciones), métricas completas, curva ROC, matriz de confusión |
| **03 — PySpark** | Dataset completo, StringIndexer + OneHotEncoder + VectorAssembler + StandardScaler, MultilayerPerceptronClassifier (12 configuraciones) |
| **04 — LIME + Comparación** | Explicaciones locales de instancias mal clasificadas (FP/FN), tabla comparativa, curvas ROC superpuestas, comparación de tiempos, reflexión crítica |

## Dataset

El dataset se espera en `data/raw/train.gz` y se lee directamente en formato comprimido. No se requiere descompresión manual.

El dataset está excluido del control de versiones por su tamaño.

## Stack Tecnológico

- Python 3.11+
- Pandas / NumPy
- Matplotlib / Seaborn (tema oscuro profesional)
- scikit-learn (MLPClassifier, GridSearchCV, TargetEncoder)
- PySpark (MultilayerPerceptronClassifier)
- LIME (explicabilidad local)
- Jupyter Book

## Instalación

Activar el entorno `deep_env` e instalar el paquete:

```bash
pip install -e .
```

## Estilo Visual

Todas las gráficas del proyecto usan un tema oscuro profesional con paleta de colores vibrantes (Indigo, Violeta, Rosa, Esmeralda) definido en `src/ctr_mlp/config.py`.
