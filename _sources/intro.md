# Avazu CTR — Proyecto Integrador de Deep Learning

<div style="background: linear-gradient(135deg, #0c1445 0%, #1e40af 100%); color:#f8fafc; padding:26px 30px; border-radius:18px; margin-bottom:14px; border:1px solid rgba(148,163,184,0.18); box-shadow:0 18px 40px rgba(0,0,0,0.30);">
  <h1 style="margin: 0; font-size: 2.1rem;">Avazu CTR — Proyecto Integrador de Deep Learning</h1>
  <p style="margin: 10px 0 0 0; font-size: 1.02rem; line-height: 1.6;">Clasificacion binaria de Click-Through Rate sobre ~40 millones de impresiones publicitarias moviles, utilizando redes neuronales multicapa (MLP) con scikit-learn y PySpark, e interpretabilidad con LIME.</p>
</div>

<div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:18px;">
  <span style="background-color:#111827; color:#93c5fd; border:1px solid #3b82f6; padding:4px 10px; border-radius:999px; font-weight:600;">Python</span>
  <span style="background-color:#111827; color:#93c5fd; border:1px solid #3b82f6; padding:4px 10px; border-radius:999px; font-weight:600;">scikit-learn</span>
  <span style="background-color:#111827; color:#93c5fd; border:1px solid #3b82f6; padding:4px 10px; border-radius:999px; font-weight:600;">PySpark</span>
  <span style="background-color:#111827; color:#93c5fd; border:1px solid #3b82f6; padding:4px 10px; border-radius:999px; font-weight:600;">LIME</span>
  <span style="background-color:#111827; color:#93c5fd; border:1px solid #3b82f6; padding:4px 10px; border-radius:999px; font-weight:600;">Jupyter Book</span>
</div>

## Objetivo

<div style="background-color:#1f2937; border-left:5px solid #3b82f6; color:#e5e7eb; padding:14px 18px; border-radius:12px; box-shadow:0 10px 25px rgba(0,0,0,0.18);">

Predecir si un usuario hara clic (`click = 1`) o no (`click = 0`) en un anuncio movil, a partir de variables categoricas del dataset Avazu. Se entrenan y evaluan modelos MLP en dos frameworks distintos, y se aplica LIME para explicar las predicciones a nivel individual.

</div>

## Contenido del libro

<div style="background-color:#1f2937; border-left:5px solid #3b82f6; color:#e5e7eb; padding:14px 18px; border-radius:12px; box-shadow:0 10px 25px rgba(0,0,0,0.18);">

1. **Exploracion de Datos (EDA)** — Analisis descriptivo, distribucion de clases, visualizacion de variables categoricas y correlaciones sobre el dataset Avazu completo.
2. **Pipeline scikit-learn** — Entrenamiento de un MLPClassifier con GridSearchCV sobre un millon de registros, incluyendo preprocesamiento, evaluacion y curvas de aprendizaje.
3. **Pipeline PySpark** — Entrenamiento distribuido de un MultilayerPerceptronClassifier sobre los ~40 millones de registros completos del dataset.
4. **LIME y Comparacion de Frameworks** — Explicabilidad local con LIME sobre ambos modelos y comparacion cuantitativa de metricas entre scikit-learn y PySpark.

</div>

## Tecnologias

<div style="background-color:#1f2937; border-left:5px solid #3b82f6; color:#e5e7eb; padding:14px 18px; border-radius:12px; box-shadow:0 10px 25px rgba(0,0,0,0.18);">

- **scikit-learn** — MLPClassifier con GridSearchCV para busqueda de hiperparametros.
- **PySpark MLlib** — MultilayerPerceptronClassifier para entrenamiento distribuido a gran escala.
- **LIME** — Local Interpretable Model-agnostic Explanations para interpretabilidad de predicciones.
- **Jupyter Book** — Generacion del sitio de documentacion estatico a partir de notebooks.

</div>

## Autora

<div style="background-color:#1f2937; border-left:5px solid #3b82f6; color:#e5e7eb; padding:14px 18px; border-radius:12px; box-shadow:0 10px 25px rgba(0,0,0,0.18);">

**Juana** — Proyecto integrador para el curso de Deep Learning.

</div>
