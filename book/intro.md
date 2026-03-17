# Avazu CTR — Proyecto de Deep Learning

Bienvenido a la documentación del proyecto de clasificación de Click-Through Rate (CTR) sobre el dataset Avazu.

## Objetivo del proyecto

Predecir si un usuario hará clic en un anuncio publicitario móvil utilizando redes neuronales multicapa (MLP), comparando dos entornos de ejecución:

- **scikit-learn** sobre una muestra representativa local (1M registros)
- **PySpark** sobre el dataset completo (~40M registros)

## Contenido del libro

1. **Análisis Exploratorio de Datos (EDA)** — Inspección del dataset, distribución del target, cardinalidad de variables, feature engineering temporal y visualizaciones completas.

2. **Pipeline scikit-learn** — Muestreo estratificado, codificación (Target Encoding + OneHot), MLPClassifier con GridSearchCV y evaluación con métricas completas.

3. **Pipeline PySpark** — Procesamiento distribuido del dataset completo con MultilayerPerceptronClassifier y búsqueda de hiperparámetros.

4. **LIME y Comparación** — Explicabilidad local de predicciones erróneas, tabla comparativa de frameworks, curvas ROC superpuestas y reflexión crítica.

## Estilo visual

Todas las visualizaciones usan un tema oscuro profesional con colores vibrantes, diseñado para máxima legibilidad y presentación.
