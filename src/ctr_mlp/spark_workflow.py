"""Pipeline completo de PySpark para clasificación CTR con MultilayerPerceptronClassifier.

Incluye creación de sesión Spark, lectura del dataset comprimido,
preprocesamiento distribuido, entrenamiento con búsqueda de hiperparámetros
y guardado de modelos.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd


def create_spark_session(
    app_name: str = "avazu-ctr-mlp",
    shuffle_partitions: int = 100,
    driver_memory: str = "8g",
    executor_memory: str = "4g",
):
    """Crea y configura una sesión de Spark optimizada para el proyecto.

    Configura memoria suficiente para procesar el dataset Avazu completo
    (~40M registros) en modo local, con timeouts extendidos para evitar
    que la JVM se desconecte durante operaciones largas.

    Parámetros
    ----------
    app_name : str
        Nombre de la aplicación Spark.
    shuffle_partitions : int
        Número de particiones para operaciones de shuffle.
    driver_memory : str
        Memoria asignada al driver (ej. ``'8g'``).
    executor_memory : str
        Memoria asignada a cada executor (ej. ``'4g'``).

    Retorna
    -------
    pyspark.sql.SparkSession
        Sesión de Spark configurada con memoria y timeouts adecuados.
    """
    from pyspark.sql import SparkSession

    # Asegurar que PySpark use el mismo intérprete Python que el notebook
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # Cerrar sesión zombie previa si existe
    try:
        existing = SparkSession.getActiveSession()
        if existing is not None:
            existing.stop()
    except Exception:
        pass

    return (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "2g")
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.autoBroadcastJoinThreshold", "50m")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "120s")
        .config("spark.py4j.gateway.timeout", "300")
        .getOrCreate()
    )


def read_ctr_csv_spark(spark, csv_path, infer_schema: bool = False, repartition: int | None = None):
    """Lee el archivo CSV comprimido de Avazu directamente con Spark.

    Spark maneja la descompresión de ``.gz`` de forma transparente
    sin necesidad de descompresión manual previa.

    Parámetros
    ----------
    spark : pyspark.sql.SparkSession
        Sesión activa de Spark.
    csv_path : str | Path
        Ruta al archivo CSV comprimido.
    infer_schema : bool
        Si True, Spark infiere los tipos automáticamente.
    repartition : int | None
        Número de particiones objetivo tras la lectura.

    Retorna
    -------
    pyspark.sql.DataFrame
        DataFrame de Spark con los datos cargados.
    """
    df = (
        spark.read.option("header", True)
        .option("inferSchema", infer_schema)
        .csv(str(csv_path))
    )
    if repartition:
        df = df.repartition(repartition)
    return df


def cast_spark_columns(df, categorical_columns: list[str], numeric_columns: list[str], label_col: str = "click"):
    """Convierte columnas al tipo apropiado para el pipeline de Spark ML.

    Parámetros
    ----------
    df : pyspark.sql.DataFrame
        DataFrame de Spark con columnas sin tipar.
    categorical_columns : list[str]
        Columnas a convertir a string.
    numeric_columns : list[str]
        Columnas a convertir a double.
    label_col : str
        Columna target a convertir a double.

    Retorna
    -------
    pyspark.sql.DataFrame
        DataFrame con las columnas tipadas correctamente.
    """
    from pyspark.sql import functions as F

    casted = df.withColumn(label_col, F.col(label_col).cast("double"))
    for column in categorical_columns:
        casted = casted.withColumn(column, F.col(column).cast("string"))
    for column in numeric_columns:
        casted = casted.withColumn(column, F.col(column).cast("double"))
    return casted


def reduce_cardinality(df, column: str, top_n: int = 100):
    """Mantiene solo los top_n valores más frecuentes; el resto se reemplaza por 'OTHER'.

    Previene la explosión de dimensionalidad en columnas de alta
    cardinalidad (site_id, site_domain, app_id, app_domain, device_model, etc.)
    y reduce drásticamente el tiempo de preprocesamiento en modo local.

    Parámetros
    ----------
    df : pyspark.sql.DataFrame
        DataFrame de Spark.
    column : str
        Nombre de la columna a reducir.
    top_n : int
        Número de valores más frecuentes a conservar.

    Retorna
    -------
    pyspark.sql.DataFrame
        DataFrame con la columna reducida.
    """
    from pyspark.sql import functions as F

    top_values = (
        df.groupBy(column)
        .count()
        .orderBy(F.desc("count"))
        .limit(top_n)
        .select(column)
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    return df.withColumn(
        column,
        F.when(F.col(column).isin(top_values), F.col(column)).otherwise("OTHER"),
    )


def reduce_cardinality_columns(df, columns: list[str], top_n: int = 100):
    """Aplica reduce_cardinality a múltiples columnas.

    Parámetros
    ----------
    df : pyspark.sql.DataFrame
        DataFrame de Spark.
    columns : list[str]
        Columnas a reducir.
    top_n : int
        Número de valores más frecuentes a conservar por columna.

    Retorna
    -------
    pyspark.sql.DataFrame
        DataFrame con todas las columnas reducidas.
    """
    for col in columns:
        df = reduce_cardinality(df, col, top_n=top_n)
    return df


def build_feature_pipeline(
    categorical_columns: list[str],
    numeric_columns: list[str],
    scale_features: bool = True,
):
    """Construye el pipeline de preprocesamiento de Spark ML.

    Usa StringIndexer para convertir categóricas a índices numéricos,
    VectorAssembler para combinar todas las features y opcionalmente
    StandardScaler para normalización. No usa OneHotEncoder para evitar
    la explosión de dimensionalidad con columnas de alta cardinalidad.

    Parámetros
    ----------
    categorical_columns : list[str]
        Columnas categóricas a indexar.
    numeric_columns : list[str]
        Columnas numéricas a ensamblar.
    scale_features : bool
        Si True, aplica StandardScaler al vector ensamblado.

    Retorna
    -------
    tuple
        ``(Pipeline, feature_col_name)`` con el pipeline y el nombre
        de la columna de features resultante.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler

    indexers = [
        StringIndexer(
            inputCol=column,
            outputCol=f"{column}_idx",
            handleInvalid="keep",
            stringOrderType="frequencyDesc",
        )
        for column in categorical_columns
    ]

    assembled_inputs = [f"{column}_idx" for column in categorical_columns] + list(numeric_columns)
    raw_feature_col = "raw_features"
    assembler = VectorAssembler(inputCols=assembled_inputs, outputCol=raw_feature_col, handleInvalid="keep")

    stages = [*indexers, assembler]
    feature_col = raw_feature_col
    if scale_features:
        stages.append(
            StandardScaler(
                inputCol=raw_feature_col,
                outputCol="features",
                withMean=False,
                withStd=True,
            )
        )
        feature_col = "features"

    return Pipeline(stages=stages), feature_col


def prepare_spark_features(
    train_df,
    test_df,
    categorical_columns: list[str],
    numeric_columns: list[str],
    label_col: str = "click",
    scale_features: bool = True,
):
    """Prepara los conjuntos de train y test con el pipeline de features de Spark.

    Parámetros
    ----------
    train_df : pyspark.sql.DataFrame
        DataFrame de entrenamiento.
    test_df : pyspark.sql.DataFrame
        DataFrame de prueba.
    categorical_columns : list[str]
        Columnas categóricas.
    numeric_columns : list[str]
        Columnas numéricas.
    label_col : str
        Nombre de la columna target.
    scale_features : bool
        Si True, aplica StandardScaler.

    Retorna
    -------
    tuple
        ``(feature_model, train_features, test_features, feature_col)``
    """
    # Filtrar columnas categóricas que realmente existen en el DataFrame
    available_cols = set(train_df.columns)
    valid_categorical = [c for c in categorical_columns if c in available_cols]
    valid_numeric = [c for c in numeric_columns if c in available_cols]

    prepared_train = cast_spark_columns(train_df, valid_categorical, valid_numeric, label_col=label_col)
    prepared_test = cast_spark_columns(test_df, valid_categorical, valid_numeric, label_col=label_col)

    # Repartir para evitar que una sola partición acumule demasiados datos
    prepared_train = prepared_train.repartition(100)
    prepared_test = prepared_test.repartition(100)

    pipeline, feature_col = build_feature_pipeline(
        categorical_columns=valid_categorical,
        numeric_columns=valid_numeric,
        scale_features=scale_features,
    )
    feature_model = pipeline.fit(prepared_train)

    from pyspark import StorageLevel

    train_features = feature_model.transform(prepared_train).select(feature_col, label_col).persist(StorageLevel.MEMORY_AND_DISK)
    test_features = feature_model.transform(prepared_test).select(feature_col, label_col).persist(StorageLevel.MEMORY_AND_DISK)

    print(f"Particiones train: {train_features.rdd.getNumPartitions()}")
    print(f"Particiones test: {test_features.rdd.getNumPartitions()}")
    # El count se hará implícitamente cuando el modelo entrene
    return feature_model, train_features, test_features, feature_col


def evaluate_spark_predictions(predictions, label_col: str = "click", prediction_col: str = "prediction") -> dict[str, float]:
    """Evalúa las predicciones de un modelo Spark con métricas de clasificación.

    Parámetros
    ----------
    predictions : pyspark.sql.DataFrame
        DataFrame con las predicciones del modelo.
    label_col : str
        Nombre de la columna target.
    prediction_col : str
        Nombre de la columna de predicciones.

    Retorna
    -------
    dict[str, float]
        Diccionario con accuracy, precision, recall, f1, roc_auc y
        componentes de la matriz de confusión.
    """
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.sql import functions as F

    summary = predictions.select(
        F.sum(F.when((F.col(label_col) == 1.0) & (F.col(prediction_col) == 1.0), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col(label_col) == 0.0) & (F.col(prediction_col) == 0.0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col(label_col) == 0.0) & (F.col(prediction_col) == 1.0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col(label_col) == 1.0) & (F.col(prediction_col) == 0.0), 1).otherwise(0)).alias("fn"),
    ).first()

    tp = float(summary["tp"] or 0.0)
    tn = float(summary["tn"] or 0.0)
    fp = float(summary["fp"] or 0.0)
    fn = float(summary["fn"] or 0.0)
    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    evaluator = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    roc_auc = evaluator.evaluate(predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def run_spark_mlp_search(
    train_features,
    test_features,
    feature_col: str = "features",
    label_col: str = "click",
    hidden_layer_widths: tuple[int, ...] = (10, 50, 100),
    step_sizes: tuple[float, ...] = (0.1, 0.01),
    max_iters: tuple[int, ...] = (100, 200),
    seed: int = 42,
) -> pd.DataFrame:
    """Ejecuta una búsqueda exhaustiva de hiperparámetros para MLP en Spark.

    Entrena y evalúa múltiples combinaciones de arquitectura, tasa de
    aprendizaje e iteraciones máximas del MultilayerPerceptronClassifier.

    Parámetros
    ----------
    train_features : pyspark.sql.DataFrame
        Datos de entrenamiento preparados (features + label).
    test_features : pyspark.sql.DataFrame
        Datos de prueba preparados.
    feature_col : str
        Nombre de la columna de features.
    label_col : str
        Nombre de la columna target.
    hidden_layer_widths : tuple[int, ...]
        Anchos de la capa oculta a probar.
    step_sizes : tuple[float, ...]
        Tasas de aprendizaje a probar.
    max_iters : tuple[int, ...]
        Iteraciones máximas a probar.
    seed : int
        Semilla para reproducibilidad.

    Retorna
    -------
    pd.DataFrame
        Tabla de resultados ordenada por ROC AUC descendente.
    """
    from pyspark.ml.classification import MultilayerPerceptronClassifier

    num_features = train_features.select(feature_col).first()[0].size
    results: list[dict[str, float | int]] = []

    for hidden_width in hidden_layer_widths:
        layers = [num_features, hidden_width, 2]
        for step_size in step_sizes:
            for max_iter in max_iters:
                classifier = MultilayerPerceptronClassifier(
                    featuresCol=feature_col,
                    labelCol=label_col,
                    layers=layers,
                    stepSize=step_size,
                    maxIter=max_iter,
                    seed=seed,
                )

                train_start = perf_counter()
                model = classifier.fit(train_features)
                training_seconds = perf_counter() - train_start

                predict_start = perf_counter()
                predictions = model.transform(test_features).cache()
                predictions.count()
                prediction_seconds = perf_counter() - predict_start

                metrics = evaluate_spark_predictions(predictions, label_col=label_col)
                metrics.update(
                    {
                        "hidden_width": hidden_width,
                        "step_size": step_size,
                        "max_iter": max_iter,
                        "training_seconds": training_seconds,
                        "prediction_seconds": prediction_seconds,
                    }
                )
                results.append(metrics)
                predictions.unpersist()

    return pd.DataFrame(results).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)


def save_spark_model(model, save_path: str | Path, model_name: str = "spark_best_mlp") -> Path:
    """Guarda un modelo de Spark ML en disco.

    Parámetros
    ----------
    model : pyspark.ml.Model
        Modelo entrenado de Spark.
    save_path : str | Path
        Directorio base donde guardar el modelo.
    model_name : str
        Nombre del subdirectorio del modelo.

    Retorna
    -------
    Path
        Ruta completa donde se guardó el modelo.
    """
    path = Path(save_path) / model_name
    model.write().overwrite().save(str(path))
    print(f"Modelo Spark guardado en: {path}")
    return path
