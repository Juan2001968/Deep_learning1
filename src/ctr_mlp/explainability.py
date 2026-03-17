"""Funciones de explicabilidad local con LIME para el proyecto Avazu CTR.

Proporciona herramientas para construir explicadores LIME sobre el pipeline
de scikit-learn, generar explicaciones de predicciones individuales y
encontrar instancias mal clasificadas para análisis detallado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy import sparse


@dataclass(slots=True)
class LimeArtifacts:
    """Contiene el explicador LIME y los nombres de features transformadas.

    Atributos
    ----------
    explainer : LimeTabularExplainer
        Instancia del explicador LIME configurada.
    feature_names : list[str]
        Nombres de las features después de la transformación del preprocesador.
    """
    explainer: LimeTabularExplainer
    feature_names: list[str]


def build_lime_explainer_from_pipeline(
    pipeline,
    X_reference: pd.DataFrame,
    class_names: tuple[str, str] = ("no_click", "click"),
    background_size: int = 2000,
    random_state: int = 42,
) -> LimeArtifacts:
    """Crea un explicador LIME a partir de un pipeline de scikit-learn.

    Utiliza una muestra del conjunto de referencia transformada por el
    preprocesador del pipeline como datos de background para LIME.

    Parámetros
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline entrenado con pasos 'preprocessor' y 'classifier'.
    X_reference : pd.DataFrame
        Datos de referencia (típicamente X_train) para el background.
    class_names : tuple[str, str]
        Nombres de las clases para las explicaciones.
    background_size : int
        Tamaño de la muestra de background.
    random_state : int
        Semilla para reproducibilidad.

    Retorna
    -------
    LimeArtifacts
        Objeto con el explicador LIME y los nombres de features.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    reference = X_reference.sample(
        n=min(len(X_reference), background_size),
        random_state=random_state,
    )
    transformed_reference = preprocessor.transform(reference)
    if sparse.issparse(transformed_reference):
        transformed_reference = transformed_reference.astype(np.float32).toarray()

    feature_names = list(preprocessor.get_feature_names_out())
    explainer = LimeTabularExplainer(
        training_data=np.asarray(transformed_reference),
        feature_names=feature_names,
        class_names=list(class_names),
        mode="classification",
        discretize_continuous=False,
        random_state=random_state,
    )
    return LimeArtifacts(explainer=explainer, feature_names=feature_names)


def explain_pipeline_prediction(
    pipeline,
    lime_artifacts: LimeArtifacts,
    X_instance: pd.DataFrame,
    label: int = 1,
    num_features: int = 10,
):
    """Genera una explicación LIME para una predicción individual del pipeline.

    Parámetros
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline entrenado.
    lime_artifacts : LimeArtifacts
        Artefactos LIME con el explicador configurado.
    X_instance : pd.DataFrame
        DataFrame con exactamente una fila (la instancia a explicar).
    label : int
        Clase para la cual generar la explicación (1 = click).
    num_features : int
        Número de features más importantes a incluir.

    Retorna
    -------
    lime.explanation.Explanation
        Objeto de explicación LIME con pesos y visualizaciones.

    Raises
    ------
    ValueError
        Si ``X_instance`` no contiene exactamente una fila.
    """
    if len(X_instance) != 1:
        raise ValueError("X_instance debe contener exactamente una fila.")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed_instance = preprocessor.transform(X_instance)
    if sparse.issparse(transformed_instance):
        transformed_instance = transformed_instance.astype(np.float32).toarray()

    return lime_artifacts.explainer.explain_instance(
        data_row=np.asarray(transformed_instance)[0],
        predict_fn=classifier.predict_proba,
        labels=(label,),
        num_features=num_features,
    )


def explanation_to_frame(explanation, label: int = 1) -> pd.DataFrame:
    """Convierte una explicación LIME a un DataFrame con features y pesos.

    Parámetros
    ----------
    explanation : lime.explanation.Explanation
        Explicación generada por LIME.
    label : int
        Clase de la explicación.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas 'feature' y 'weight', ordenado por peso absoluto.
    """
    return pd.DataFrame(explanation.as_list(label=label), columns=["feature", "weight"])


def find_misclassified_instances(
    y_true,
    y_pred,
    y_score=None,
    error_type: str = "both",
    n: int = 5,
) -> dict[str, np.ndarray]:
    """Encuentra instancias mal clasificadas para análisis con LIME.

    Identifica falsos positivos y/o falsos negativos, opcionalmente
    ordenados por la confianza (probabilidad) de la predicción errónea.

    Parámetros
    ----------
    y_true : array-like
        Etiquetas verdaderas.
    y_pred : array-like
        Predicciones del modelo.
    y_score : array-like | None
        Probabilidades predichas (para ordenar por confianza).
    error_type : str
        Tipo de error a buscar: 'fp' (falsos positivos), 'fn' (falsos negativos)
        o 'both' (ambos).
    n : int
        Número máximo de instancias a retornar por tipo.

    Retorna
    -------
    dict[str, np.ndarray]
        Diccionario con claves 'false_positives' y/o 'false_negatives',
        cada una con los índices de las instancias mal clasificadas.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    result = {}

    if error_type in ("fp", "both"):
        fp_mask = (y_pred == 1) & (y_true == 0)
        fp_indices = np.where(fp_mask)[0]
        if y_score is not None and len(fp_indices) > 0:
            scores = np.asarray(y_score)[fp_indices]
            fp_indices = fp_indices[np.argsort(-scores)]
        result["false_positives"] = fp_indices[:n]

    if error_type in ("fn", "both"):
        fn_mask = (y_pred == 0) & (y_true == 1)
        fn_indices = np.where(fn_mask)[0]
        if y_score is not None and len(fn_indices) > 0:
            scores = np.asarray(y_score)[fn_indices]
            fn_indices = fn_indices[np.argsort(scores)]
        result["false_negatives"] = fn_indices[:n]

    return result
