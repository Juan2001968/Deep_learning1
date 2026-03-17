"""Pipeline completo de scikit-learn para clasificación CTR con MLPClassifier.

Incluye construcción del preprocesador (Target Encoding + OneHot Encoding),
pipeline con MLPClassifier, GridSearchCV y guardado de modelos.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder


DEFAULT_PARAM_GRID = {
    "classifier__hidden_layer_sizes": [(50,), (100,), (100, 50)],
    "classifier__alpha": [0.0001, 0.001, 0.01],
    "classifier__max_iter": [50, 100],
}


def build_sklearn_preprocessor(
    categorical_columns: list[str],
    numeric_columns: list[str],
    high_cardinality_columns: list[str] | None = None,
    min_frequency: int = 20,
    max_categories: int = 100,
) -> ColumnTransformer:
    """Construye el preprocesador con Target Encoding y OneHot Encoding.

    Las variables de alta cardinalidad se codifican con Target Encoding
    para evitar explosión dimensional, mientras que las de baja cardinalidad
    usan OneHot Encoding estándar.

    Parámetros
    ----------
    categorical_columns : list[str]
        Todas las columnas categóricas.
    numeric_columns : list[str]
        Columnas numéricas a escalar.
    high_cardinality_columns : list[str] | None
        Columnas de alta cardinalidad para Target Encoding.
        Si es None, todas las categóricas usan OneHot.
    min_frequency : int
        Frecuencia mínima para categorías en OneHotEncoder.
    max_categories : int
        Número máximo de categorías para OneHotEncoder.

    Retorna
    -------
    ColumnTransformer
        Preprocesador configurado.
    """
    transformers = []

    if high_cardinality_columns:
        low_cardinality = [c for c in categorical_columns if c not in high_cardinality_columns]
        transformers.append(
            ("target_enc", TargetEncoder(smooth="auto"), high_cardinality_columns)
        )
    else:
        low_cardinality = list(categorical_columns)

    if low_cardinality:
        transformers.append(
            ("onehot", OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                sparse_output=True,
                min_frequency=min_frequency,
                max_categories=max_categories,
            ), low_cardinality)
        )

    transformers.append(
        ("numeric", StandardScaler(with_mean=False), numeric_columns)
    )

    return ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0.8,
    )


def build_sklearn_pipeline(
    categorical_columns: list[str],
    numeric_columns: list[str],
    high_cardinality_columns: list[str] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Construye el pipeline completo de preprocesamiento + MLPClassifier.

    Parámetros
    ----------
    categorical_columns : list[str]
        Columnas categóricas.
    numeric_columns : list[str]
        Columnas numéricas.
    high_cardinality_columns : list[str] | None
        Columnas de alta cardinalidad para Target Encoding.
    random_state : int
        Semilla para reproducibilidad.

    Retorna
    -------
    Pipeline
        Pipeline de scikit-learn listo para entrenar.
    """
    preprocessor = build_sklearn_preprocessor(
        categorical_columns, numeric_columns, high_cardinality_columns
    )
    classifier = MLPClassifier(
        hidden_layer_sizes=(100,),
        alpha=0.001,
        max_iter=50,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def split_train_test(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Divide los datos en conjuntos de entrenamiento y prueba con estratificación.

    Parámetros
    ----------
    X : pd.DataFrame
        Matriz de features.
    y : pd.Series
        Vector target.
    test_size : float
        Proporción del conjunto de prueba.
    random_state : int
        Semilla para reproducibilidad.

    Retorna
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def run_grid_search(
    pipeline: Pipeline,
    X_train,
    y_train,
    param_grid: dict | None = None,
    scoring: str = "roc_auc",
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 2,
):
    """Ejecuta GridSearchCV sobre el pipeline con los hiperparámetros especificados.

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline de scikit-learn.
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento.
    param_grid : dict | None
        Grilla de hiperparámetros. Usa DEFAULT_PARAM_GRID si es None.
    scoring : str
        Métrica de evaluación para la selección.
    cv : int
        Número de folds de validación cruzada.
    n_jobs : int
        Número de jobs paralelos (-1 = todos los cores).
    verbose : int
        Nivel de verbosidad.

    Retorna
    -------
    tuple
        ``(GridSearchCV, training_seconds)`` con el objeto de búsqueda
        ajustado y el tiempo total de entrenamiento en segundos.
    """
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid or DEFAULT_PARAM_GRID,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=False,
    )
    start = perf_counter()
    search.fit(X_train, y_train)
    training_seconds = perf_counter() - start
    return search, training_seconds


def save_sklearn_model(model, save_path: str | Path, filename: str = "sklearn_best_mlp.joblib") -> Path:
    """Guarda el mejor modelo de scikit-learn en disco usando joblib.

    Parámetros
    ----------
    model : object
        Modelo o pipeline entrenado a guardar.
    save_path : str | Path
        Directorio donde guardar el modelo.
    filename : str
        Nombre del archivo de salida.

    Retorna
    -------
    Path
        Ruta completa del modelo guardado.
    """
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    full_path = path / filename
    joblib.dump(model, full_path)
    print(f"Modelo guardado en: {full_path}")
    return full_path
