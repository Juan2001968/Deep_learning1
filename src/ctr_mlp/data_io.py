"""Funciones de carga, muestreo y guardado de datos para el proyecto Avazu CTR.

Proporciona utilidades para leer el dataset comprimido (.gz) de forma
eficiente en memoria, generar muestras estratificadas y separar
features del target.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_PANDAS_COMPRESSION = "gzip"
RAW_STRING_COLUMNS = (
    "id",
    "hour",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
)


def make_pandas_dtype_map() -> dict[str, str]:
    """Retorna un mapa de tipos ligero para columnas que deben permanecer como strings.

    Retorna
    -------
    dict[str, str]
        Diccionario con nombre de columna como clave y tipo de dato como valor.
    """
    dtype_map = {column: "string" for column in RAW_STRING_COLUMNS}
    dtype_map["click"] = "int8"
    return dtype_map


def resolve_dtype_map(usecols: list[str] | tuple[str, ...] | None = None) -> dict[str, str]:
    """Filtra el mapa de tipos para incluir solo las columnas solicitadas.

    Parámetros
    ----------
    usecols : list[str] | tuple[str, ...] | None
        Lista de columnas a incluir. Si es None, retorna el mapa completo.

    Retorna
    -------
    dict[str, str]
        Mapa de tipos filtrado.
    """
    dtype_map = make_pandas_dtype_map()
    if usecols is None:
        return dtype_map
    allowed = set(usecols)
    return {column: dtype for column, dtype in dtype_map.items() if column in allowed}


def optimize_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce el uso de memoria convirtiendo columnas numéricas a tipos más pequeños.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas numéricas a optimizar.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame con tipos numéricos reducidos (downcast).
    """
    optimized = df.copy()
    for column in optimized.select_dtypes(include=["int64", "Int64"]).columns:
        optimized[column] = pd.to_numeric(optimized[column], downcast="integer")
    for column in optimized.select_dtypes(include=["float64"]).columns:
        optimized[column] = pd.to_numeric(optimized[column], downcast="float")
    return optimized


def read_dataframe_preview(
    csv_path: str | Path,
    nrows: int = 100_000,
    usecols: list[str] | tuple[str, ...] | None = None,
    compression: str = DEFAULT_PANDAS_COMPRESSION,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Lee una vista previa del dataset comprimido, optimizada en memoria.

    Parámetros
    ----------
    csv_path : str | Path
        Ruta al archivo CSV comprimido (train.gz).
    nrows : int
        Número máximo de filas a leer (por defecto 100,000).
    usecols : list[str] | tuple[str, ...] | None
        Columnas específicas a leer. None lee todas.
    compression : str
        Tipo de compresión del archivo (por defecto ``'gzip'``).

    Retorna
    -------
    pd.DataFrame
        Vista previa del dataset con tipos numéricos optimizados.
    """
    csv_path = Path(csv_path)
    dtype_map = read_csv_kwargs.pop("dtype", resolve_dtype_map(usecols))
    preview = pd.read_csv(
        csv_path,
        nrows=nrows,
        usecols=usecols,
        dtype=dtype_map,
        compression=compression,
        low_memory=False,
        **read_csv_kwargs,
    )
    return optimize_numeric_dtypes(preview)


def estimate_target_distribution(
    csv_path: str | Path,
    target_col: str = "click",
    chunksize: int = 250_000,
    compression: str = DEFAULT_PANDAS_COMPRESSION,
) -> pd.Series:
    """Estima la distribución global de la variable target leyendo por chunks.

    Lee únicamente la columna target en bloques para calcular el balance
    de clases sin cargar el archivo completo en memoria.

    Parámetros
    ----------
    csv_path : str | Path
        Ruta al archivo CSV comprimido.
    target_col : str
        Nombre de la columna target (por defecto ``'click'``).
    chunksize : int
        Tamaño de cada bloque de lectura.
    compression : str
        Tipo de compresión del archivo.

    Retorna
    -------
    pd.Series
        Conteo de cada valor de la variable target, ordenado por índice.
    """
    csv_path = Path(csv_path)
    counts = pd.Series(dtype="int64")
    for chunk in pd.read_csv(
        csv_path,
        usecols=[target_col],
        dtype={target_col: "int8"},
        chunksize=chunksize,
        compression=compression,
    ):
        counts = counts.add(chunk[target_col].value_counts(dropna=False), fill_value=0)
    return counts.fillna(0).astype("int64").sort_index()


def sample_csv_for_local_training(
    csv_path: str | Path,
    sample_size: int = 1_000_000,
    target_col: str = "click",
    chunksize: int = 200_000,
    random_state: int = 42,
    usecols: list[str] | tuple[str, ...] | None = None,
    compression: str = DEFAULT_PANDAS_COMPRESSION,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Construye una muestra aproximadamente estratificada para experimentos locales.

    Lee el archivo comprimido por bloques y realiza muestreo estratificado
    por la variable target para mantener la proporción de clases original.

    Parámetros
    ----------
    csv_path : str | Path
        Ruta al archivo CSV comprimido.
    sample_size : int
        Tamaño deseado de la muestra (por defecto 1,000,000).
    target_col : str
        Nombre de la columna target para estratificación.
    chunksize : int
        Tamaño de cada bloque de lectura.
    random_state : int
        Semilla para reproducibilidad.
    usecols : list[str] | tuple[str, ...] | None
        Columnas específicas a incluir en la muestra.
    compression : str
        Tipo de compresión del archivo.

    Retorna
    -------
    pd.DataFrame
        Muestra estratificada del dataset con tipos numéricos optimizados.
    """
    csv_path = Path(csv_path)
    requested_columns = list(usecols) if usecols is not None else None
    if requested_columns is not None and target_col not in requested_columns:
        requested_columns.append(target_col)

    dtype_map = read_csv_kwargs.pop("dtype", resolve_dtype_map(requested_columns))
    target_counts = estimate_target_distribution(
        csv_path,
        target_col=target_col,
        chunksize=chunksize,
        compression=compression,
    )
    total_rows = int(target_counts.sum())
    if total_rows == 0:
        raise ValueError(f"No se encontraron filas en {csv_path}")

    fraction = min(1.0, sample_size / total_rows)
    sampled_parts: list[pd.DataFrame] = []

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            csv_path,
            usecols=requested_columns,
            chunksize=chunksize,
            dtype=dtype_map,
            compression=compression,
            low_memory=False,
            **read_csv_kwargs,
        )
    ):
        per_class_samples = []
        for label, group in chunk.groupby(target_col, dropna=False):
            if group.empty:
                continue
            sample_count = max(1, int(round(len(group) * fraction)))
            sample_count = min(sample_count, len(group))
            per_class_samples.append(
                group.sample(n=sample_count, random_state=random_state + chunk_index + int(label))
            )
        sampled_parts.append(pd.concat(per_class_samples, ignore_index=True))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > sample_size:
        class_shares = sampled[target_col].value_counts(normalize=True)
        target_sizes = (class_shares * sample_size).round().astype(int)
        final_parts = []
        for label, group in sampled.groupby(target_col, dropna=False):
            desired_n = max(1, int(target_sizes.get(label, len(group))))
            desired_n = min(desired_n, len(group))
            final_parts.append(group.sample(n=desired_n, random_state=random_state + int(label)))
        sampled = pd.concat(final_parts, ignore_index=True)

    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    return optimize_numeric_dtypes(sampled)


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "click",
    drop_columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa un DataFrame en matriz de features (X) y vector target (y).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame completo con features y target.
    target_col : str
        Nombre de la columna target.
    drop_columns : list[str] | tuple[str, ...] | None
        Columnas adicionales a excluir de las features.

    Retorna
    -------
    tuple[pd.DataFrame, pd.Series]
        Tupla ``(X, y)`` con las features y el target separados.
    """
    columns_to_drop = [target_col, *(drop_columns or [])]
    X = df.drop(columns=columns_to_drop, errors="ignore")
    y = df[target_col].astype("int8")
    return X, y
