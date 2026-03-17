"""Ingeniería de features para el proyecto Avazu CTR MLP.

Contiene las transformaciones temporales requeridas para convertir la
columna ``hour`` (formato YYMMDDHH) en features útiles para el modelado,
tanto en Pandas como en PySpark.
"""

from __future__ import annotations

import pandas as pd


TIME_BUCKET_ORDER = ["madrugada", "manana", "tarde", "noche"]

FRANJA_HORARIA_MAP = {
    "madrugada": "Madrugada (00-05h)",
    "manana": "Mañana (06-11h)",
    "tarde": "Tarde (12-17h)",
    "noche": "Noche (18-23h)",
}


def add_time_features_pandas(df: pd.DataFrame, hour_col: str = "hour") -> pd.DataFrame:
    """Crea features temporales a partir de la columna ``hour`` del dataset Avazu.

    Transforma la columna en formato YYMMDDHH en múltiples variables derivadas
    que capturan patrones temporales útiles para la predicción de clics.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la columna ``hour`` en formato YYMMDDHH.
    hour_col : str
        Nombre de la columna que contiene el timestamp codificado.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame con las columnas adicionales:
        - ``event_day``: día del mes.
        - ``event_hour``: hora del día (0-23).
        - ``day_of_week``: día de la semana (0=lunes, 6=domingo).
        - ``is_weekend``: 1 si es fin de semana, 0 en caso contrario.
        - ``is_business_hour``: 1 si la hora está entre 8 y 18.
        - ``time_bucket``: franja horaria (madrugada/manana/tarde/noche).
        - ``franja_horaria``: etiqueta descriptiva en español de la franja.
    """
    engineered = df.copy()
    hour_as_text = engineered[hour_col].astype("string").str.zfill(8)
    timestamp = pd.to_datetime("20" + hour_as_text, format="%Y%m%d%H", errors="coerce")

    engineered["event_day"] = timestamp.dt.day.astype("Int16")
    engineered["event_hour"] = timestamp.dt.hour.astype("Int16")
    engineered["day_of_week"] = timestamp.dt.dayofweek.astype("Int16")
    engineered["is_weekend"] = timestamp.dt.dayofweek.isin([5, 6]).astype("int8")
    engineered["is_business_hour"] = timestamp.dt.hour.between(8, 18, inclusive="both").astype("int8")
    engineered["time_bucket"] = pd.cut(
        timestamp.dt.hour,
        bins=[-1, 5, 11, 17, 23],
        labels=TIME_BUCKET_ORDER,
    ).astype("string")
    engineered["franja_horaria"] = engineered["time_bucket"].map(FRANJA_HORARIA_MAP).astype("string")

    return engineered


def add_time_features_spark(df, hour_col: str = "hour"):
    """Crea features temporales en un DataFrame de PySpark (equivalente Spark).

    Aplica la misma lógica de ingeniería temporal que ``add_time_features_pandas``
    pero usando funciones nativas de Spark SQL para procesamiento distribuido.

    Parámetros
    ----------
    df : pyspark.sql.DataFrame
        DataFrame de Spark con la columna ``hour`` en formato YYMMDDHH.
    hour_col : str
        Nombre de la columna que contiene el timestamp codificado.

    Retorna
    -------
    pyspark.sql.DataFrame
        DataFrame con las columnas temporales derivadas añadidas.
    """
    from pyspark.sql import functions as F

    padded_hour = F.lpad(F.col(hour_col).cast("string"), 8, "0")
    timestamp = F.to_timestamp(F.concat(F.lit("20"), padded_hour), "yyyyMMddHH")

    return (
        df.withColumn("event_timestamp", timestamp)
        .withColumn("event_day", F.dayofmonth("event_timestamp"))
        .withColumn("event_hour", F.hour("event_timestamp"))
        .withColumn("day_of_week", F.dayofweek("event_timestamp"))
        .withColumn("is_weekend", F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0))
        .withColumn(
            "is_business_hour",
            F.when(F.col("event_hour").between(8, 18), 1).otherwise(0),
        )
        .withColumn(
            "time_bucket",
            F.when(F.col("event_hour").between(0, 5), F.lit("madrugada"))
            .when(F.col("event_hour").between(6, 11), F.lit("manana"))
            .when(F.col("event_hour").between(12, 17), F.lit("tarde"))
            .otherwise(F.lit("noche")),
        )
    )
