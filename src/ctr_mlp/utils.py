"""Funciones auxiliares de uso general para el proyecto Avazu CTR MLP.

Incluye control de semillas aleatorias, gestión de directorios,
medición de tiempos y formateo de valores.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Fija las semillas aleatorias de Python y NumPy para reproducibilidad.

    Parámetros
    ----------
    seed : int
        Valor de la semilla aleatoria (por defecto 42).
    """
    random.seed(seed)
    np.random.seed(seed)


def ensure_directory(path: str | Path) -> Path:
    """Crea un directorio (y sus padres) si no existe.

    Parámetros
    ----------
    path : str | Path
        Ruta del directorio a crear.

    Retorna
    -------
    Path
        La ruta del directorio creado o existente.
    """
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


@contextmanager
def timer(label: str):
    """Context manager que mide y muestra el tiempo de ejecución de un bloque.

    Parámetros
    ----------
    label : str
        Etiqueta descriptiva que se muestra junto al tiempo medido.

    Ejemplo
    -------
    >>> with timer("Entrenamiento"):
    ...     model.fit(X, y)
    Entrenamiento: 12.34 seconds
    """
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        print(f"{label}: {elapsed:.2f} seconds")


def format_seconds(seconds: float) -> str:
    """Formatea un valor en segundos a una cadena legible.

    Parámetros
    ----------
    seconds : float
        Tiempo en segundos a formatear.

    Retorna
    -------
    str
        Representación legible del tiempo (ej. ``'2m 13.4s'`` o ``'5.23s'``).
    """
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    return f"{seconds:.2f}s"
