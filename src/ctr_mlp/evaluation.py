"""Funciones de evaluación y visualización de métricas para modelos de clasificación.

Incluye cálculo de métricas binarias, visualización de matrices de confusión,
curvas ROC, gráficos comparativos y tablas de comparación entre frameworks.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ctr_mlp.config import COLORS, AXES_BG, FIGURE_BG


def compute_binary_metrics(y_true, y_pred, y_score=None) -> dict[str, float | int | None]:
    """Calcula las métricas de clasificación binaria requeridas.

    Parámetros
    ----------
    y_true : array-like
        Etiquetas verdaderas.
    y_pred : array-like
        Predicciones del modelo.
    y_score : array-like | None
        Probabilidades predichas para la clase positiva (para ROC AUC).

    Retorna
    -------
    dict[str, float | int | None]
        Diccionario con accuracy, precision, recall, f1, roc_auc y
        componentes de la matriz de confusión (tn, fp, fn, tp).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None else None,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return metrics


def benchmark_predictions(estimator, X_test) -> tuple:
    """Mide el tiempo de predicción y obtiene predicciones y probabilidades.

    Parámetros
    ----------
    estimator : sklearn estimator
        Modelo entrenado con métodos ``predict`` y opcionalmente ``predict_proba``.
    X_test : array-like
        Datos de prueba.

    Retorna
    -------
    tuple
        ``(y_pred, y_score, prediction_seconds)`` donde ``y_score`` es None
        si el estimador no soporta ``predict_proba``.
    """
    start = perf_counter()
    y_pred = estimator.predict(X_test)
    prediction_seconds = perf_counter() - start

    y_score = None
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X_test)[:, 1]

    return y_pred, y_score, prediction_seconds


def metrics_to_frame(metrics: dict[str, float | int | None]) -> pd.DataFrame:
    """Convierte un diccionario de métricas a un DataFrame formateado.

    Parámetros
    ----------
    metrics : dict
        Diccionario con nombres de métricas como claves.

    Retorna
    -------
    pd.DataFrame
        DataFrame transpuesto con una columna 'value'.
    """
    return pd.DataFrame([metrics]).T.rename(columns={0: "value"})


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str] | None = None,
    figsize: tuple = (10, 8),
    save_path: str | Path | None = None,
):
    """Genera un heatmap de la matriz de confusión con estilo oscuro profesional.

    Parámetros
    ----------
    y_true : array-like
        Etiquetas verdaderas.
    y_pred : array-like
        Predicciones del modelo.
    labels : list[str] | None
        Etiquetas para las clases. Por defecto ``['No Click (0)', 'Click (1)']``.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    if labels is None:
        labels = ["No Click (0)", "Click (1)"]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.color_palette("blend:#1E293B,#8B5CF6,#EC4899", as_cmap=True)
    sns.heatmap(cm, annot=True, fmt=",d", cmap=cmap, ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=2, linecolor=AXES_BG,
                annot_kws={"fontsize": 16, "fontweight": "bold", "color": COLORS["light"]},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Matriz de Confusión", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Valor Real", fontsize=12)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_roc_curve(
    y_true,
    y_score,
    model_name: str = "MLPClassifier",
    figsize: tuple = (12, 8),
    save_path: str | Path | None = None,
):
    """Genera la curva ROC con área sombreada y AUC anotado.

    Parámetros
    ----------
    y_true : array-like
        Etiquetas verdaderas.
    y_score : array-like
        Probabilidades predichas para la clase positiva.
    model_name : str
        Nombre del modelo para la leyenda.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["primary"])
    ax.plot(fpr, tpr, linewidth=2.5, color=COLORS["primary"],
            label=f"{model_name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "--", linewidth=1.5, color=COLORS["danger"], alpha=0.5,
            label="Clasificador aleatorio")
    ax.set_title("Curva ROC", fontsize=16, fontweight="bold")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    ax.legend(loc="lower right", fontsize=11,
              facecolor=AXES_BG, edgecolor="#334155", labelcolor=COLORS["light"])
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_comparative_roc(
    results: list[dict],
    figsize: tuple = (12, 8),
    save_path: str | Path | None = None,
):
    """Genera curvas ROC superpuestas para comparar múltiples modelos/frameworks.

    Parámetros
    ----------
    results : list[dict]
        Lista de diccionarios con claves: 'name', 'y_true', 'y_score'.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    line_colors = [COLORS["primary"], COLORS["accent"], COLORS["success"], COLORS["warning"]]
    fig, ax = plt.subplots(figsize=figsize)

    for i, result in enumerate(results):
        fpr, tpr, _ = roc_curve(result["y_true"], result["y_score"])
        auc_val = roc_auc_score(result["y_true"], result["y_score"])
        color = line_colors[i % len(line_colors)]
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.plot(fpr, tpr, linewidth=2.5, color=color,
                label=f'{result["name"]} (AUC = {auc_val:.4f})')

    ax.plot([0, 1], [0, 1], "--", linewidth=1.5, color=COLORS["danger"], alpha=0.5,
            label="Clasificador aleatorio")
    ax.set_title("Comparación de Curvas ROC — scikit-learn vs PySpark",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    ax.legend(loc="lower right", fontsize=11,
              facecolor=AXES_BG, edgecolor="#334155", labelcolor=COLORS["light"])
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_comparative_times(
    comparison_df: pd.DataFrame,
    figsize: tuple = (12, 6),
    save_path: str | Path | None = None,
):
    """Genera un gráfico de barras agrupadas comparando tiempos de cómputo.

    Parámetros
    ----------
    comparison_df : pd.DataFrame
        DataFrame con columnas 'framework', 'training_seconds', 'prediction_seconds'.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(comparison_df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, comparison_df["training_seconds"],
                   width, label="Entrenamiento", color=COLORS["primary"],
                   edgecolor="none", alpha=0.85)
    bars2 = ax.bar(x + width / 2, comparison_df["prediction_seconds"],
                   width, label="Predicción", color=COLORS["accent"],
                   edgecolor="none", alpha=0.85)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}s", ha="center", fontsize=10,
                color=COLORS["light"], fontweight="bold")

    ax.set_title("Comparación de Tiempos de Cómputo", fontsize=16, fontweight="bold")
    ax.set_xlabel("Framework")
    ax.set_ylabel("Tiempo (segundos)")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["framework"])
    ax.legend(facecolor=AXES_BG, edgecolor="#334155", labelcolor=COLORS["light"])
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def build_comparison_table(
    sklearn_metrics: dict,
    spark_metrics: dict,
    sklearn_dataset_size: int,
    spark_dataset_size: int,
) -> pd.DataFrame:
    """Construye una tabla comparativa lado a lado entre scikit-learn y PySpark.

    Parámetros
    ----------
    sklearn_metrics : dict
        Métricas del modelo scikit-learn (incluyendo tiempos).
    spark_metrics : dict
        Métricas del modelo PySpark (incluyendo tiempos).
    sklearn_dataset_size : int
        Número de registros usados en scikit-learn.
    spark_dataset_size : int
        Número de registros usados en PySpark.

    Retorna
    -------
    pd.DataFrame
        Tabla comparativa con métricas, tiempos y tamaños de dataset.
    """
    rows = []
    for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        sk_val = sklearn_metrics.get(metric_name)
        sp_val = spark_metrics.get(metric_name)
        rows.append({
            "Métrica": metric_name.upper().replace("_", " "),
            "scikit-learn": f"{sk_val:.4f}" if sk_val is not None else "N/A",
            "PySpark": f"{sp_val:.4f}" if sp_val is not None else "N/A",
        })
    rows.append({
        "Métrica": "Tiempo entrenamiento",
        "scikit-learn": f'{sklearn_metrics.get("training_seconds", 0):.1f}s',
        "PySpark": f'{spark_metrics.get("training_seconds", 0):.1f}s',
    })
    rows.append({
        "Métrica": "Tiempo predicción",
        "scikit-learn": f'{sklearn_metrics.get("prediction_seconds", 0):.1f}s',
        "PySpark": f'{spark_metrics.get("prediction_seconds", 0):.1f}s',
    })
    rows.append({
        "Métrica": "Tamaño dataset",
        "scikit-learn": f"{sklearn_dataset_size:,}",
        "PySpark": f"{spark_dataset_size:,}",
    })
    return pd.DataFrame(rows)


def _save(fig, output_path: str | Path) -> Path:
    """Utilidad interna para guardar figuras."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path
