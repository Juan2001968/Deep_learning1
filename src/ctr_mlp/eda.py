"""Funciones de visualización para el Análisis Exploratorio de Datos (EDA).

Todas las gráficas utilizan el tema oscuro profesional definido en
``config.py`` y están diseñadas para guardarse en ``reports/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from ctr_mlp.config import COLORS, PALETTE_CATEGORICAL, FIGURE_BG, AXES_BG


# ── Funciones de análisis tabular ─────────────────────────────────────────────


def build_schema_report(df: pd.DataFrame) -> pd.DataFrame:
    """Genera un reporte de esquema con tipos, missings y cardinalidad.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a analizar.

    Retorna
    -------
    pd.DataFrame
        Tabla con columnas: dtype, missing_values, missing_rate, n_unique.
    """
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype("string"),
            "missing_values": df.isna().sum(),
            "missing_rate": df.isna().mean().round(4),
            "n_unique": df.nunique(dropna=True),
        }
    ).sort_values(by=["missing_rate", "n_unique"], ascending=[False, False])


def target_distribution(df: pd.DataFrame, target_col: str = "click") -> pd.DataFrame:
    """Calcula la distribución de la variable target con conteos y porcentajes.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la columna target.
    target_col : str
        Nombre de la columna target.

    Retorna
    -------
    pd.DataFrame
        Tabla con conteos y proporciones por clase.
    """
    counts = df[target_col].value_counts(dropna=False).sort_index()
    shares = df[target_col].value_counts(normalize=True, dropna=False).sort_index()
    return pd.DataFrame({"count": counts, "share": shares.round(4)})


def categorical_cardinality(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Calcula la cardinalidad de todas las variables categóricas.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las columnas a analizar.
    categorical_cols : list[str]
        Lista de nombres de columnas categóricas.

    Retorna
    -------
    pd.DataFrame
        Tabla de cardinalidad ordenada de mayor a menor.
    """
    return (
        pd.Series(
            {col: df[col].nunique(dropna=True) for col in categorical_cols},
            name="cardinality",
        )
        .sort_values(ascending=False)
        .to_frame()
    )


# ── Funciones de visualización ────────────────────────────────────────────────


def plot_missing_values(df: pd.DataFrame, figsize: tuple = (14, 6), save_path: str | Path | None = None):
    """Genera un gráfico de barras horizontales con el conteo de valores faltantes.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a analizar.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura. None = no guardar.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    missing = df.isna().sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(missing.index.astype(str), missing.values, color=COLORS["accent"], edgecolor="none")
    for bar, val in zip(bars, missing.values):
        if val > 0:
            ax.text(bar.get_width() + missing.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9, color=COLORS["light"])
    ax.set_title("Valores Faltantes por Columna", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cantidad de valores faltantes")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "click",
    figsize: tuple = (12, 6),
    save_path: str | Path | None = None,
):
    """Genera un gráfico de barras de la variable target con porcentajes anotados.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la columna target.
    target_col : str
        Nombre de la columna target.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    counts = df[target_col].value_counts().sort_index()
    total = counts.sum()
    labels = ["No Click (0)", "Click (1)"]
    colors = [COLORS["primary"], COLORS["accent"]]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="none", width=0.5)
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                f"{count:,}\n({pct:.2f}%)", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=COLORS["light"])
    ax.set_title("Distribución de la Variable Target (click)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cantidad de registros")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_cardinality_chart(
    cardinality_df: pd.DataFrame,
    figsize: tuple = (14, 8),
    save_path: str | Path | None = None,
):
    """Genera un gráfico de barras horizontal ordenado por cardinalidad.

    Parámetros
    ----------
    cardinality_df : pd.DataFrame
        DataFrame con índice = nombre de columna y columna 'cardinality'.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    sorted_df = cardinality_df.sort_values("cardinality", ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    colors_grad = plt.cm.cool(np.linspace(0.2, 0.9, len(sorted_df)))
    bars = ax.barh(sorted_df.index.astype(str), sorted_df["cardinality"].values,
                   color=colors_grad, edgecolor="none")
    for bar, val in zip(bars, sorted_df["cardinality"].values):
        ax.text(bar.get_width() + sorted_df["cardinality"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9, color=COLORS["light"])
    ax.set_title("Cardinalidad de Variables Categóricas", fontsize=14, fontweight="bold")
    ax.set_xlabel("Valores únicos")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_numeric_histograms(
    df: pd.DataFrame,
    numeric_cols: list[str],
    bins: int = 30,
    figsize_per_plot: tuple = (5, 4),
    save_path: str | Path | None = None,
):
    """Genera histogramas para múltiples variables numéricas con tema oscuro.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las columnas numéricas.
    numeric_cols : list[str]
        Lista de columnas numéricas a graficar.
    bins : int
        Número de bins para los histogramas.
    figsize_per_plot : tuple
        Tamaño por sub-gráfico individual.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    n = len(numeric_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.hist(df[col].dropna(), bins=bins, color=COLORS["primary"], edgecolor=AXES_BG, alpha=0.85)
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distribución de Variables Numéricas", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_top_categories(
    df: pd.DataFrame,
    column: str,
    top_n: int = 15,
    figsize: tuple = (12, 6),
    save_path: str | Path | None = None,
):
    """Genera gráfico de barras horizontales con los top N valores de una variable categórica.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la columna categórica.
    column : str
        Nombre de la columna a analizar.
    top_n : int
        Número de categorías top a mostrar.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    top_values = df[column].value_counts(dropna=False).head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    colors_grad = [PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)] for i in range(len(top_values))]
    bars = ax.barh(top_values.index.astype(str)[::-1], top_values.values[::-1],
                   color=colors_grad[::-1], edgecolor="none")
    for bar, val in zip(bars, top_values.values[::-1]):
        ax.text(bar.get_width() + top_values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9, color=COLORS["light"])
    ax.set_title(f"Top {top_n} Categorías — {column}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Frecuencia")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_click_rate_by_hour(
    df: pd.DataFrame,
    hour_col: str = "event_hour",
    target_col: str = "click",
    figsize: tuple = (14, 6),
    save_path: str | Path | None = None,
):
    """Genera un gráfico de línea/barras del CTR por hora del día.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas de hora y target.
    hour_col : str
        Nombre de la columna de hora (0-23).
    target_col : str
        Nombre de la columna target.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    ctr_by_hour = df.groupby(hour_col)[target_col].mean().reset_index()
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(ctr_by_hour[hour_col], ctr_by_hour[target_col],
                    alpha=0.15, color=COLORS["info"])
    ax.plot(ctr_by_hour[hour_col], ctr_by_hour[target_col],
            marker="o", linewidth=2.5, color=COLORS["info"], markersize=8,
            markerfacecolor=COLORS["accent"], markeredgecolor=COLORS["info"])
    for _, row in ctr_by_hour.iterrows():
        ax.text(row[hour_col], row[target_col] + 0.002,
                f"{row[target_col]:.3f}", ha="center", fontsize=8, color=COLORS["light"])
    ax.set_title("Click-Through Rate (CTR) por Hora del Día", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("CTR (proporción de clics)")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: list[str],
    figsize: tuple = (14, 8),
    save_path: str | Path | None = None,
):
    """Genera una matriz de correlación con heatmap anotado.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las columnas numéricas.
    numeric_cols : list[str]
        Lista de columnas numéricas para la correlación.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.color_palette("blend:#1E293B,#6366F1,#EC4899", as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                linewidths=0.5, linecolor=AXES_BG, cbar_kws={"shrink": 0.8},
                annot_kws={"fontsize": 10, "color": COLORS["light"]})
    ax.set_title("Matriz de Correlación — Variables Numéricas", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_ctr_by_category(
    df: pd.DataFrame,
    category_col: str,
    target_col: str = "click",
    figsize: tuple = (12, 6),
    save_path: str | Path | None = None,
):
    """Genera un gráfico de barras del CTR por una variable categórica.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la variable categórica y el target.
    category_col : str
        Nombre de la columna categórica.
    target_col : str
        Nombre de la columna target.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    ctr = df.groupby(category_col)[target_col].agg(["mean", "count"]).reset_index()
    ctr.columns = [category_col, "ctr", "volume"]
    ctr = ctr.sort_values("ctr", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors_bar = [COLORS["primary"] if v < ctr["ctr"].median() else COLORS["accent"] for v in ctr["ctr"]]
    bars = ax.barh(ctr[category_col].astype(str), ctr["ctr"], color=colors_bar, edgecolor="none")
    for bar, val in zip(bars, ctr["ctr"]):
        ax.text(bar.get_width() + ctr["ctr"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color=COLORS["light"])
    ax.set_title(f"CTR por {category_col}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Click-Through Rate")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_top_with_ctr(
    df: pd.DataFrame,
    category_col: str,
    target_col: str = "click",
    top_n: int = 10,
    figsize: tuple = (14, 7),
    save_path: str | Path | None = None,
):
    """Genera un gráfico dual con volumen (barras) y CTR (línea) para top N categorías.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la variable categórica y el target.
    category_col : str
        Nombre de la columna categórica.
    target_col : str
        Nombre de la columna target.
    top_n : int
        Número de categorías top por volumen a mostrar.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    stats = df.groupby(category_col)[target_col].agg(["count", "mean"]).reset_index()
    stats.columns = [category_col, "volume", "ctr"]
    stats = stats.nlargest(top_n, "volume").sort_values("volume", ascending=True)

    fig, ax1 = plt.subplots(figsize=figsize)
    bars = ax1.barh(stats[category_col].astype(str), stats["volume"],
                    color=COLORS["primary"], alpha=0.7, edgecolor="none", label="Volumen")
    ax1.set_xlabel("Volumen (registros)", color=COLORS["primary"])
    ax1.set_title(f"Top {top_n} {category_col} — Volumen y CTR", fontsize=14, fontweight="bold")

    ax2 = ax1.twiny()
    ax2.plot(stats["ctr"], stats[category_col].astype(str),
             marker="D", color=COLORS["accent"], linewidth=2, markersize=8, label="CTR")
    ax2.set_xlabel("CTR", color=COLORS["accent"])

    for _, row in stats.iterrows():
        ax1.text(row["volume"] + stats["volume"].max() * 0.01,
                 str(row[category_col]),
                 f'{row["volume"]:,}', va="center", fontsize=8, color=COLORS["light"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
               facecolor=AXES_BG, edgecolor="#334155", labelcolor=COLORS["light"])
    ax1.grid(axis="x", alpha=0.15)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_records_by_day(
    df: pd.DataFrame,
    day_col: str = "event_day",
    figsize: tuple = (14, 6),
    save_path: str | Path | None = None,
):
    """Genera un gráfico de barras con la distribución de registros por día.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la columna de día.
    day_col : str
        Nombre de la columna de día del mes.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    daily = df[day_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(daily.index.astype(str), daily.values,
                  color=COLORS["secondary"], edgecolor="none", alpha=0.85)
    for bar, val in zip(bars, daily.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + daily.max() * 0.01,
                f"{val:,}", ha="center", fontsize=8, color=COLORS["light"])
    ax.set_title("Distribución de Registros por Día", fontsize=14, fontweight="bold")
    ax.set_xlabel("Día del mes")
    ax.set_ylabel("Cantidad de registros")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_dashboard_summary(
    df: pd.DataFrame,
    target_col: str = "click",
    figsize: tuple = (16, 8),
    save_path: str | Path | None = None,
):
    """Genera un dashboard visual resumen con las métricas clave del dataset.

    Muestra total de registros, CTR global, número de features,
    categóricas, numéricas y balance de clases en un formato de tarjetas.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame completo (o preview) para calcular estadísticas.
    target_col : str
        Nombre de la columna target.
    figsize : tuple
        Tamaño de la figura.
    save_path : str | Path | None
        Ruta para guardar la figura.

    Retorna
    -------
    matplotlib.figure.Figure
        La figura generada.
    """
    total = len(df)
    ctr = df[target_col].mean()
    n_features = len(df.columns)
    n_cat = len(df.select_dtypes(include=["object", "string", "category"]).columns)
    n_num = len(df.select_dtypes(include=["number"]).columns)
    n_missing = int(df.isna().sum().sum())

    metrics = [
        ("Total Registros", f"{total:,}", COLORS["primary"]),
        ("CTR Global", f"{ctr:.4f} ({ctr*100:.2f}%)", COLORS["accent"]),
        ("# Features", str(n_features), COLORS["success"]),
        ("# Categóricas", str(n_cat), COLORS["secondary"]),
        ("# Numéricas", str(n_num), COLORS["info"]),
        ("Valores Faltantes", f"{n_missing:,}", COLORS["warning"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    for ax, (title, value, color) in zip(axes, metrics):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True,
                                   facecolor=color, alpha=0.15, linewidth=2,
                                   edgecolor=color, transform=ax.transAxes))
        ax.text(0.5, 0.65, value, ha="center", va="center",
                fontsize=20, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.3, title, ha="center", va="center",
                fontsize=12, color=COLORS["light"], alpha=0.8, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(AXES_BG)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Dashboard Resumen del Dataset Avazu",
                 fontsize=18, fontweight="bold", color=COLORS["light"], y=1.02)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# ── Utilidad para guardar figuras ─────────────────────────────────────────────


def save_figure(fig, output_path: str | Path) -> Path:
    """Guarda una figura matplotlib en disco.

    Parámetros
    ----------
    fig : matplotlib.figure.Figure
        Figura a guardar.
    output_path : str | Path
        Ruta de destino para el archivo de imagen.

    Retorna
    -------
    Path
        La ruta donde se guardó la figura.
    """
    return _save(fig, output_path)


def _save(fig, output_path: str | Path) -> Path:
    """Utilidad interna para guardar figuras creando directorios padre si es necesario."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path
