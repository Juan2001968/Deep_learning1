"""Configuración central del proyecto Avazu CTR MLP.

Contiene rutas, constantes, paleta de colores, estilo visual oscuro
y funciones de carga de configuración reutilizables por todos los
módulos y notebooks del proyecto.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project_settings.toml"

# ── Paleta de colores moderna (fondo oscuro profesional) ──────────────────────

COLORS = {
    "primary": "#6366F1",
    "secondary": "#8B5CF6",
    "accent": "#EC4899",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#06B6D4",
    "dark": "#1E1B4B",
    "light": "#F8FAFC",
    "gradient": ["#6366F1", "#8B5CF6", "#A78BFA", "#C4B5FD", "#DDD6FE"],
}

PALETTE_CATEGORICAL = [
    "#6366F1", "#EC4899", "#10B981", "#F59E0B", "#06B6D4",
    "#8B5CF6", "#EF4444", "#A78BFA", "#34D399", "#FBBF24",
    "#22D3EE", "#F472B6", "#818CF8", "#4ADE80", "#FB923C",
]

FIGURE_BG = "#0F172A"
AXES_BG = "#1E293B"

# ── Estilo base matplotlib (tema oscuro profesional) ──────────────────────────

DARK_STYLE: dict[str, object] = {
    "figure.facecolor": FIGURE_BG,
    "axes.facecolor": AXES_BG,
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#E2E8F0",
    "text.color": "#E2E8F0",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "grid.color": "#334155",
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": FIGURE_BG,
}


def apply_dark_style() -> None:
    """Aplica el estilo visual oscuro profesional a matplotlib globalmente.

    Debe llamarse al inicio de cada notebook para garantizar consistencia
    visual en todas las gráficas generadas.
    """
    plt.rcParams.update(DARK_STYLE)


def load_project_settings(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Carga la configuración del proyecto y resuelve rutas relativas.

    Parámetros
    ----------
    config_path : str | Path
        Ruta al archivo TOML de configuración del proyecto.

    Retorna
    -------
    dict
        Diccionario con todas las secciones de configuración.
        Las rutas en ``[paths]`` se resuelven como absolutas desde la raíz
        del repositorio.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("rb") as stream:
        settings = tomllib.load(stream)

    settings["paths"] = {
        key: (PROJECT_ROOT / value).resolve()
        for key, value in settings.get("paths", {}).items()
    }
    settings["project_root"] = PROJECT_ROOT
    return settings
