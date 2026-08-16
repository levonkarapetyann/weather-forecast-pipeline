"""
=============================================================================
МОДУЛЬ: Project Paths & Configuration Resolver (project_paths.py)
-----------------------------------------------------------------------------
НАЗНАЧЕНИЕ:
Вспомогательный модуль для резолвинга абсолютных путей к файлам конфигурации,
моделям и датасетам независимо от текущей рабочей директории вызова (OS / WSL / macOS).

ОСНОВНЫЕ ФУНКЦИИ:
1. `resolve_path(*paths)`: резолвинг абсолютного пути относительно корня проекта.
2. `load_settings()`: безопасное чтение центрального `config/settings.json`.
=============================================================================
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(*parts: Union[str, os.PathLike]) -> str:
    """Возвращает абсолютный путь относительно корня репозитория."""
    return str(REPO_ROOT.joinpath(*map(str, parts)))


def load_settings(settings_path: str | None = None) -> Dict[str, Any]:
    """Загружает settings.json из корня проекта или из пользовательского пути."""
    if settings_path is None:
        settings_path = resolve_path("config", "settings.json")

    with open(settings_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path: str | None, default: Any = None) -> Any:
    """Безопасно загружает JSON-файл. Возвращает default при отсутствии файла."""
    if path is None:
        return default

    resolved = path if os.path.isabs(path) else resolve_path(path)
    if not os.path.exists(resolved):
        return default

    with open(resolved, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_stations_config(stations_path: str | None = None) -> List[Dict[str, Any]]:
    """Загружает список станций из config/stations.json."""
    if stations_path is None:
        stations_path = resolve_path("config", "stations.json")
    data = load_json(stations_path, default={"stations": []})
    return data.get("stations", []) if isinstance(data, dict) else []


def load_scalers_config(scalers_path: str | None = None) -> Dict[str, Any]:
    """Загружает параметры масштабирования из config/scalers.json."""
    if scalers_path is None:
        try:
            settings = load_settings()
            scalers_path = resolve_path(settings["paths"]["scalers_file"])
        except Exception:
            scalers_path = resolve_path("config", "scalers.json")
    return load_json(scalers_path, default={})
