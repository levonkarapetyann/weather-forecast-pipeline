"""
=============================================================================
MODULE: Project Paths & Configuration Resolver (project_paths.py)
-----------------------------------------------------------------------------
PURPOSE:
Utility module for resolving absolute paths to configuration files, models,
and datasets regardless of the current working directory (OS / WSL / macOS).

KEY FUNCTIONS:
1. 
esolve_path(*paths): Resolves absolute paths relative to the project root.
2. load_settings(): Safely reads the central config/settings.json.
3. load_json(): Safely parses JSON files with fallback defaults.
4. load_stations_config(): Loads stations list from config/stations.json.
5. load_scalers_config(): Loads scaling parameters from config/scalers.json.
=============================================================================
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(*parts: Union[str, os.PathLike]) -> str:
    """Returns an absolute filesystem path relative to the repository root."""
    return str(REPO_ROOT.joinpath(*map(str, parts)))


def load_settings(settings_path: str | None = None) -> Dict[str, Any]:
    """Loads settings.json from the project root or a user-specified path."""
    if settings_path is None:
        settings_path = resolve_path("config", "settings.json")

    with open(settings_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path: str | None, default: Any = None) -> Any:
    """Safely loads a JSON file. Returns default if the file does not exist."""
    if path is None:
        return default

    resolved = path if os.path.isabs(path) else resolve_path(path)
    if not os.path.exists(resolved):
        return default

    with open(resolved, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_stations_config(stations_path: str | None = None) -> List[Dict[str, Any]]:
    """Loads the list of meteorological stations from config/stations.json."""
    if stations_path is None:
        stations_path = resolve_path("config", "stations.json")
    data = load_json(stations_path, default={"stations": []})
    return data.get("stations", []) if isinstance(data, dict) else []


def load_scalers_config(scalers_path: str | None = None) -> Dict[str, Any]:
    """Loads scaling parameters from config/scalers.json."""
    if scalers_path is None:
        try:
            settings = load_settings()
            scalers_path = resolve_path(settings["paths"]["scalers_file"])
        except Exception:
            scalers_path = resolve_path("config", "scalers.json")
    return load_json(scalers_path, default={})
