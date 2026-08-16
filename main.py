#!/usr/bin/env python3
"""
Единый точка входа (CLI-оркестратор) гиперлокальной системы прогнозирования погоды.

Команды:
  python main.py run-all    - Полный цикл (сбор данных -> фичи -> TFT -> CatBoost -> PID)
  python main.py collect    - Сбор данных датчиков и внешних прогнозов
  python main.py features   - Расчет физических признаков
  python main.py train      - Обучение моделей (TFT + CatBoost + PID)
  python main.py serve      - Запуск FastAPI бэкенд сервера
  python main.py dashboard  - Запуск Streamlit дашборда
  python main.py benchmark  - Оценка точности и сравнение прогнозов
"""

import argparse
import os
import subprocess
import sys

# Добавляем src/ в sys.path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PYTHON_EXE = sys.executable


def run_script(script_name: str, args: list[str] | None = None) -> int:
    """Запускает скрипт из папки src/ с переданными аргументами."""
    script_path = os.path.join(SRC_DIR, script_name)
    cmd = [PYTHON_EXE, script_path] + (args or [])
    print(f"\n[Main Orchestrator] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[Main Orchestrator] ERROR: {script_name} failed with exit code {result.returncode}")
    return result.returncode


def cmd_collect(args: argparse.Namespace) -> None:
    """Шаг 1: Сбор данных с датчиков и Open-Meteo."""
    print("=== [1/2] Сбор показаний локальных станций ===")
    collect_args = []
    if args.days:
        collect_args.extend(["--days", str(args.days)])
    if args.test:
        collect_args.append("--test")
    run_script("data_fetcher.py", collect_args)

    # Внешние прогнозы Open-Meteo (второй блок __main__ в data_fetcher.py
    # не вызывается автоматически — запуск через отдельный CLI)
    # TODO: при разделении data_fetcher.py на два файла — обновить здесь


def cmd_features(args: argparse.Namespace) -> None:
    """Шаг 2: Расчет фичей и датасетов."""
    print("=== Расчет признаков и подготовки датасетов ===")
    run_script("data_pipeline.py")


def cmd_train(args: argparse.Namespace) -> None:
    """Шаг 3: Обучение TFT, CatBoost и PID-тюнера."""
    print("=== [1/3] Обучение TFT-нейросети ===")
    tft_args = ["--epochs", str(args.epochs), "--batch_size", str(args.batch_size)]
    if run_script("train.py", tft_args) != 0:
        print("Ошибка при обучении TFT. Остановка пайплайна.")
        return

    print("\n=== [2/3] Обучение CatBoost модели остатков ===")
    run_script("residual_engine.py")

    print("\n=== [3/3] Подбор параметров PID-коррекции ===")
    run_script("pid_tuner.py")


def cmd_run_all(args: argparse.Namespace) -> None:
    """Сквозной запуск всего пайплайна."""
    print("🚀 === ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ПРОГНОЗИРОВАНИЯ ===")
    cmd_collect(args)
    cmd_features(args)
    cmd_train(args)
    print("\n✅ === ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН ===")


def cmd_serve(args: argparse.Namespace) -> None:
    """Запуск FastAPI бэкенда."""
    print(f"🌐 Запуск FastAPI бэкенда на {args.host}:{args.port}...")
    cmd = [PYTHON_EXE, "-m", "uvicorn", "src.app:app", "--host", args.host, "--port", str(args.port), "--reload"]
    subprocess.run(cmd)


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Запуск Streamlit дашборда."""
    dashboard_path = os.path.join(SRC_DIR, "dashboard.py")
    print(f"📊 Запуск Streamlit дашборда на порту {args.port}...")
    cmd = [PYTHON_EXE, "-m", "streamlit", "run", dashboard_path, "--server.port", str(args.port)]
    subprocess.run(cmd)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Оценка точности моделей."""
    print("=== Оценка точности и сравнение с Open-Meteo ===")
    run_script("evaluate.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Главный консольный оркестратор метеомодели.")
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    # run-all
    p_all = subparsers.add_parser("run-all", help="Сквозной запуск всего пайплайна")
    p_all.add_argument("--epochs", type=int, default=30, help="Количество эпох TFT (default: 30)")
    p_all.add_argument("--batch_size", type=int, default=64, help="Размер батча (default: 64)")
    p_all.add_argument("--days", type=int, default=14, help="Загружать внешние данные за N дней")
    p_all.add_argument("--test", action="store_true", help="Тестовый режим на 1 станции")
    p_all.set_defaults(func=cmd_run_all)

    # collect
    p_coll = subparsers.add_parser("collect", help="Сбор данных локальных станций и Open-Meteo")
    p_coll.add_argument("--days", type=int, default=14, help="Ограничить сбор последними N днями")
    p_coll.add_argument("--test", action="store_true", help="Тестовый запуск на 1 станции")
    p_coll.set_defaults(func=cmd_collect)

    # features
    p_feat = subparsers.add_parser("features", help="Расчет признаков в data/processed/")
    p_feat.set_defaults(func=cmd_features)

    # train
    p_tr = subparsers.add_parser("train", help="Обучение всех моделей (TFT, CatBoost, PID)")
    p_tr.add_argument("--epochs", type=int, default=30, help="Количество эпох TFT")
    p_tr.add_argument("--batch_size", type=int, default=64, help="Размер батча")
    p_tr.set_defaults(func=cmd_train)

    # serve
    p_srv = subparsers.add_parser("serve", help="Запуск FastAPI бэкенд сервера")
    p_srv.add_argument("--host", type=str, default="127.0.0.1", help="Хост")
    p_srv.add_argument("--port", type=int, default=8000, help="Порт")
    p_srv.set_defaults(func=cmd_serve)

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Запуск Streamlit дашборда")
    p_dash.add_argument("--port", type=int, default=8501, help="Порт Streamlit")
    p_dash.set_defaults(func=cmd_dashboard)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Расчет точности прогнозов")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
