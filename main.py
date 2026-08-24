#!/usr/bin/env python3
"""
Master CLI Orchestrator for the Hyperlocal Weather Forecasting Pipeline.

Commands:
  python main.py run-all    - Full pipeline (collect -> features -> train TFT + CatBoost + Rain + PID)
  python main.py collect    - Ingest sensor readings and external synoptic forecasts
  python main.py features   - Compute meteorological and thermodynamic features
  python main.py train      - Train all ML models (TFT, Residuals, Rain, PID)
  python main.py serve      - Launch FastAPI REST API backend server
  python main.py dashboard  - Launch Streamlit interactive web dashboard
  python main.py benchmark  - Evaluate multi-horizon accuracy against NWP baselines
"""

import argparse
import os
import subprocess
import sys

# Add src/ to sys.path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PYTHON_EXE = sys.executable


def run_script(script_name: str, args: list[str] | None = None) -> int:
    """Executes a Python script from src/ with provided CLI arguments."""
    script_path = os.path.join(SRC_DIR, script_name)
    cmd = [PYTHON_EXE, script_path] + (args or [])
    print(f"\n[Main Orchestrator] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[Main Orchestrator] ERROR: {script_name} failed with exit code {result.returncode}")
    return result.returncode


def cmd_collect(args: argparse.Namespace) -> None:
    """Step 1: Collect station observations and Open-Meteo external forecasts."""
    print("=== Ingesting Sensor Data & External NWP Forecasts ===")
    collect_args = []
    if args.days:
        collect_args.extend(["--days", str(args.days)])
    if args.test:
        collect_args.append("--test")
    if hasattr(args, "source") and args.source:
        collect_args.extend(["--source", args.source])
    run_script("data_fetcher.py", collect_args)


def cmd_features(args: argparse.Namespace) -> None:
    """Step 2: Compute physical features and prepare processed datasets."""
    print("=== Computing Meteorological Features & Datasets ===")
    run_script("data_pipeline.py")


def cmd_train(args: argparse.Namespace) -> None:
    """Step 3: Train TFT, CatBoost Residuals, Rain Classifier, and PID Tuner."""
    print("=== [1/4] Training TFT Neural Network ===")
    tft_args = ["--epochs", str(args.epochs), "--batch_size", str(args.batch_size)]
    if run_script("train.py", tft_args) != 0:
        print("Error training TFT. Halting pipeline.")
        return

    print("\n=== [2/4] Training CatBoost Residual Models ===")
    run_script("residual_engine.py")

    print("\n=== [3/4] Training CatBoost Rain Classifier ===")
    run_script("rain_engine.py")

    print("\n=== [4/4] Tuning Dynamic PID Error Compensation ===")
    run_script("pid_tuner.py")


def cmd_run_all(args: argparse.Namespace) -> None:
    """End-to-end execution of the complete forecasting pipeline."""
    print("🚀 === STARTING COMPLETE WEATHER FORECASTING PIPELINE ===")
    cmd_collect(args)
    cmd_features(args)
    cmd_train(args)
    print("\n✅ === PIPELINE COMPLETED SUCCESSFULLY ===")


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch FastAPI backend server."""
    print(f"🌐 Starting FastAPI backend on {args.host}:{args.port}...")
    cmd = [PYTHON_EXE, "-m", "uvicorn", "src.app:app", "--host", args.host, "--port", str(args.port), "--reload"]
    subprocess.run(cmd)


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch Streamlit web dashboard."""
    dashboard_path = os.path.join(SRC_DIR, "dashboard.py")
    print(f"📊 Starting Streamlit dashboard on port {args.port}...")
    cmd = [PYTHON_EXE, "-m", "streamlit", "run", dashboard_path, "--server.port", str(args.port)]
    subprocess.run(cmd)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Evaluate forecast accuracy and benchmark against Open-Meteo."""
    print("=== Offline Accuracy Benchmarking against Open-Meteo ===")
    run_script("evaluate.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Master CLI Orchestrator for Hyperlocal Weather Forecasting Pipeline.")
    subparsers = parser.add_subparsers(dest="command", help="Available CLI subcommands")

    # run-all
    p_all = subparsers.add_parser("run-all", help="Execute full end-to-end pipeline")
    p_all.add_argument("--epochs", type=int, default=30, help="Number of TFT training epochs (default: 30)")
    p_all.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    p_all.add_argument("--days", type=int, default=14, help="Collect external forecast archives for last N days")
    p_all.add_argument("--test", action="store_true", help="Test mode on single station")
    p_all.set_defaults(func=cmd_run_all)

    # collect
    p_coll = subparsers.add_parser("collect", help="Collect station observations and NWP synoptic forecasts")
    p_coll.add_argument("--source", choices=["sensors", "external", "all"], default="all", help="Data source to collect (default: all)")
    p_coll.add_argument("--days", type=int, default=14, help="Limit collection to the last N days")
    p_coll.add_argument("--test", action="store_true", help="Test run on station 8")
    p_coll.set_defaults(func=cmd_collect)

    # features
    p_feat = subparsers.add_parser("features", help="Compute meteorological features in data/processed/")
    p_feat.set_defaults(func=cmd_features)

    # train
    p_tr = subparsers.add_parser("train", help="Train all models (TFT, Residuals, Rain, PID)")
    p_tr.add_argument("--epochs", type=int, default=30, help="Number of TFT training epochs")
    p_tr.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p_tr.set_defaults(func=cmd_train)

    # serve
    p_srv = subparsers.add_parser("serve", help="Launch FastAPI REST API server")
    p_srv.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    p_srv.add_argument("--port", type=int, default=8000, help="Port number")
    p_srv.set_defaults(func=cmd_serve)

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch Streamlit frontend dashboard")
    p_dash.add_argument("--port", type=int, default=8501, help="Streamlit port number")
    p_dash.set_defaults(func=cmd_dashboard)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark model forecast accuracy")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
