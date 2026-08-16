#!/usr/bin/env python3
"""
=============================================================================
МОДУЛЬ: Очистка скачанных данных Meteostat (tests/clean_meteostat_data.py)
-----------------------------------------------------------------------------
НАЗНАЧЕНИЕ:
Удаляет все ранее сохраненные файлы прогнозов Meteostat (meteostat_*.csv)
из папки data/raw/external_forecasts/ и локальные кэш-файлы.
=============================================================================
"""

import glob
import os
import sys

# Настройка путей
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)


def clean_meteostat_files():
    target_dir = os.path.join("data", "raw", "external_forecasts")
    if not os.path.exists(target_dir):
        print(f"Директория {target_dir} не найдена.")
        return

    pattern = os.path.join(target_dir, "meteostat_*.csv")
    files_to_remove = glob.glob(pattern)

    if not files_to_remove:
        print(f"✅ Файлы Meteostat не найдены в {target_dir}.")
        return

    total_bytes = 0
    removed_count = 0

    print(f"🗑️ Найдено файлов Meteostat для удаления: {len(files_to_remove)}")
    for fpath in files_to_remove:
        try:
            size = os.path.getsize(fpath)
            os.remove(fpath)
            total_bytes += size
            removed_count += 1
            print(f"  • Удален: {os.path.basename(fpath)} ({size / 1024:.1f} KB)")
        except Exception as e:
            print(f"  ❌ Ошибка при удалении {fpath}: {e}")

    mb_freed = total_bytes / (1024 * 1024)
    print(f"\n✨ Успешно удалено {removed_count} файлов Meteostat. Освобождено {mb_freed:.2f} MB дискового пространства.")


if __name__ == "__main__":
    clean_meteostat_files()
