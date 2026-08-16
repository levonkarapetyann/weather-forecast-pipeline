import subprocess
import sys
import time
import psutil

def is_train_running():
    """Проверяет, запущен ли процесс train.py прямо сейчас."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline and any("train.py" in arg for arg in cmdline):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def wait_for_train_and_push():
    print("🔍 Поиск запущенного процесса train.py...")
    pid = is_train_running()

    if pid:
        print(f"⏳ Найден процесс train.py (PID: {pid}). Ожидаю завершения обучения...")
        while is_train_running() == pid:
            time.sleep(15)  # проверяем каждые 15 секунд
        print("✅ Процесс train.py завершился!")
    else:
        print("ℹ️ Процесс train.py не найден (возможно, обучение уже закончилось).")

    # Выполнение Git команд
    print("🚀 Выполняю Git commit & push...")
    try:
        commit_msg = "feat: auto-commit after successful TFT training"
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push", "origin", "TFT/CatBoost"], check=True)
        print("🎉 Изменения и веса успешно отправлены в репозиторий!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Ошибка при работе с Git (или нет новых изменений): {e}")

if __name__ == "__main__":
    wait_for_train_and_push()
