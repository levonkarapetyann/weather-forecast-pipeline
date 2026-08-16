# Hyperlocal Weather Forecasting Pipeline

Кодовая база для гибридного прогноза погоды: Temporal Fusion Transformer (TFT)
в связке с CatBoost-ансамблями для коррекции остатков и отдельным классификатором осадков.

**Кратко:** проект собирает сырые данные с метеостанций, подготавливает окна временных рядов,
обучает TFT-модель и ансамбль CatBoost для исправления систематических ошибок, а также
предоставляет FastAPI-сервис и Streamlit-дашборд для развёртывания прогнозов.

**Быстрый старт**
- **Требования:** Python 3.8+ и доступ в интернет для скачивания зависимостей.
- Установите зависимости:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Создайте и заполните конфиги в каталоге `config/` (см. `config/settings.json` и `config/stations.json`).
- При необходимости добавьте переменные окружения в `.env` (опционально). Пример переменной: `CLIMATENET_API_BASE_URL`.

**Запуск API**
- Локально (разработка):

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

**Запуск Streamlit-дашборда**
- Локально:

```bash
streamlit run src/dashboard.py
```

**Обучение модели**
- Запустить тренировочный скрипт:

```bash
python src/train.py --help
python src/train.py --epochs 30 --batch_size 64
```

**Сбор данных**
- Сырые данные с ClimateNet собираются через `src/data_fetcher.py`.

```bash
python src/data_fetcher.py --days 30
```

**Запуск тестов**

```bash
pytest tests
```

**Структура репозитория (основные файлы)**
- `src/app.py` — FastAPI-сервис API.
- `src/dashboard.py` — Streamlit интерфейс.
- `src/train.py` — скрипт обучения TFT.
- `src/model.py` — реализация архитектуры TFT и датасета.
- `src/data_pipeline.py` — подготовка данных и утилиты работы со скейлерами.
- `src/residual_engine.py` — CatBoost residual-модели и генерация признаков.
- `config/` — конфигурации: `settings.json`, `stations.json`, `scalers.json`.
- `models/` — сохранённые веса и артефакты моделей.

Если нужно, могу расширить README с примерами конфигов, подробным описанием форматов данных
и командами для развёртывания в Docker или облаке.
