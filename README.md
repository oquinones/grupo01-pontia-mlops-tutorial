# grupo01-pontia-mlops-tutorial

Pipeline de entrenamiento de un modelo de clasificación sobre el dataset
**Adult Income** (UCI), con seguimiento de experimentos en **MLflow** e
integración con **Azure Machine Learning**.

Proyecto práctico del Módulo 4 (Introducción a DevOps / MLOps) del Máster en IA de PontIA.

## Objetivo

Predecir si una persona gana más de 50K $/año (`income > 50K`) a partir de variables demográficas y laborales, usando un `RandomForestClassifier`.

## Estructura

```
.
├── src/
│   ├── data_loader.py      # Carga del CSV, limpieza, LabelEncoder + StandardScaler
│   ├── model.py            # Entrenamiento del RandomForestClassifier
│   ├── evaluate.py         # Accuracy y classification report
│   └── main.py             # Orquestación + tracking y registro en MLflow
├── scripts/
│   └── download_model.py   # Descarga el último modelo registrado en Azure ML
├── tests/                  # Tests unitarios (pytest)
├── data/raw/               # Dataset (no versionado)
├── models/                 # Artefactos generados (no versionado)
├── pyproject.toml          # Dependencias y metadatos (uv)
├── requirements.txt        # Dependencias para pip
├── ruff.toml               # Linter y formateador
└── pytest.ini
```

## Requisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (recomendado) o `pip`
- Un servidor MLflow accesible (por defecto `http://localhost:5000`)

## Instalación

Con uv:

```bash
uv sync
```

Con pip:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Datos

Descarga el dataset [Adult](https://archive.ics.uci.edu/dataset/2/adult) y
coloca los ficheros en `data/raw/`:

```
data/raw/adult.data
data/raw/adult.test
```

La carpeta `data/` está en `.gitignore` y no se versiona.

## Uso

### 1. Levantar MLflow en local

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

### 2. Entrenar el modelo

```bash
uv run python src/main.py
```

El script:

1. Carga y preprocesa los datos (elimina nulos, codifica categóricas y escala).
2. Entrena el `RandomForestClassifier` (`random_state=42`).
3. Evalúa sobre el conjunto de test y registra las métricas en el log.
4. Guarda en `models/` los ficheros `model.pkl`, `scaler.pkl` y `encoders.pkl`.
5. Registra el modelo y los artefactos de preprocesado en MLflow, en el
   experimento `adult-income`.
6. Escribe el ID del run en `run_id.txt` y el log en `training.log`.

### 3. Descargar el modelo desde Azure ML

```bash
uv run python scripts/download_model.py
```

Descarga la última versión del modelo registrado a `./models`. Hace falta
estar autenticado en Azure (por ejemplo con `az login`) y tener acceso al
workspace configurado en el script.

## Tests

```bash
uv run pytest --cov=src
```

> `tests/test_model.py` espera un modelo en `models/model/model.pkl`.
> Ejecuta antes `scripts/download_model.py` o entrena el modelo localmente.

## Calidad de código

```bash
uv run ruff check .
uv run ruff format .
```
