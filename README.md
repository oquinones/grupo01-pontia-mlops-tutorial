# grupo01-pontia-mlops-tutorial

Pipeline de entrenamiento de un modelo de clasificación sobre el datase **Adult Income**, con seguimiento de experimentos en **MLflow**,
registro del modelo en **Azure Machine Learning** y automatización con **GitHub Actions**.

Proyecto práctico del Módulo 4 (Introducción a DevOps / MLOps) del Máster en IA de PontIA.

## Colaboradores
- Mary León.
- Orlando Quiñones.
- Limberg Rivera.

## Objetivo

Predecir si una persona gana más de 50K $/año (`income > 50K`) a partir de variables demográficas y laborales, usando un `RandomForestClassifier`.

## Estructura


```
.
├── .github/
│   └── workflows/
│       ├── build.yml        # CI: entrena y registra el modelo en cada push a main
│       └── integration.yml  # CI: tests + cobertura en cada pull request al repo
|       └── deploy.yml       # CD: empaqueta y despliega la app en Azure Container Apps
├── src/
│   ├── data_loader.py       # Carga, limpieza, LabelEncoder + StandardScaler
│   ├── model.py             # Entrenamiento del RandomForestClassifier
│   ├── evaluate.py          # Accuracy y classification report
│   └── main.py              # Orquestación + tracking y registro en MLflow
├── deployment/              # Archivos para Docker y despliegue
├── scripts/
│   └── download_model.py    # Descarga la última versión del modelo registrado
├── tests/                   # Tests unitarios (pytest)
├── data/raw/                # Dataset (no versionado en el repo)
├── models/                  # Artefactos generados (no versionado en el repo)
├── pyproject.toml           # Dependencias y metadatos (uv)
├── requirements.txt         # Dependencias para pip (las que usa la CI)
├── ruff.toml                # Linter y formateador
└── pytest.ini
```

## Requisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (recomendado) o `pip`
- Un servidor de tracking MLflow accesible: local (`http://localhost:5000`) 
  o el tracking URI de un workspace de Azure ML.

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

## Configuración

Los scripts leen la configuración de variables y secrets de entorno. En GitHub Actions (Settings → Environments → Staging) debes configurar los siguientes parámetros:


| Tipo | Nombre | Descripción | Valor por defecto / Ejemplo | Workflow |
|---|---|---|---|---|
| **Secret** | `DOCKERHUB_TOKEN` | Token de acceso para autenticarse y subir imágenes a Docker Hub | *Confidencial* | `deploy.yml` |
| **Secret** | `DOCKERHUB_USERNAME` | Nombre de usuario de Docker Hub | *Confidencial* | `deploy.yml` |
| **Secret** | `AZURE_SUBSCRIPTION_ID` | ID de la suscripción de Azure para la Container App | *Confidencial* | `deploy.yml` |
| **Secret** | `AZURE_TENANT_ID` | Tenant ID del service principal (solo Azure ML) | *Confidencial* | `integration.yml`, `build.yml`, `deploy.yml` |
| **Secret** | `AZURE_CLIENT_ID` | Client ID del service principal (solo Azure ML) | *Confidencial* | `integration.yml`, `build.yml`, `deploy.yml` |
| **Secret** | `AZURE_CLIENT_SECRET` | Secreto del service principal (solo Azure ML) | *Confidencial* | `build.yml`, `integration.yml` |
| **Variable** | `DOCKERHUB_REPO` | Repositorio o namespace dentro de Docker Hub | `tu-usuario` | `deploy.yml` |
| **Variable** | `CONTAINER_APP` | Nombre de la aplicación contenedora en Azure | `grupo01-devops-0926` | `deploy.yml` |
| **Variable** | `RESOURCE_GROUP` | Nombre del grupo de recursos de Azure | `rg-albertog-devops` | `deploy.yml` |
| **Variable** | `IMAGE_NAME` | Nombre base de la imagen del contenedor | `model-app` | `deploy.yml` |
| **Variable** | `MLFLOW_URL` | Tracking URI de MLflow (local o Azure ML) | `http://localhost:5000/` | `integration.yml`, `build.yml` |
| **Variable** | `EXPERIMENT_NAME` | Nombre del experimento y del modelo registrado | `grupo01-adult-income` | `integration.yml`, `build.yml` |


Para trabajar contra Azure ML en local también basta con autenticarse con `az login` en lugar de definir las variables `AZURE_*`.

## Datos

Descarga el dataset [Adult](https://archive.ics.uci.edu/dataset/2/adult) y coloca los ficheros en `data/raw/`:

```
data/raw/adult.data
data/raw/adult.test
```

🔴**ELIMINAR ->** 

Desde la línea de comandos (es lo mismo que hace la CI):

```bash
mkdir -p data/raw
curl -fL "https://archive.ics.uci.edu/static/public/20/census+income.zip" -o census-income.zip
unzip -o census-income.zip -d data/raw && rm census-income.zip
```

La carpeta `data/` está en `.gitignore` y no se versiona.

## Uso

### 1. Levantar MLflow en local

Solo si no se usa Azure ML como servidor de tracking:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

### 2. Entrenar el modelo

```bash
uv run python src/main.py
```

🔴**ELIMINAR ->** El script:

1. Carga y preprocesa los datos (elimina nulos, codifica categóricas y escala).
2. Entrena el `RandomForestClassifier` (`random_state=42`).
3. Evalúa sobre el conjunto de test y registra accuracy y classification report en el log.
4. Guarda en `models/` los ficheros `model.pkl`, `scaler.pkl` y `encoders.pkl`.
5. Registra en MLflow, dentro del experimento `EXPERIMENT_NAME`, el modelo
   (`model/`) y los artefactos de preprocesado (`preprocessing/`).
6. Registra el modelo en el *Model Registry* con el nombre `EXPERIMENT_NAME`.
7. Escribe el ID del run en `run_id.txt` y el log en `training.log`.

### 3. Descargar el modelo registrado

```bash
uv run python scripts/download_model.py
```
Descarga la última versión del modelo `models:/<EXPERIMENT_NAME>/latest` del
servidor indicado en `MLFLOW_URL` a `./models/model/`.

## Tests

```bash
uv run pytest --cov=src
```

>**Nota**: `tests/test_model.py` busca el archivo del modelo en `./models/model/model.pkl` (que es donde lo descarga el script de MLflow). Al entrenar el modelo localmente, este se guarda en `./models/model.pkl`, es recomendable descargar el modelo registrado antes de lanzar los tests en local.

## Calidad de código

```bash
uv run ruff check .
uv run ruff format .
```

## Ciclo MLOps y Despliegue Continuo (GitHub Actions)
## CI - Integración continua (GitHub Actions)

El repositorio define tres workflows automatizados en [.github/workflows/](.github/workflows/)
que forman un ciclo de MLOps: **la integración valida cada cambio contra el último modelo registrado**
y **el build reentrena y registra un modelo nuevo** cuando el cambio llega a `main`.

```
 Pull request ──► integration.yml ──► descarga modelo (Azure ML)
                                      descarga dataset (UCI)
                                      pytest + cobertura
                                      comentario en la PR con resultados
       │
       ▼ merge
 Push a main ───► build.yml ────────► descarga dataset (UCI)
                                      entrena + registra modelo (Azure ML)
                                      publica model.pkl como artefacto
       │  
       ▼ automatizado / secuencial en main
 Push a main ───► deploy.yml ───────► construye imagen Docker y la sube a Docker Hub
                                     autenticación en Azure vía OIDC
                                     verifica y configura secretos / variables en Azure Container Apps
                                     configura registro privado (Docker Hub)
                                     despliega la nueva imagen en Azure Container Apps
```


Ambos workflows se ejecutan en `ubuntu-latest` con Python 3.12, instalan las dependencias desde `requirements.txt` 
y usan el environment **`Staging`** de GitHub.

### Integration ([integration.yml](.github/workflows/integration.yml))

- **Disparador:** cualquier `pull_request`.
- **Permisos:** `contents: read`, `pull-requests: write`.
- **Pasos:**
  1. Checkout del código y configuración de Python 3.12.
  2. Instalación de dependencias con `pip`.
  3. **Download Model:** ejecuta `scripts/download_model.py` para traer la
     última versión del modelo registrado en Azure ML a `models/model/`.
  4. **Download and extract dataset:** descarga `census+income.zip` de UCI y
     lo descomprime en `data/raw/`.
  5. **Run tests:** ejecuta `pytest` con `PYTHONPATH=./src`, cobertura
     (informe en terminal y HTML) y resultados JUnit en
     `test-results/results.xml`. La salida se guarda en
     `test-results/results.log`.
  6. **Comment test results and coverage on PR:** con `actions/github-script` publica el
     contenido de `results.log` (truncado a 65 000 caracteres) como
     comentario en la PR. Se ejecuta siempre (`if: always()`), también
     cuando fallan los tests.

### Build Model ([build.yml](.github/workflows/build.yml))

- **Disparador:** `push` a la rama `main`.
- **Permisos:** `contents: write`.
- **Pasos:**
  1. Checkout del código y configuración de Python 3.12.
  2. Instalación de dependencias con `pip`.
  3. **Download and extract dataset:** igual que en la integración.
  4. **Train and save model:** ejecuta `src/main.py`, que entrena el modelo,
     lo registra en MLflow / Azure ML y genera los `.pkl` en `models/`.
  5. **Publicar artefactos:** sube `models/model.pkl` como artefacto del run
     (`model.pkl`), con una retención de 30 días.



El service principal necesita permisos sobre el workspace de Azure ML para
registrar runs y modelos (build) y para descargarlos (integración).

> **Nota:** como la integración descarga el modelo `latest` del registro,
> el workflow `build.yml` debe haberse ejecutado al menos una vez (o el
> modelo debe haberse registrado manualmente) antes de que los tests de las
> PRs puedan pasar.


 ### Build and Deploy Azure Container Apps ([deploy.yml](.github/workflows/deploy.yml))

- **Disparador:** `push` a la rama `main` o ejecución manual (`workflow_dispatch`).
- **Permisos:** `contents: read` y `id-token: write` (necesario para la autenticación segura con Azure mediante OIDC).
- **Pasos:**
  1. **Checkout repository:** descarga el código fuente del repositorio.
  2. **Setup Docker Buildx:** configura Docker Buildx para permitir compilaciones multiplataforma avanzadas.
  3. **Login to Docker Hub:** inicia sesión en Docker Hub utilizando los secretos del repositorio (`DOCKERHUB_USERNAME` y `DOCKERHUB_TOKEN`).
  4. **Build and push image:** construye la imagen Docker usando el contexto y Dockerfile especificados en `./deployment/` y la sube a Docker Hub con la etiqueta `:latest`.
  5. **Login to Azure:** se autentica en Azure de forma segura utilizando OpenID Connect (OIDC) y los secretos de la suscripción.
  6. **Verify Container App:** ejecuta un comando de Azure CLI para comprobar que la Container App existe y recuperar su configuración.
  7. **Configure application secrets:** configura y guarda los secretos (tenant, client ID, client secret) y establece las variables de entorno necesarias (como `MLFLOW_TRACKING_URI`) directamente en Azure Container Apps.
  8. **Configure private Docker Hub registry:** configura las credenciales dentro de Azure para que la aplicación pueda descargar imágenes privadas desde Docker Hub si lo necesita.
  9. **Deploy to Azure Container Apps:** actualiza la aplicación en Azure Container Apps para desplegar y ejecutar la nueva versión de la imagen subida a Docker Hub.