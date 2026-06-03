# Lab02-ML-2026-01

Codigo modular para el Laboratorio 02 de Machine Learning:

- clasificacion de genero con `GaussianNB`
- guia de regresion de edad para estudiantes
- base de despliegue minima con `Streamlit`

La implementacion sigue la idea del PDF y reutiliza la logica central del notebook de referencia: carga del dataset UTKFace, preprocesamiento clasico de rostros, vectorizacion, PCA y clasificacion probabilistica.

## Objetivo didactico

El proyecto esta pensado para que los alumnos puedan:

1. entender un flujo completo de Machine Learning,
2. leer codigo modular en vez de un solo notebook,
3. entrenar y evaluar un clasificador de genero,
4. completar por su cuenta la regresion de edad,
5. preparar el salto hacia una aplicacion visual.

## Estructura del proyecto

```text
.
├── dataset
│   └── .gitkeep
├── main.py
├── main_visual.py
├── requirements.txt
└── src
    ├── __init__.py
    ├── classification.py
    ├── config.py
    ├── data.py
    ├── inference.py
    ├── preprocessing.py
    ├── regression.py
    ├── streamlit_app.py
    └── visualization.py
```

## Que hace cada modulo

- `main.py`: orquesta la ejecucion completa del laboratorio desde consola.
- `main_visual.py`: punto de entrada para `Streamlit`.
- `src/config.py`: parametros globales del experimento.
- `src/data.py`: lectura de archivos, parseo de etiquetas y construccion del dataset.
- `src/preprocessing.py`: conversion a grises, resize, ecualizacion, mascara oval y vectorizacion.
- `src/classification.py`: separacion train/test, entrenamiento con `PCA + GaussianNB`, evaluacion y guardado.
- `src/regression.py`: interfaces comentadas para que los estudiantes implementen la regresion de edad.
- `src/inference.py`: utilidades para reutilizar el preprocesamiento y hacer inferencia de genero.
- `src/visualization.py`: figuras para distribucion de datos, matriz de confusion y proyeccion PCA.
- `src/streamlit_app.py`: app minima para cargar una imagen y dejar marcado donde integrar detector de caras y modelos.

## Flujo implementado en `main.py`

1. leer las imagenes de UTKFace,
2. extraer `edad` y `genero` desde el nombre del archivo,
3. preprocesar cada rostro,
4. construir la matriz `X`,
5. separar entrenamiento y prueba,
6. ajustar `PCA + GaussianNB` con `GridSearchCV`,
7. evaluar con `accuracy`, `precision`, `recall`, `f1` y matriz de confusion,
8. guardar el mejor pipeline y reportes.

## Lo que si esta implementado

- clasificador completo de genero con `GaussianNB`
- seleccion de componentes PCA
- evaluacion sobre conjunto de prueba
- guardado del pipeline entrenado
- figuras de apoyo para analisis

## Lo que deben completar los estudiantes

- `src/regression.py`
- integracion de un detector de caras para inferencia sobre imagenes nuevas
- prediccion de edad en la app visual

La app de `Streamlit` queda deliberadamente minima para que el curso complete esa parte en la fase de deployment.

## Requisitos

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Copiar las imagenes de UTKFace dentro de `dataset/`. El codigo recorre esa carpeta de forma recursiva, por lo que tambien funciona si el dataset viene organizado en subcarpetas.

## Ejecucion del laboratorio

Ejemplo general:

```bash
python main.py
```

El comando anterior usa por defecto la carpeta local `dataset/`.

Ejemplo indicando otra ruta:

```bash
python main.py --dataset-dir "/ruta/a/UTKFace"
```

Ejemplo rapido con menos imagenes:

```bash
python main.py --max-images 200
```

Opciones utiles:

- `--output-dir`: carpeta donde se guardan modelos, reportes y figuras.
- `--img-size WIDTH HEIGHT`: tamano del rostro preprocesado.
- `--pca-components 30 50 80 100 150 200`: lista de componentes a evaluar.
- `--max-images`: limite opcional para pruebas rapidas.

## Archivos generados al entrenar

Dentro de `artifacts/` se generan subcarpetas como estas:

- `artifacts/models/pipeline_genero.pkl`
- `artifacts/reports/metricas_genero.json`
- `artifacts/reports/resumen_dataset.csv`
- `artifacts/figures/distribucion_dataset.png`
- `artifacts/figures/matriz_confusion_genero.png`
- `artifacts/figures/proyeccion_pca_genero.png`

## App visual minima

Para abrir la version minima en `Streamlit`:

```bash
streamlit run main_visual.py
```

La interfaz actual:

- permite cargar una imagen,
- muestra la fotografia cargada,
- explica donde integrar el detector de caras,
- deja marcado donde cargar los modelos.

## Detector de caras

El PDF indica que la etapa final debe detectar rostros antes de inferir. Esa parte no esta automatizada todavia en esta version minima. Los estudiantes deben agregar:

1. un detector de caras,
2. el recorte de cada rostro detectado,
3. la reutilizacion del mismo preprocesamiento,
4. la llamada a los modelos entrenados.

## Nota sobre la regresion

Por requerimiento del laboratorio, la regresion de edad no se implementa por completo aqui. En su lugar se dejan interfaces comentadas, docstrings y sugerencias de trabajo para que los alumnos construyan esa parte.
