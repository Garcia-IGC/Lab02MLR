from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


@dataclass(slots=True)
class AgeRegressionGuide:
    """Resume los pasos que deben seguir los estudiantes para la regresion."""

    scoring: str = "neg_mean_absolute_error"
    suggested_metrics: tuple[str, ...] = ("MAE", "RMSE", "R2")

    def to_text(self) -> str:
        """Genera un recordatorio corto para acompanar el laboratorio."""

        return (
            "Guia de regresion de edad\n"
            "========================\n"
            "\n"
            "La parte de regresion no se implementa en esta version.\n"
            "Los estudiantes deben completar src/regression.py siguiendo estos pasos:\n"
            "\n"
            "1. Reutilizar la misma matriz X preprocesada.\n"
            "2. Construir un Pipeline con PCA + LinearRegression.\n"
            "3. Ajustar pca__n_components con GridSearchCV.\n"
            "4. Evaluar con MAE, RMSE y R2.\n"
            "5. Guardar el modelo cuando este listo.\n"
        )


def build_age_regression_pipeline(random_state: int) -> Pipeline:
    """Interfaz guia para la regresion de edad.

    Construye un pipeline con PCA + LinearRegression siguiendo el mismo
    patron que el clasificador de género.
    """

    return Pipeline(
        [
            ("pca", PCA(whiten=True, random_state=random_state)),
            ("reg", LinearRegression()),
        ]
    )


def resolve_cv_folds(y_train: np.ndarray, requested_cv: int = 5) -> int:
    """Ajusta la cantidad de folds al tamano real del conjunto de entrenamiento."""

    counts = np.bincount(y_train.astype(int))
    valid_counts = counts[counts > 0]
    if valid_counts.size == 0 or int(valid_counts.min()) < 2:
        raise ValueError(
            "Se requieren al menos dos muestras por target para usar validacion cruzada."
        )
    return min(requested_cv, int(valid_counts.min()))


def resolve_pca_components(
    candidates: tuple[int, ...],
    X_train: np.ndarray,
    cv_folds: int,
) -> tuple[int, ...]:
    """Filtra los componentes PCA a valores seguros para el tamano de los folds."""

    n_samples, n_features = X_train.shape
    smallest_train_fold_size = n_samples - math.ceil(n_samples / cv_folds)
    max_allowed = min(n_features, smallest_train_fold_size)

    if max_allowed < 1:
        raise ValueError("No hay suficientes muestras para ajustar PCA.")

    valid_candidates = tuple(
        component for component in candidates if 1 <= component <= max_allowed
    )
    if valid_candidates:
        return valid_candidates

    return (max_allowed,)


def train_age_regressor(
    X_train: Any,
    y_age_train: Any,
    pca_components: tuple[int, ...],
    random_state: int,
    requested_cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 1,
) -> Any:
    """Interfaz guia para ajustar el regresor de edad.

    Crea un pipeline, configura GridSearchCV con neg_mean_absolute_error
    como scoring y retorna el mejor estimador encontrado.
    """

    cv_folds = resolve_cv_folds(y_age_train, requested_cv=requested_cv)
    safe_components = resolve_pca_components(
        candidates=pca_components,
        X_train=X_train,
        cv_folds=cv_folds,
    )

    pipeline = build_age_regression_pipeline(random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={"pca__n_components": safe_components},
        scoring="neg_mean_absolute_error",
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    grid_search.fit(X_train, y_age_train)

    return grid_search.best_estimator_


def evaluate_age_regressor(model: Any, X_test: Any, y_age_test: Any) -> dict[str, float]:
    """Interfaz guia para calcular metricas de regresion.

    Obtiene predicciones y calcula MAE, RMSE y R2.
    """

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_age_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_age_test, y_pred)))
    r2 = float(r2_score(y_age_test, y_pred))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def save_age_regressor(model: Any, output_path: str | Path) -> None:
    """Interfaz guia para guardar el modelo de edad.

    Guarda el pipeline completo con joblib.
    """

    joblib.dump(model, output_path)
