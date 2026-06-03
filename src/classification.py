from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.data import DatasetBundle


@dataclass(slots=True)
class DatasetSplit:
    """Contiene la separacion train/test para ambas tareas del laboratorio."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_gender_train: np.ndarray
    y_gender_test: np.ndarray
    y_age_train: np.ndarray
    y_age_test: np.ndarray
    fp_train: np.ndarray
    fp_test: np.ndarray


@dataclass(slots=True)
class TrainingResult:
    """Agrupa el resultado del ajuste por validacion cruzada."""

    grid_search: GridSearchCV
    best_estimator: Pipeline
    cv_folds: int
    pca_components_tested: tuple[int, ...]


@dataclass(slots=True)
class ClassificationMetrics:
    """Resume el comportamiento del clasificador sobre el conjunto de prueba."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    best_params: dict[str, Any] | None = None
    best_cv_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serializa las metricas a un diccionario listo para JSON."""

        serialized_params: dict[str, Any] | None = None
        if self.best_params is not None:
            serialized_params = {}
            for key, value in self.best_params.items():
                if hasattr(value, "item"):
                    serialized_params[key] = value.item()
                else:
                    serialized_params[key] = value

        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "best_params": serialized_params,
            "best_cv_score": (
                None if self.best_cv_score is None else float(self.best_cv_score)
            ),
        }


def split_dataset(
    dataset: DatasetBundle,
    test_size: float,
    random_state: int,
) -> DatasetSplit:
    """Separa el dataset manteniendo la proporcion de clases de genero."""

    split = train_test_split(
        dataset.X,
        dataset.y_gender,
        dataset.y_age,
        dataset.filepaths,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.y_gender,
    )
    return DatasetSplit(*split)


def build_gender_classifier(random_state: int) -> Pipeline:
    """Construye el pipeline principal: PCA seguido de GaussianNB."""

    return Pipeline(
        [
            ("pca", PCA(whiten=True, random_state=random_state)),
            ("clf", GaussianNB()),
        ]
    )


def resolve_cv_folds(y_train: np.ndarray, requested_cv: int = 5) -> int:
    """Ajusta la cantidad de folds al tamano real del conjunto de entrenamiento."""

    counts = np.bincount(y_train)
    valid_counts = counts[counts > 0]
    if valid_counts.size == 0 or int(valid_counts.min()) < 2:
        raise ValueError(
            "Se requieren al menos dos muestras por clase para usar validacion cruzada."
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


def train_gender_classifier(
    split: DatasetSplit,
    pca_components: tuple[int, ...],
    random_state: int,
    requested_cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 1,
) -> TrainingResult:
    """Entrena el clasificador de genero con validacion cruzada."""

    cv_folds = resolve_cv_folds(split.y_gender_train, requested_cv=requested_cv)
    safe_components = resolve_pca_components(
        candidates=pca_components,
        X_train=split.X_train,
        cv_folds=cv_folds,
    )

    pipeline = build_gender_classifier(random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={"pca__n_components": safe_components},
        scoring="f1",
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    grid_search.fit(split.X_train, split.y_gender_train)

    return TrainingResult(
        grid_search=grid_search,
        best_estimator=grid_search.best_estimator_,
        cv_folds=cv_folds,
        pca_components_tested=safe_components,
    )


def evaluate_gender_classifier(
    model: Pipeline,
    split: DatasetSplit,
    best_params: dict[str, Any] | None = None,
    best_cv_score: float | None = None,
) -> ClassificationMetrics:
    """Calcula las metricas clasicas del laboratorio."""

    y_pred = model.predict(split.X_test)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(split.y_gender_test, y_pred)),
        precision=float(precision_score(split.y_gender_test, y_pred, zero_division=0)),
        recall=float(recall_score(split.y_gender_test, y_pred, zero_division=0)),
        f1=float(f1_score(split.y_gender_test, y_pred, zero_division=0)),
        confusion_matrix=confusion_matrix(split.y_gender_test, y_pred),
        best_params=best_params,
        best_cv_score=None if best_cv_score is None else float(best_cv_score),
    )


def save_gender_classifier(model: Pipeline, output_path: str | Path) -> None:
    """Guarda el pipeline completo para reutilizarlo luego en inferencia."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
