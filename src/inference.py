from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.preprocessing import preprocess_face_array


@dataclass(slots=True)
class GenderPrediction:
    """Representa una prediccion de genero lista para mostrar al usuario."""

    label_id: int
    label_name: str


def load_gender_pipeline(model_path: str | Path) -> Any:
    """Carga desde disco el pipeline entrenado de clasificacion."""

    return joblib.load(model_path)


def predict_gender_from_face(
    face_array: np.ndarray,
    pipeline: Any,
    image_size: tuple[int, int],
    gender_map: dict[int, str],
) -> GenderPrediction:
    """Aplica el mismo preprocesamiento de entrenamiento antes de inferir."""

    vector, _ = preprocess_face_array(face_array, size=image_size)
    label_id = int(pipeline.predict([vector])[0])
    return GenderPrediction(
        label_id=label_id,
        label_name=gender_map.get(label_id, str(label_id)),
    )
