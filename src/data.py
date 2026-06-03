from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import LabConfig
from src.preprocessing import preprocess_image_path


@dataclass(slots=True)
class DatasetBundle:
    """Agrupa la matriz principal del laboratorio y sus etiquetas."""

    X: np.ndarray
    y_gender: np.ndarray
    y_age: np.ndarray
    filepaths: np.ndarray
    skipped_files: tuple[str, ...] = ()

    def __len__(self) -> int:
        return int(self.X.shape[0])


def parse_filename(filepath: str | Path) -> tuple[int, int]:
    """Extrae edad y genero desde nombres tipo edad_genero_raza_fecha.jpg."""

    name = Path(filepath).stem
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Formato de nombre invalido: {filepath}")

    try:
        age = int(parts[0])
        gender = int(parts[1])
    except ValueError as error:
        raise ValueError(f"No fue posible leer edad y genero desde: {filepath}") from error

    return age, gender


def iter_image_files(root_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Recorre el dataset y devuelve las imagenes compatibles ordenadas."""

    if not root_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {root_dir}")

    extensions_normalized = {extension.lower() for extension in extensions}
    image_files = [
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions_normalized
    ]
    image_files.sort()
    return image_files


def build_dataset(config: LabConfig) -> DatasetBundle:
    """Construye la matriz `X` del laboratorio a partir del dataset UTKFace."""

    X: list[np.ndarray] = []
    y_gender: list[int] = []
    y_age: list[int] = []
    filepaths: list[str] = []
    skipped_files: list[str] = []

    image_files = iter_image_files(config.dataset_dir, config.image_extensions)
    if config.max_images is not None:
        image_files = image_files[: config.max_images]

    for image_path in image_files:
        try:
            age, gender = parse_filename(image_path)
            x_vector, _ = preprocess_image_path(image_path, size=config.image_size)
        except Exception:
            skipped_files.append(str(image_path))
            continue

        X.append(x_vector)
        y_gender.append(gender)
        y_age.append(age)
        filepaths.append(str(image_path))

    if not X:
        return DatasetBundle(
            X=np.empty((0, config.image_size[0] * config.image_size[1]), dtype=np.float32),
            y_gender=np.empty((0,), dtype=np.int32),
            y_age=np.empty((0,), dtype=np.int32),
            filepaths=np.empty((0,), dtype=str),
            skipped_files=tuple(skipped_files),
        )

    return DatasetBundle(
        X=np.asarray(X, dtype=np.float32),
        y_gender=np.asarray(y_gender, dtype=np.int32),
        y_age=np.asarray(y_age, dtype=np.int32),
        filepaths=np.asarray(filepaths),
        skipped_files=tuple(skipped_files),
    )


def dataset_to_dataframe(dataset: DatasetBundle) -> pd.DataFrame:
    """Convierte el dataset a una tabla util para exportar o inspeccionar."""

    return pd.DataFrame(
        {
            "filepath": dataset.filepaths,
            "gender": dataset.y_gender,
            "age": dataset.y_age,
        }
    )
