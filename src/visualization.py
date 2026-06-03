from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay


def save_dataset_distribution_figure(
    y_gender: np.ndarray,
    y_age: np.ndarray,
    gender_map: dict[int, str],
    output_path: str | Path,
) -> None:
    """Guarda una figura simple con distribucion de genero y edad."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(int(label) for label in np.unique(y_gender))
    counts = [int(np.sum(y_gender == label)) for label in unique_labels]
    labels = [gender_map.get(label, str(label)) for label in unique_labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, counts, color=["#4C78A8", "#F58518"][: len(labels)])
    axes[0].set_title("Distribucion de genero")
    axes[0].set_ylabel("Frecuencia")

    axes[1].hist(y_age, bins=20, color="#54A24B", edgecolor="black")
    axes[1].set_title("Distribucion de edad")
    axes[1].set_xlabel("Edad")
    axes[1].set_ylabel("Frecuencia")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_figure(
    confusion_matrix: np.ndarray,
    labels: list[str],
    output_path: str | Path,
) -> None:
    """Guarda la matriz de confusion del clasificador."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusion - genero")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pca_projection_figure(
    X: np.ndarray,
    y: np.ndarray,
    gender_map: dict[int, str],
    random_state: int,
    output_path: str | Path,
) -> None:
    """Proyecta el conjunto de entrenamiento a 2D para una visualizacion simple."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    projector = PCA(n_components=2, random_state=random_state)
    projected = projector.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label in sorted(int(value) for value in np.unique(y)):
        mask = y == label
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            label=gender_map.get(label, str(label)),
            alpha=0.65,
            s=18,
        )

    ax.set_title("Proyeccion PCA del conjunto de entrenamiento")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
