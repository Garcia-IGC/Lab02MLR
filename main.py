from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import LabConfig


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos del orquestador principal."""

    parser = argparse.ArgumentParser(
        description=(
            "Laboratorio 02: clasificacion de genero con UTKFace. "
            "La regresion de edad queda como guia para estudiantes."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("../Lab2ML/UTKFace"),
        help=(
            "Ruta a la carpeta que contiene las imagenes de UTKFace. "
            "Por defecto se usa ./dataset"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Carpeta donde se guardaran los modelos, reportes y figuras.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(25, 25),
        help="Tamano del rostro preprocesado. Ejemplo: --img-size 25 25",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proporcion del conjunto de prueba.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para mantener resultados reproducibles.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        nargs="+",
        default=[30, 50, 80, 100, 150, 200],
        help="Lista de componentes PCA a evaluar.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limite opcional para pruebas rapidas con menos imagenes.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> LabConfig:
    """Construye el objeto de configuracion del laboratorio."""

    return LabConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        image_size=tuple(args.img_size),
        test_size=args.test_size,
        random_state=args.random_state,
        pca_components=tuple(args.pca_components),
        max_images=args.max_images,
    )


def save_metrics(metrics: dict[str, object], output_path: Path) -> None:
    """Guarda un diccionario JSON de metricas en disco."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Ejecuta el flujo completo del clasificador de genero."""

    args = parse_args()

    from src.classification import (
        evaluate_gender_classifier,
        save_gender_classifier,
        split_dataset,
        train_gender_classifier,
    )
    from src.data import build_dataset, dataset_to_dataframe
    from src.regression import (
        AgeRegressionGuide,
        evaluate_age_regressor,
        save_age_regressor,
        train_age_regressor,
    )
    from src.visualization import (
        save_confusion_matrix_figure,
        save_dataset_distribution_figure,
        save_pca_projection_figure,
    )

    config = build_config(args)
    config.ensure_output_dirs()

    print("[1/6] Cargando y preprocesando el dataset...")
    dataset = build_dataset(config)

    if len(dataset) == 0:
        raise RuntimeError(
            "No fue posible construir el dataset. "
            "Revise la ruta y el formato de nombres del dataset UTKFace."
        )

    print(
        f"    Muestras validas: {len(dataset)} | "
        f"Imagenes omitidas: {len(dataset.skipped_files)}"
    )

    dataset_df = dataset_to_dataframe(dataset)
    dataset_df.to_csv(config.reports_dir / "resumen_dataset.csv", index=False)
    save_dataset_distribution_figure(
        y_gender=dataset.y_gender,
        y_age=dataset.y_age,
        gender_map=config.gender_map,
        output_path=config.figures_dir / "distribucion_dataset.png",
    )

    print("[2/6] Separando entrenamiento y prueba...")
    split = split_dataset(
        dataset=dataset,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    print(
        "    Train:",
        split.X_train.shape,
        "| Test:",
        split.X_test.shape,
    )

    print("[3/6] Entrenando clasificador GaussianNB con PCA...")
    training_result = train_gender_classifier(
        split=split,
        pca_components=config.pca_components,
        random_state=config.random_state,
    )
    print(f"    Componentes PCA probados: {training_result.pca_components_tested}")
    print(f"    Mejor configuracion: {training_result.grid_search.best_params_}")

    print("[4/6] Evaluando clasificador de genero...")
    evaluation = evaluate_gender_classifier(
        model=training_result.best_estimator,
        split=split,
        best_params=training_result.grid_search.best_params_,
        best_cv_score=training_result.grid_search.best_score_,
    )

    print(
        "    Accuracy={:.4f} | Precision={:.4f} | Recall={:.4f} | F1={:.4f}".format(
            evaluation.accuracy,
            evaluation.precision,
            evaluation.recall,
            evaluation.f1,
        )
    )

    print("[5/6] Entrenando regresor de edad...")
    try:
        age_regressor = train_age_regressor(
            X_train=split.X_train,
            y_age_train=split.y_age_train,
            pca_components=config.pca_components,
            random_state=config.random_state,
        )
        age_metrics = evaluate_age_regressor(
            model=age_regressor,
            X_test=split.X_test,
            y_age_test=split.y_age_test,
        )
        print(
            "    MAE={:.2f} | RMSE={:.2f} | R2={:.4f}".format(
                age_metrics["MAE"],
                age_metrics["RMSE"],
                age_metrics["R2"],
            )
        )
    except Exception as e:
        print(f"    Error en regresion de edad: {e}")
        age_regressor = None
        age_metrics = None

    print("[6/6] Guardando artefactos...")
    save_gender_classifier(
        model=training_result.best_estimator,
        output_path=config.models_dir / "pipeline_genero.pkl",
    )
    save_metrics(
        metrics=evaluation.as_dict(),
        output_path=config.reports_dir / "metricas_genero.json",
    )

    if age_regressor is not None:
        save_age_regressor(
            model=age_regressor,
            output_path=config.models_dir / "pipeline_edad.pkl",
        )
        if age_metrics is not None:
            save_metrics(
                metrics=age_metrics,
                output_path=config.reports_dir / "metricas_edad.json",
            )
    save_confusion_matrix_figure(
        confusion_matrix=evaluation.confusion_matrix,
        labels=[
            config.gender_map.get(0, "Clase 0"),
            config.gender_map.get(1, "Clase 1"),
        ],
        output_path=config.figures_dir / "matriz_confusion_genero.png",
    )
    save_pca_projection_figure(
        X=split.X_train,
        y=split.y_gender_train,
        gender_map=config.gender_map,
        random_state=config.random_state,
        output_path=config.figures_dir / "proyeccion_pca_genero.png",
    )

    print("[7/7] Resumen de ejecucion...")
    print("Proceso completado exitosamente.")
    print(f"Modelo de genero: {config.models_dir / 'pipeline_genero.pkl'}")
    if age_regressor is not None:
        print(f"Modelo de edad: {config.models_dir / 'pipeline_edad.pkl'}")
    print(f"Metricas guardadas en: {config.reports_dir}")
    print(f"Figuras guardadas en: {config.figures_dir}")
    print("\nProximo paso: ejecuta la app visual con:")
    print("  streamlit run main_visual.py")


if __name__ == "__main__":
    main()
