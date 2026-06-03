from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

from src.config import LabConfig
from src.preprocessing import preprocess_face_array


def load_models(config: LabConfig) -> tuple[object, object]:
    """Carga los modelos de género y edad desde disco."""

    gender_model_path = config.models_dir / "pipeline_genero.pkl"
    age_model_path = config.models_dir / "pipeline_edad.pkl"

    gender_model = joblib.load(gender_model_path)

    # Cargar el modelo de edad si existe, si no retornar None
    age_model = None
    if age_model_path.exists():
        age_model = joblib.load(age_model_path)

    return gender_model, age_model


def apply_nms(boxes: np.ndarray, overlap_thresh: float = 0.3) -> list[tuple[int, int, int, int]]:
    """Aplica Non-Maximum Suppression para eliminar detecciones solapadas."""
    
    if len(boxes) == 0:
        return []
    
    # Convertir a float si es necesario
    boxes = boxes.astype(float)
    
    # Calcular coordenadas de las esquinas
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    
    # Calcular áreas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Ordenar por área (de mayor a menor)
    order = np.argsort(areas)[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(boxes[i].astype(int))
        
        if len(order) == 1:
            break
        
        # Calcular IoU con resto
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        # Mantener solo si IoU < threshold
        order = order[np.where(iou <= overlap_thresh)[0] + 1]
    
    return [(int(box[0]), int(box[1]), int(box[2]), int(box[3])) for box in keep]


def detect_faces_in_image(image_array: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detecta rostros usando MediaPipe (fallback a Haar Cascade si no está disponible)."""

    # Asegurar que la imagen está en RGB
    if len(image_array.shape) == 2:
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = image_array

    # Intentar usar MediaPipe si está disponible
    if MEDIAPIPE_AVAILABLE and mp is not None:
        try:
            mp_face_detection = mp.solutions.face_detection
            with mp_face_detection.FaceDetection(
                model_selection=1,  # 1 para mayor rango de distancia
                min_detection_confidence=0.5
            ) as face_detector:
                results = face_detector.process(image_rgb)

            faces = []
            if results.detections:
                h, w, _ = image_rgb.shape
                for detection in results.detections:
                    # Obtener bounding box normalizado
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convertir coordenadas normalizadas a píxeles
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    box_w = int(bbox.width * w)
                    box_h = int(bbox.height * h)
                    
                    # Asegurar que el rostro está dentro de la imagen
                    x = min(x, w - 1)
                    y = min(y, h - 1)
                    box_w = min(box_w, w - x)
                    box_h = min(box_h, h - y)
                    
                    if box_w > 0 and box_h > 0:
                        faces.append((x, y, box_w, box_h))

            return faces
        except Exception:
            pass  # Fallback a Haar Cascade si falla

    # Fallback a Haar Cascade
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.equalizeHist(gray_image)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) > 0:
        faces = apply_nms(faces, overlap_thresh=0.4)

    return faces


def predict_for_face(
    face_array: np.ndarray,
    gender_model: object,
    age_model: object | None,
    config: LabConfig,
) -> dict[str, str | float]:
    """Predice género y edad para un rostro recortado."""

    # Preprocesar el rostro
    face_vector, _ = preprocess_face_array(face_array, size=config.image_size)

    # Predecir género
    gender_id = int(gender_model.predict([face_vector])[0])
    gender_name = config.gender_map.get(gender_id, "unknown")

    # Predecir edad si el modelo está disponible
    age = None
    if age_model is not None:
        age = float(age_model.predict([face_vector])[0])

    return {"gender": gender_name, "age": age}


def annotate_image_with_predictions(
    image_pil: Image.Image,
    faces: list[tuple[int, int, int, int]],
    gender_model: object,
    age_model: object | None,
    config: LabConfig,
) -> Image.Image:
    """Anota la imagen con rectángulos y predicciones para cada rostro detectado."""

    # Convertir imagen PIL a array numpy para procesamiento
    image_array = np.array(image_pil)

    # Crear copia para anotar
    annotated_image = image_pil.copy()
    draw = ImageDraw.Draw(annotated_image)

    # Intentar cargar una fuente; si no está disponible, usar default
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Procesar cada rostro
    for x, y, w, h in faces:
        # Recortar rostro
        face_crop = image_array[y : y + h, x : x + w]

        # Predecir género y edad
        predictions = predict_for_face(face_crop, gender_model, age_model, config)

        # Dibujar rectángulo alrededor del rostro
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

        # Preparar texto de anotación
        label_text = f"Gender: {predictions['gender']}"
        if predictions["age"] is not None:
            label_text += f", Age: {predictions['age']:.0f}"

        # Dibujar texto
        text_position = (x, max(0, y - 20))
        draw.text(text_position, label_text, fill="red", font=font)

    return annotated_image


def run_app() -> None:
    """Ejecuta la app visual con detector de caras e inferencia."""

    st.set_page_config(page_title="Lab02 ML - Gender & Age Prediction", layout="wide")
    st.title("Laboratorio 02: Predicción de Género y Edad")
    st.write(
        "Carga una fotografía para detectar rostros y predecir género y edad usando "
        "los modelos entrenados."
    )

    # Inicializar configuración
    config = LabConfig()

    # Cargar modelos
    try:
        gender_model, age_model = load_models(config)
        models_loaded = True
    except FileNotFoundError as e:
        st.error(f"Error al cargar los modelos: {e}")
        st.info(
            "Asegúrate de haber ejecutado `python main.py` primero para generar "
            "pipeline_genero.pkl y (opcionalmente) pipeline_edad.pkl"
        )
        models_loaded = False

    if not models_loaded:
        st.stop()

    # Interfaz para cargar imagen
    uploaded_file = st.file_uploader(
        "Sube una fotografía",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Por favor, sube una imagen para comenzar.")
        st.stop()

    # Cargar imagen
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image_pil)

    # Detectar rostros
    st.subheader("Detección de Rostros")
    with st.spinner("Detectando rostros..."):
        faces = detect_faces_in_image(image_array)

    if len(faces) == 0:
        st.warning("No se detectaron rostros en la imagen.")
        st.image(image_pil, caption="Imagen cargada", use_container_width=True)
        st.stop()

    st.success(f"Se detectaron {len(faces)} rostro(s).")

    # Hacer predicciones y anotar
    with st.spinner("Haciendo predicciones..."):
        annotated_image = annotate_image_with_predictions(
            image_pil, faces, gender_model, age_model, config
        )

    # Mostrar resultado
    st.subheader("Resultados")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Imagen Original**")
        st.image(image_pil, use_container_width=True)

    with col2:
        st.write("**Imagen Anotada con Predicciones**")
        st.image(annotated_image, use_container_width=True)

    # Mostrar predicciones detalladas
    st.subheader("Predicciones Detalladas por Rostro")
    for i, (x, y, w, h) in enumerate(faces):
        face_crop = image_array[y : y + h, x : x + w]
        predictions = predict_for_face(face_crop, gender_model, age_model, config)

        with st.expander(f"Rostro #{i+1} (x={x}, y={y}, w={w}, h={h})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Género:** {predictions['gender']}")
            with col2:
                if predictions["age"] is not None:
                    st.write(f"**Edad Estimada:** {predictions['age']:.1f} años")
                else:
                    st.info("Modelo de edad aún no entrenado")

