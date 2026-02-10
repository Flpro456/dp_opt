import cv2
import json

def get_boxes(results):
    results = json.loads(results.decode('ascii'))
    dict_boxes= results["boxes"]
    return dict_boxes

def get_keypoints(results):
    pass

def dibujar_boxes(frame, data_json):
    """
    Dibuja cajas y etiquetas basadas en el JSON del servidor.
    """
    if isinstance(data_json, (bytes, bytearray)):
        data_json = json.loads(data_json)

    boxes = data_json.get("boxes", [])
    confidences = data_json.get("confidences", [])
    classes = data_json.get("classes", [])

    for i in range(len(boxes)):
        # Extraer coordenadas (x1, y1, x2, y2) y convertir a int
        x1, y1, x2, y2 = map(int, boxes[i])
        conf = confidences[i]
        cls = int(classes[i])

        # Definir color y etiqueta
        color = (0, 255, 0)  # Verde
        label = f"ID: {cls} {conf:.2f}"

        # 1. Dibujar el rectángulo de la caja
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 2. Dibujar un pequeño fondo para el texto (opcional, mejora lectura)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 5), color, -1)

        # 3. Poner el texto
        cv2.putText(frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame