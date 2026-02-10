import cv2
import numpy as np
import json

def get_boxes(results):
    results = json.loads(results.decode('ascii'))
    dict_boxes= results["boxes"]
    return dict_boxes

def get_keypoints(results):
    results = json.loads(results.decode('ascii'))
    dict_keypoints= results["keypoints"]
    return dict_keypoints

# Definimos las conexiones entre keypoints (basado en COCO)
# Cada par indica los índices de los puntos que deben conectarse
POSE_CONNECTIONS = [
    (5, 6),   # hombros
    (5, 7), (7, 9),   # brazo izquierdo
    (6, 8), (8, 10),  # brazo derecho
    (11, 12),         # caderas
    (11, 13), (13, 15), # pierna izquierda
    (12, 14), (14, 16), # pierna derecha
    (5, 11), (6, 12)    # torso
]

def dibujar_keypoints(frame, data_json):
    """
    Dibuja keypoints y conexiones (esqueleto) sobre el frame.
    """
    if isinstance(data_json, (bytes, bytearray)):
        data_json = json.loads(data_json)

    keypoints = data_json.get("keypoints", [])
    confidences = data_json.get("confidences", [])

    for i, person in enumerate(keypoints):
        # Dibujar puntos
        for j, (x, y) in enumerate(person):
            conf = confidences[i][j] if i < len(confidences) else 1.0
            if conf > 0.5:  # umbral de confianza
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        # Dibujar conexiones
        for (p1, p2) in POSE_CONNECTIONS:
            if p1 < len(person) and p2 < len(person):
                x1, y1 = person[p1]
                x2, y2 = person[p2]
                conf1 = confidences[i][p1] if i < len(confidences) else 1.0
                conf2 = confidences[i][p2] if i < len(confidences) else 1.0
                if conf1 > 0.5 and conf2 > 0.5:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return frame



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

class ContadorFlexionesBrazo:
    def __init__(self, umbral_flex=60, umbral_ext=150):
        self.umbral_flex = umbral_flex
        self.umbral_ext = umbral_ext
        self.estado = "extendido"
        self.contador = 0

    def calcular_angulo(self, p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return None
        angle = np.arccos(dot / (norm_v1 * norm_v2))
        return np.degrees(angle)

    def actualizar(self, keypoints):
        hombro = keypoints[6]
        codo = keypoints[8]
        muneca = keypoints[10]

        angulo = self.calcular_angulo(hombro, codo, muneca)
        if angulo is None:
            return self.contador

        # Detectar transición
        if self.estado == "extendido" and angulo < self.umbral_flex:
            self.estado = "flexionado"
        elif self.estado == "flexionado" and angulo > self.umbral_ext:
            self.estado = "extendido"
            self.contador += 1

        return self.contador
