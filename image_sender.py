# run this program on each RPi to send a labelled image stream
# you can run it on multiple RPi's; 8 RPi's running in above example
import socket
import imagezmq
import cv2


def dibujar_detecciones(frame, data_json):
    """
    Dibuja cajas y etiquetas basadas en el JSON del servidor.
    """
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

image_hub = imagezmq.ImageHub()

sender = imagezmq.ImageSender(connect_to='tcp://192.168.100.130:5555')

rpi_name = "socket.gethostname()" # send RPi hostname with each image
picam = cv2.VideoCapture(0)
while True:  # send images as stream until Ctrl-C
    ret, frame = picam.read()
    results = sender.send_image(rpi_name, frame)
    print(results)
    new_frame = dibujar_detecciones(frame.copy(),results)
    #name, image = image_hub.recv_image()
    cv2.imshow('Floro', new_frame)
    cv2.waitKey(1)
    #image_hub.send_reply(b'OK')
    