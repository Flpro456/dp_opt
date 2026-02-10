import cv2
import imagezmq
import ultralytics
import json

sender = imagezmq.ImageSender(connect_to='tcp://192.168.100.166:5555')
model = ultralytics.YOLO("yolo11s-pose.pt")
image_hub = imagezmq.ImageHub()
while True:  # show streamed images until Ctrl-C
    print('Servidor en escucha')
    rpi_name, image = image_hub.recv_image()
    #image_hub.send_reply(b'OK')
    results = model(image)
    # Ejemplo de cómo extraer datos de 'results[0].boxes'
    boxes = results[0].boxes
    keypoints = results[0].keypoints
    results_dic = {
        "boxes": boxes.xyxy.tolist(),      # Coordenadas en formato lista
        "confidences": boxes.conf.tolist(), # Confidencias
        "classes": boxes.cls.tolist()       # IDs de las clases
    }
    results_key = {
        "keypoints": keypoints.xy.tolist(),
        "confidences": keypoints.conf.tolist(),
    }

    master = {
        "boxes": results_dic,
        "keypoints": results_key
    }

    results_json = json.dumps(master).encode('ascii')
    image_hub.send_reply(results_json)
    #annotated_frame = results[0].plot()
    #sender.send_image(rpi_name, annotated_frame)