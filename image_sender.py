# run this program on each RPi to send a labelled image stream
# you can run it on multiple RPi's; 8 RPi's running in above example
import imagezmq
import cv2
from tools import (dibujar_boxes, 
                   get_boxes,
                   dibujar_keypoints,
                   get_keypoints,
                   POSE_CONNECTIONS)
import dotenv
import os

dotenv.load_dotenv()

image_hub = imagezmq.ImageHub()

sender = imagezmq.ImageSender(connect_to=f'tcp://{os.getenv("IP")}')

camera_type = input("elige camara: IP = 1  o WEBCAM = 2")

if camera_type == "1":
    ip_cam = os.getenv("IPCAM")
    port_cam = os.getenv("PORTCAM")
    user_cam = os.getenv("USERCAM")
    pass_cam = os.getenv("PASSCAM")
    source_cam = f"rtsp://{user_cam}:{pass_cam}@{ip_cam}:{port_cam}/Streaming/Channels/1"
else:
    source_cam = int(os.getenv("CAM"))

cap = cv2.VideoCapture(source_cam)
while True:  # send images as stream until Ctrl-C
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 420))
    
    
    results = sender.send_image("video", frame)
    frame_boxes = dibujar_boxes(frame.copy(),
                              get_boxes(results))
    
    frame_keypoints = dibujar_keypoints(frame_boxes.copy(), get_keypoints(results))
    
    cv2.imshow('frame_boxes', frame_keypoints)
    cv2.waitKey(1)
    