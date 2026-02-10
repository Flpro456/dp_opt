# run this program on each RPi to send a labelled image stream
# you can run it on multiple RPi's; 8 RPi's running in above example
import imagezmq
import cv2
from tools import (dibujar_boxes, 
                   get_boxes)


image_hub = imagezmq.ImageHub()

sender = imagezmq.ImageSender(connect_to='tcp://192.168.100.130:5555')

cap = cv2.VideoCapture(0)
while True:  # send images as stream until Ctrl-C
    ret, frame = cap.read()
    results = sender.send_image("video", frame)
    frame_boxes = dibujar_boxes(frame.copy(),
                              get_boxes(results))
    cv2.imshow('frame_boxes', frame_boxes)
    cv2.waitKey(1)
    