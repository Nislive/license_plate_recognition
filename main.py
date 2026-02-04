from ultralytics import YOLO
import sqlite3
import numpy as np
import cv2
from paddleocr import PaddleOCR
import threading
from queue import Queue
from paddleocr import TextRecognition

def init_db():
    connection = sqlite3.connect("traffic_data.db")
    cursor = connection.cursor()

    cursor.execute("DROP TABLE IF EXISTS detections")
    table_creation_text = """
        CREATE TABLE detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATETIME DEFAULT CURRENT_TIMESTAMP,
            vehicle_type VARCHAR(25),
            plate_text VARCHAR(25),
            plate_confidence FLOAT
        );
"""
    cursor.execute(table_creation_text)
    connection.commit()
    connection.close()

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def fix_plate_text_errors(text):    
    dict_num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}
    dict_char_to_num = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8'}

    text = text.replace(" ", "")
    corrected= list(text)
    length = len(text)
    if length<5:
        return
    for i in range(length):
        if i<2:
            if corrected[i] in dict_char_to_num:
                corrected[i]=dict_char_to_num[corrected[i]]

        elif i==2:
            if corrected[i] in dict_num_to_char:
                corrected[i]=dict_num_to_char[corrected[i]]
                
        elif i>=length-2:
            if corrected[i] in dict_char_to_num:
                corrected[i]=dict_char_to_num[corrected[i]]
        elif i == 3  and length == 7:
                if corrected[i] in dict_num_to_char:
                    corrected[i] = dict_num_to_char[corrected[i]]
    return "".join(corrected)

