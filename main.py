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

