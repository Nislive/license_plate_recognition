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


def ocr_worker(pose_model, ocr_instance):
    
    db_con = sqlite3.connect("traffic_data.db")
    db_cur = db_con.cursor()
    while True:
        data = ocr_queue.get()
        if data is None: break
        
        vehicle_crop, track_id, vehicle_type = data
        #cv2.imwrite(f"debug_plates/{track_id}.jpg",vehicle_crop)
        pose_results = pose_model(vehicle_crop, verbose=False) 
        
        for pose_result in pose_results:
            if pose_result.keypoints is not None and pose_result.keypoints.xy.shape[0] > 0 and pose_result.keypoints.xy.shape[1] >= 4:
                points = pose_result.keypoints.xy[0].cpu().numpy()

                src_pts = order_points(points)
                
                height_plate=100
                width_plate =int(height_plate*4.70)
                
                dst_pts = np.float32([
                    [0,0],
                    [width_plate, 0],
                    [width_plate, height_plate],
                    [0, height_plate]
                ])

                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

                plate_flat = cv2.warpPerspective(vehicle_crop, matrix, (width_plate, height_plate), flags=cv2.INTER_CUBIC)


                kernel = np.array([[-1,-1,-1], 
                [-1, 9,-1], 
                [-1,-1,-1]])
                plate_final = cv2.filter2D(plate_flat, -1, kernel)
                #cv2.imwrite(f"debug_plates/{track_id}.jpg", plate_final)

                try:
                    ocr_results = text_model.predict(plate_final)
                except Exception as e:
                    print(e)
                    continue
                    
                if ocr_results:
                    for res in ocr_results:
                        if res is None:
                            continue
                        text_raw, score = res["rec_text"],res["rec_score"]                         
                        if score>0.60:                        
                            plate_text = fix_plate_text_errors(text_raw)
                            if plate_text is not None:
                                try:
                                    print(vehicle_type)
                                    print(plate_text)
                                    db_cur.execute("INSERT INTO detections (vehicle_type, plate_text, plate_confidence) VALUES (?,?,?)", (vehicle_type, plate_text, score))
                                    db_con.commit()
                                except Exception as e:
                                    print(e)
        ocr_queue.task_done()
    db_con.close()


text_model = TextRecognition(model_name="PP-OCRv5_server_rec")
ocr_queue = Queue()
