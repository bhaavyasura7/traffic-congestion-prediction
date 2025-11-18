import traceback
import mysql.connector
import cv2
from ultralytics import YOLO
from datetime import datetime, timedelta
import os

try:
    # Test MySQL connection
    print('Testing MySQL connection...')
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="aneeshrao",
        database="ml"
    )
    cursor = db.cursor()
    print('MySQL connection successful')
    
    # Test YOLO model loading
    print('\nTesting YOLO model loading...')
    model_coco = YOLO("yolov8n.pt")
    model_custom = YOLO("runs/detect/train5/weights/best.pt")
    print('YOLO models loaded successfully')
    
    # Test video file access and frame reading
    print('\nTesting video file access and frame reading...')
    video_path = "yolotest2.dav"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f'Could not open video file: {video_path}')
    
    ret, frame = cap.read()
    if not ret:
        raise Exception('Could not read frame from video')
    print('Video frame read successfully')
    
    # Test YOLO detection
    print('\nTesting YOLO detection...')
    results_coco = model_coco(frame, verbose=False)[0]
    results_custom = model_custom(frame, verbose=False)[0]
    print('YOLO detection successful')
    
    # Test database operations
    print('\nTesting database operations...')
    cursor.execute("INSERT INTO videos (file_name) VALUES (%s)", (video_path,))
    db.commit()
    video_id = cursor.lastrowid
    print(f'Video record inserted with ID: {video_id}')
    
    # Get video properties for timestamp calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    # Calculate video timestamp based on frame position
    base_timestamp = datetime(2025, 6, 26, 11, 34, 19)  # Video's actual start time
    current_frame_time_seconds = frame_count / fps
    video_timestamp = base_timestamp + timedelta(seconds=current_frame_time_seconds)
    
    cursor.execute(
        "INSERT INTO frames (video_id, video_timestamp, hour, minute) VALUES (%s, %s, %s, %s)",
        (video_id, video_timestamp, video_timestamp.hour, video_timestamp.minute)
    )
    db.commit()
    frame_id = cursor.lastrowid
    print(f'Frame record inserted with ID: {frame_id}')
    
    # Test directory creation
    print('\nTesting directory creation...')
    os.makedirs('data/raw_videos', exist_ok=True)
    os.makedirs('data/yolo_detections', exist_ok=True)
    print('Directories created successfully')
    
    cap.release()
    cursor.close()
    db.close()
    
    print('\nAll preliminary tests passed successfully')
    print('\nRunning test2.py...')
    exec(open('test2.py').read())
    
except mysql.connector.Error as err:
    print(f'MySQL Error: {err}')
except Exception as e:
    print(f'Error: {str(e)}\nTraceback:\n{traceback.format_exc()}')