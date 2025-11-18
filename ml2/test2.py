from ultralytics import YOLO
import cv2
import os
import csv
from collections import Counter
import mysql.connector
import sys
import traceback
import gc
import numpy as np
import torch

try:
    # Database connection
    print('Connecting to database...')
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="aneeshrao",
        database="ml"
    )
    cursor = db.cursor()
    print('Database connection established')

    # Create tables if they don't exist
    print('Checking database schema...')
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id INT AUTO_INCREMENT PRIMARY KEY,
            file_name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS frames (
            frame_id INT AUTO_INCREMENT PRIMARY KEY,
            video_id INT,
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_counts (
            count_id INT AUTO_INCREMENT PRIMARY KEY,
            frame_id INT,
            car INT DEFAULT 0,
            bike INT DEFAULT 0,
            truck INT DEFAULT 0,
            bus INT DEFAULT 0,
            auto INT DEFAULT 0,
            total INT DEFAULT 0,
            weighted_count FLOAT DEFAULT 0,
            FOREIGN KEY (frame_id) REFERENCES frames(frame_id)
        )
    """)
    db.commit()
    print('Database schema verified')

    # Load trained models
    print('Loading YOLO models...')
    model_coco = YOLO("yolov8n.pt")  # COCO model: car, bus, truck, etc.
    model_custom = YOLO("runs/detect/train5/weights/best.pt")  # Your model: auto, bike
    print('YOLO models loaded')

    # Vehicle class IDs for COCO
    coco_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    # Ensure directories exist
    print('Creating directories...')
    os.makedirs('data/raw_videos', exist_ok=True)
    os.makedirs('data/yolo_detections', exist_ok=True)
    print('Directories created')

    # Load video
    if len(sys.argv) < 2:
        raise Exception('No video file path provided. Usage: python test2.py <video_path>')
    video_path = sys.argv[1]
    print(f'Opening video file: {video_path}')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f'Could not open video file: {video_path}')
    print(f'Video file opened successfully: {video_path}')

    # Clean up existing video data if exists
    print('Checking for existing video data...')
    cursor.execute("SELECT video_id FROM videos WHERE file_name = %s", (video_path,))
    existing_video = cursor.fetchone()
    if existing_video:
        video_id = existing_video[0]
        print(f'Found existing video data with ID: {video_id}. Cleaning up...')
        cursor.execute("""
            DELETE vc FROM vehicle_counts vc
            INNER JOIN frames f ON vc.frame_id = f.frame_id
            WHERE f.video_id = %s
        """, (video_id,))
        cursor.execute("DELETE FROM frames WHERE video_id = %s", (video_id,))
        cursor.execute("DELETE FROM videos WHERE video_id = %s", (video_id,))
        db.commit()
        print('Existing video data cleaned up')

    # Insert video record
    print('Inserting video record...')
    cursor.execute("INSERT INTO videos (file_name) VALUES (%s)", (video_path,))
    db.commit()
    video_id = cursor.lastrowid
    print(f'Video record inserted with ID: {video_id}')

    # Open CSV file
    print('Creating CSV file...')
    csv_path = f'data/yolo_detections/{os.path.splitext(os.path.basename(video_path))[0]}_detections.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame', 'car', 'bike', 'auto', 'bus', 'truck', 'total', 'weighted'])

        frame_count = 0
        frame_interval = 60  # Process 1 frame per 2 seconds for 30fps video

        print('Starting frame processing...')
        try:
            # Setup video writer to save the displayed window
            frame_width, frame_height = 640, 360  # Fixed size to match resized frames
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('dashboard/data/detected_output.mp4', fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                raise Exception('Failed to open VideoWriter for dashboard/data/detected_output.mp4')
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print('End of video reached')
                    break

                # Process every nth frame
                if frame_count % frame_interval == 0:
                    print(f'Processing frame {frame_count}...')
                    try:
                        # Resize frame for processing
                        frame = cv2.resize(frame, (640, 360))

                        # Insert frame record
                        cursor.execute(
                            "INSERT INTO frames (video_id) VALUES (%s)",
                            (video_id,)
                        )
                        db.commit()
                        frame_id = cursor.lastrowid

                        # Detect vehicles
                        with torch.no_grad():
                            results_coco = model_coco(frame, verbose=False)[0]
                            results_custom = model_custom(frame, verbose=False)[0]

                        # Create a copy of frame for visualization
                        frame_viz = frame.copy()

                        # Draw COCO model detections
                        for box in results_coco.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            cls = int(box.cls.cpu().numpy()[0])
                            conf = float(box.conf.cpu().numpy()[0])
                            
                            if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                                label = 'car' if cls == 2 else 'bike' if cls == 3 else 'bus' if cls == 5 else 'truck'
                                color = (0, 255, 0) if cls == 2 else (255, 0, 0) if cls == 3 else (0, 0, 255) if cls == 5 else (255, 255, 0)
                                cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(frame_viz, f'{label} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Draw custom model detections (auto)
                        for box in results_custom.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = float(box.conf.cpu().numpy()[0])
                            cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            cv2.putText(frame_viz, f'auto {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                        # Save and display frame with detections
                        out.write(frame_viz)  # Save the frame after all drawing
                        cv2.imshow('Vehicle Detection', frame_viz)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                        # Count vehicles
                        vehicle_counts = Counter()
                        for cls in results_coco.boxes.cls.cpu().numpy().astype(int):
                            if cls == 2:  # car
                                vehicle_counts['car'] += 1
                            elif cls == 3:  # motorcycle
                                vehicle_counts['bike'] += 1
                            elif cls == 5:  # bus
                                vehicle_counts['bus'] += 1
                            elif cls == 7:  # truck
                                vehicle_counts['truck'] += 1

                        # Count custom detections (auto)
                        vehicle_counts['auto'] = len(results_custom.boxes.cls)

                        # Calculate total and weighted count
                        total = sum(vehicle_counts.values())
                        weighted = (
                            1.0 * vehicle_counts['car'] +
                            0.5 * vehicle_counts['bike'] +
                            0.8 * vehicle_counts['auto'] +
                            2.5 * vehicle_counts['truck'] +
                            3.0 * vehicle_counts['bus']
                        )

                        # Write to CSV
                        writer.writerow([
                            frame_count,
                            vehicle_counts['car'],
                            vehicle_counts['bike'],
                            vehicle_counts['auto'],
                            vehicle_counts['bus'],
                            vehicle_counts['truck'],
                            total,
                            weighted
                        ])

                        # Insert vehicle counts
                        cursor.execute("""
                            INSERT INTO vehicle_counts
                            (frame_id, car, bike, auto, bus, truck, total, weighted_count)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            frame_id,
                            vehicle_counts['car'],
                            vehicle_counts['bike'],
                            vehicle_counts['auto'],
                            vehicle_counts['bus'],
                            vehicle_counts['truck'],
                            total,
                            weighted
                        ))
                        db.commit()

                    except Exception as e:
                        print(f'Error processing frame {frame_count}: {str(e)}')
                        traceback.print_exc()

                frame_count += 1

        except Exception as e:
            print(f'Error processing video: {str(e)}')
            traceback.print_exc()

        finally:
            # Release video capture and video writer, close windows
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Run phase2 clustering
    print('\nStarting phase2 clustering...')
    try:
        import phase2_clustering
        print('Phase2 clustering completed successfully')
    except Exception as e:
        print(f'Error during phase2 clustering: {str(e)}')
        traceback.print_exc()

    # Close database connection
    try:
        cursor.close()
        db.close()
        print('Database connection closed')
    except Exception as e:
        print(f'Error while closing database connection: {str(e)}')

except Exception as e:
    print(f'Error: {str(e)}')
    traceback.print_exc()