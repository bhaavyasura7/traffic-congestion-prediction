# backend_server.py
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import subprocess
import shutil
import cv2

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploaded_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def run_script(cmd, timeout=600):
    proc = subprocess.Popen(cmd)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"Process {cmd} timed out and was killed.")

@app.route('/upload', methods=['POST'])


def upload_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    # Save the file
    save_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, file.filename))
    file.save(save_path)

    # Clear the database before video processing
    try:
        import mysql.connector
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="050506",
            database="ml"
        )
        cursor = db.cursor()
        cursor.execute("SET FOREIGN_KEY_CHECKS=0;")
        cursor.execute("TRUNCATE TABLE vehicle_counts;")
        cursor.execute("TRUNCATE TABLE frames;")
        cursor.execute("TRUNCATE TABLE videos;")
        cursor.execute("TRUNCATE TABLE clusters;")
        cursor.execute("TRUNCATE TABLE time_predictions;")
        cursor.execute("TRUNCATE TABLE predictions;")
        cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
        db.commit()
        cursor.close()
        db.close()
    except Exception as db_err:
        print(f"Error clearing database: {db_err}")

    # Call test2.py with the absolute path to the saved file, then run phase3_prediction.py and congestion_predictor.py
    try:
        # Run test2.py and wait for it to finish
        run_script(['python', 'test2.py', save_path])
        # Run phase3_prediction.py
        run_script(['python', 'phase3_prediction.py'])
        # Run congestion_predictor.py
        run_script(['python', 'congestion_predictor.py'])
        # Read output files
        model_perf_path = os.path.join('dashboard', 'data', 'model_performance.json')
        prediction_path = os.path.join('dashboard', 'data', 'prediction.json')
        model_performance = None
        prediction = None
        try:
            with open(model_perf_path, 'r') as f:
                model_performance = f.read()
        except Exception:
            model_performance = None
        try:
            with open(prediction_path, 'r') as f:
                prediction = f.read()
        except Exception:
            prediction = None
        response = jsonify({
            'success': True,
            'path': save_path,
            'model_performance': model_performance,
            'prediction': prediction
        })
        
        # Delete old uploaded videos in dashboard/data before moving new one, but keep detected_output.mp4
        try:
            target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dashboard', 'data'))
            os.makedirs(target_dir, exist_ok=True)
            # Remove all .mp4 files except detected_output.mp4
            for fname in os.listdir(target_dir):
                if fname.endswith('.mp4') and fname != 'detected_output.mp4':
                    try:
                        os.remove(os.path.join(target_dir, fname))
                    except Exception as rm_err:
                        print(f"Error deleting old video {fname}: {rm_err}")
            # Move processed detected_output.mp4 if it exists in working dir
            processed_mp4 = os.path.abspath(os.path.join(target_dir, 'detected_output.mp4'))
            if not os.path.exists(processed_mp4):
                # Try to move from working dir if not already present
                src_mp4 = os.path.abspath(os.path.join(os.path.dirname(__file__), 'detected_output.mp4'))
                if os.path.exists(src_mp4):
                    shutil.move(src_mp4, processed_mp4)
        except Exception as e:
            print(f"Error moving processed video: {e}")

        return response
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Serve processed video for dashboard playback
@app.route('/dashboard/data/<path:filename>')
def serve_dashboard_data(filename):
    from flask import send_from_directory
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'dashboard', 'data'), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
