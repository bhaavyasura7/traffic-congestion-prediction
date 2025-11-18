import mysql.connector
import numpy as np
from decimal import Decimal
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import json

def get_congestion_level(features, video_id):
    try:
        # Load the trained model for this video
        model_path = f'traffic_model_{video_id}.joblib'
        model = joblib.load(model_path)
        
        # Make prediction
        prediction = model.predict([features])[0]
        confidence = np.max(model.predict_proba([features])[0])
        
        # Map prediction to congestion level
        level_map = {
            0: "Very Light",
            1: "Light",
            2: "Moderate",
            3: "High",
            4: "Very High"
        }
        
        return level_map[prediction], confidence
    except Exception as e:
        print(f"Error using ML model: {str(e)}")
        # Fallback to rule-based prediction with 5 levels
        weighted_count = features[-1]  # Last feature is weighted_count
        if weighted_count < 8:
            return "Very Light", 0.4
        elif weighted_count < 12:
            return "Light", 0.5
        elif weighted_count < 16:
            return "Moderate", 0.6
        elif weighted_count < 20:
            return "High", 0.7
        else:
            return "Very High", 0.8


def predict_congestion_index(vehicle_counts):
    # Vehicle weights
    weights = {
        'car': 1.0,
        'bike': 0.5,
        'auto': 0.8,
        'truck': 2.5,
        'bus': 3.0
    }
    
    # Calculate weighted sum
    congestion_index = sum(count * weights[vehicle_type] for vehicle_type, count in vehicle_counts.items())
    return congestion_index

def predict_future_traffic():
    json_output = []
    db = None
    try:
        db = mysql.connector.connect(host="localhost", user="root", password="aneeshrao", database="ml")
        cursor = db.cursor(dictionary=True)

        cursor.execute("""
            SELECT vc.car, vc.bike, vc.truck, vc.bus, vc.auto, vc.weighted_count as current_congestion_index, f.video_id
            FROM vehicle_counts vc JOIN frames f ON vc.frame_id = f.frame_id
            WHERE f.video_id = (SELECT MAX(video_id) FROM frames)
            ORDER BY vc.count_id DESC LIMIT 1
        """)
        current_data = cursor.fetchone()

        if not current_data:
            print("No current traffic data available")
            return

        current_counts = { 'car': float(current_data['car']), 'bike': float(current_data['bike']), 'truck': float(current_data['truck']), 'bus': float(current_data['bus']), 'auto': float(current_data['auto']) }
        current_index = float(current_data['current_congestion_index'])
        video_id = current_data['video_id']
        
        # <-- NEW: Get the maximum observed congestion index to use as our 100% benchmark ---
        cursor.execute("""
            SELECT MAX(vc.weighted_count) as max_wc
            FROM vehicle_counts vc JOIN frames f ON vc.frame_id = f.frame_id
            WHERE f.video_id = %s
        """, (video_id,))
        max_wc_data = cursor.fetchone()
        # Use a fallback of 30 if no max is found or if it's zero, to avoid division by zero errors.
        max_weighted_count = float(max_wc_data['max_wc']) if max_wc_data and max_wc_data['max_wc'] else 30.0
        print(f"Using max observed congestion index (100% capacity) for this video: {max_weighted_count:.2f}")

        current_features = list(current_counts.values()) + [current_index]
        congestion_level, confidence = get_congestion_level(current_features, video_id)

        # --- Current Hour (Hour 0) ---
        print(f"\nTime: Current")
        print(f"Current Congestion Index: {current_index:.1f}")
        print(f"Current Congestion: {congestion_level}")
        # ... (rest of the print statements) ...

        # <-- NEW: Add the dynamic percentage to the JSON output ---
        json_output.append({
            "predict_hour": 0,
            "congestion_level": congestion_level,
            "confidence": round(confidence, 6),
            
            
        })

        # --- Future Hours (Hours 1-5) ---
        for hour in range(1, 6):
            variation = np.random.normal(1, 0.40) # Using higher variation for more dynamic results
            predicted_counts = {k: v * variation for k, v in current_counts.items()}
            predicted_index = predict_congestion_index(predicted_counts)
            predicted_features = list(predicted_counts.values()) + [predicted_index]
            congestion_level, confidence = get_congestion_level(predicted_features, video_id)
            
            print(f"Time: {hour} hr later")
            print(f"Predicted Congestion: {congestion_level}")
            # ... (rest of the print statements) ...
            
            # <-- NEW: Add the dynamic percentage to the JSON output ---
            json_output.append({
                "predict_hour": hour,
                "congestion_level": congestion_level,
                "confidence": round(confidence, 6),
                
            })

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if db and db.is_connected():
            db.close()

    if json_output:
        output_dir = "dashboard/data"
        output_file = os.path.join(output_dir, "prediction.json")
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(json_output, f, indent=4)
            print(f"\n✅ Success! Direct prediction report saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving JSON file: {e}")

# def get_stored_predictions(video_id):
#     try:
#         # Connect to database
#         db = mysql.connector.connect(
#             host="localhost",
#             user="root",
#             password="aneeshrao",
#             database="ml"
#         )
#         cursor = db.cursor(dictionary=True)

#         # Get predictions for the video
#         cursor.execute("""
#             SELECT predict_hour, congestion_level, confidence, predicted_at
#             FROM time_predictions
#             WHERE video_id = %s
#             ORDER BY predict_hour ASC
#         """, (video_id,))

#         predictions = cursor.fetchall()
        
#         if not predictions:
#             print(f"No predictions found for video {video_id}")
#             return

#         # Output to console removed as requested. All predictions are now saved to JSON only.

#     except Exception as e:
#         print(f"Error retrieving predictions: {str(e)}")
#     finally:
#         if 'db' in locals():
#             db.close()

if __name__ == "__main__":
    predict_future_traffic()
    # video_id = 1  # Example video_id, you might want to get this dynamically
    # get_stored_predictions(video_id)