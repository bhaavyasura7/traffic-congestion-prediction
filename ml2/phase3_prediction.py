import mysql.connector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
import numpy as np
import joblib

try:
    # Connect to MySQL database
    print('Connecting to database...')
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="aneeshrao",
        database="ml"
    )
    cursor = db.cursor()
    print('Database connection established')

    # Get current video ID
    cursor.execute("SELECT MAX(video_id) FROM frames")
    current_video_id = cursor.fetchone()[0]
    
    # Fetch training data for current video
    print('Fetching vehicle count and cluster data for current video...')
    cursor.execute("""
        SELECT 
            vc.car, vc.bike, vc.truck, vc.auto, vc.bus, vc.weighted_count,
            c.cluster_label
        FROM vehicle_counts vc
        JOIN clusters c ON vc.frame_id = c.frame_id
        JOIN frames f ON vc.frame_id = f.frame_id
        WHERE f.video_id = %s
    """, (current_video_id,))
    data = cursor.fetchall()
    
    if not data:
        raise Exception("No data found in the database")

    # Prepare features (X) and target (y)
    X = np.array([[car, bike, truck, auto, bus, weighted] for car, bike, truck, auto, bus, weighted, _ in data])
    y = np.array([label for *_, label in data])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f'Training data shape: {X_train.shape}')
    print(f'Testing data shape: {X_test.shape}')
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200], # Number of trees in the forest
        'max_depth': [10, 20, 30, None], # Maximum depth of the tree
        'min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4] # Minimum number of samples required to be at a leaf node
    }# 3*4*3*3 = 108 models

    # Initialize and train Random Forest with GridSearchCV
    print('\nPerforming Grid Search for best parameters...')
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print('\nBest parameters found:')
    for param, value in grid_search.best_params_.items():
        print(f'{param}: {value}')

    # Get the best model
    classifier = grid_search.best_estimator_

    # Perform cross-validation
    print('\nPerforming 5-fold Cross Validation...')
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1_weighted')
    print(f'Cross-validation scores: {cv_scores}')
    print(f'Average CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    import json
    output_json = {}
    output_json['model_performance'] = {
        'mean_squared_error': round(mse, 4),
        'r2_score': round(r2, 4),
        'precision_score': round(precision, 4),
        'recall_score': round(recall, 4),
        'f1_score': round(f1, 4)
    }

    print('\nModel Performance Metrics:')
    for k, v in output_json['model_performance'].items():
        print(f'{k.replace("_", " ").title()}: {v}')

    # Feature importance
    feature_names = ['Car', 'Bike', 'Truck', 'Auto', 'Bus', 'Weighted Count']
    importances = classifier.feature_importances_
    output_json['feature_importances'] = dict(zip(feature_names, [round(float(i), 4) for i in importances]))
    print('\nFeature Importances:')
    for name, importance in output_json['feature_importances'].items():
        print(f'{name}: {importance:.4f}')

    # Drop and recreate predictions table
    print('\nRecreating predictions table...')
    cursor.execute("DROP TABLE IF EXISTS predictions")
    cursor.execute("""
    CREATE TABLE predictions (
        prediction_id INT AUTO_INCREMENT PRIMARY KEY,
        frame_id INT,
        predicted_label INT,
        actual_label INT,
        prediction_error FLOAT,
        confidence FLOAT,
        FOREIGN KEY (frame_id) REFERENCES frames(frame_id)
    )
    """)
    db.commit()

    # Store predictions for test set of current video
    print('Storing predictions for current video...')
    cursor.execute("""
        SELECT vc.frame_id 
        FROM vehicle_counts vc
        JOIN frames f ON vc.frame_id = f.frame_id
        WHERE f.video_id = %s
        ORDER BY vc.frame_id
    """, (current_video_id,))
    frame_ids = [row[0] for row in cursor.fetchall()]
    
    # Get prediction probabilities
    y_pred_proba = classifier.predict_proba(X_test)

    for i, (pred, actual, proba) in enumerate(zip(y_pred, y_test, y_pred_proba)):
        frame_id = frame_ids[i]  # Use direct index instead of calculating test set offset
        error = abs(pred - actual)
        confidence = np.max(proba)  # Use the highest probability as confidence
        
        cursor.execute("""
            INSERT INTO predictions 
            (frame_id, predicted_label, actual_label, prediction_error, confidence)
            VALUES (%s, %s, %s, %s, %s)
        """, (frame_id, int(pred), int(actual), float(error), float(confidence)))
    db.commit()

    # Print prediction statistics
    print('\nPrediction Statistics:')
    cursor.execute("""
        SELECT 
            AVG(prediction_error) as avg_error,
            MIN(prediction_error) as min_error,
            MAX(prediction_error) as max_error,
            AVG(confidence) as avg_confidence
        FROM predictions
    """)
    stats = cursor.fetchone()
    output_json['prediction_statistics'] = {
        'average_error': round(stats[0], 2),
        'min_error': round(stats[1], 2),
        'max_error': round(stats[2], 2),
        'average_confidence': round(stats[3], 2)
    }
    for k, v in output_json['prediction_statistics'].items():
        print(f'{k.replace("_", " ").title()}: {v}')

    # Prediction distribution
    print('\nPrediction Distribution:')
    unique_predictions, counts = np.unique(y_pred, return_counts=True)
    total_predictions = len(y_pred)
    pred_dist = []
    for pred, count in zip(unique_predictions, counts):
        percentage = (count / total_predictions) * 100
        print(f'Class {pred}: {count} predictions ({percentage:.2f}%)')
        pred_dist.append({"class": int(pred), "count": int(count), "percentage": round(percentage, 2)})
    output_json['prediction_distribution'] = pred_dist

    # Actual label distribution
    print('\nActual Label Distribution:')
    unique_labels, counts = np.unique(y_test, return_counts=True)
    total_labels = len(y_test)
    label_dist = []
    for label, count in zip(unique_labels, counts):
        percentage = (count / total_labels) * 100
        print(f'Class {label}: {count} labels ({percentage:.2f}%)')
        label_dist.append({"class": int(label), "count": int(count), "percentage": round(percentage, 2)})
    output_json['actual_label_distribution'] = label_dist

    # Save the trained model with video-specific filename
    print('\nSaving trained model...')
    model_path = f'traffic_model_{current_video_id}.joblib'
    joblib.dump(classifier, model_path)
    print(f'Model saved as {model_path}')

    # Save output_json to file in dashboard/model_performance.json
    try:
        with open("dashboard/data/model_performance.json", "w") as f:
            json.dump(output_json, f, indent=2)
        print("All metrics and statistics saved to dashboard/model_performance.json")
    except Exception as json_err:
        print(f"Error writing JSON: {json_err}")

    print('\nModel training and prediction completed successfully')

except mysql.connector.Error as err:
    print(f'MySQL Error: {err}')
except Exception as e:
    print(f'Error: {str(e)}')
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'db' in locals():
        db.close()
    print('Database connection closed')