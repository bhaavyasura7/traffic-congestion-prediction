import mysql.connector
from sklearn.cluster import KMeans
import numpy as np
from datetime import datetime

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
    
    # Fetch weighted_count values from vehicle_counts table for current video only
    print('Fetching weighted count data for current video...')
    cursor.execute("""
        SELECT vc.weighted_count 
        FROM vehicle_counts vc
        JOIN frames f ON vc.frame_id = f.frame_id
        WHERE f.video_id = %s
    """, (current_video_id,))
    weighted_counts = cursor.fetchall()
    weighted_counts = np.array(weighted_counts)
    print(f'Fetched {len(weighted_counts)} weighted count records')

    # Reshape data for KMeans
    X = weighted_counts.reshape(-1, 1)

    # Perform KMeans clustering
    print('Performing KMeans clustering...')
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)

    # Get cluster centers and sort them
    centers = kmeans.cluster_centers_.flatten()
    sorted_center_indices = np.argsort(centers)
    
    # Map congestion levels to sorted clusters
    congestion_levels = [
        "Very Light",
        "Light",
        "Moderate",
        "High",
        "Very High"
    ]
    cluster_to_level = {}
    for i, center_idx in enumerate(sorted_center_indices):
        cluster_to_level[center_idx] = congestion_levels[i]

    # Create clusters table if not exists
    print('Creating clusters table...')
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clusters (
        cluster_id INT AUTO_INCREMENT PRIMARY KEY,
        frame_id INT,
        cluster_label INT,
        congestion_level VARCHAR(50),
        FOREIGN KEY (frame_id) REFERENCES frames(frame_id)
    )
    """)
    db.commit()

    # Get vehicle count records with IDs and timestamps for current video
    print('Fetching complete vehicle count records for current video...')
    cursor.execute("""
        SELECT vc.frame_id, vc.weighted_count 
        FROM vehicle_counts vc
        JOIN frames f ON vc.frame_id = f.frame_id
        WHERE f.video_id = %s
    """, (current_video_id,))
    records = cursor.fetchall()

    # Assign clusters and insert results
    print('Inserting clustering results...')
    for record in records:
        frame_id, weighted_count = record
        cluster_label = kmeans.predict([[weighted_count]])[0]
        congestion_level = cluster_to_level[cluster_label]
        
        cursor.execute("""
            INSERT INTO clusters 
            (frame_id, cluster_label, congestion_level)
            VALUES (%s, %s, %s)
        """, (frame_id, int(cluster_label), congestion_level))
    db.commit()

    # Print cluster statistics
    print('\nCluster Statistics:')
    print('Cluster Centers (weighted count values):')
    for i, center in enumerate(centers):
        print(f'Cluster {i}: {center:.2f}')

    print('\nCongestion Level Distribution:')
    cursor.execute("""
        SELECT congestion_level, COUNT(*) as count 
        FROM clusters 
        GROUP BY congestion_level 
        ORDER BY COUNT(*) DESC
    """)
    distribution = cursor.fetchall()
    total = sum(count for _, count in distribution)
    
    for level, count in distribution:
        percentage = (count / total) * 100
        print(f'{level}: {count} records ({percentage:.2f}%)')

    print('\nClustering completed successfully')

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