import cv2
import numpy as np
import pandas as pd
import os
import csv
import av
import glob


pl_input_folder = 'preprocessing/pl_input'
ad_video = glob.glob(os.path.join(pl_input_folder, '*.mp4'))[0]

# Sort the gaze data by filtering the start and end timestamps
gaze_data = pd.read_csv(os.path.join(pl_input_folder,'updated_gaze_data.csv'))

# one approach is to use MGM, create a df of gaze timestamps, based on frame timestamp

# Calculate the frame duration based on average_rate
input_container = av.open(ad_video)
input_stream = input_container.streams.video[0]
frame_rate = input_stream.average_rate
frame_duration_ns = int(1 / frame_rate * 1e9)

first_frame = next(input_container.decode(video=0))  # Get the first frame
worldCameraVid_width, worldCameraVid_height = first_frame.width, first_frame.height

# Find the start timestamp of the video
video_start_timestamp = gaze_data['timestamp [ns]'].min()

gaze_preprocessed = []

# Iterate over frames in the input video
for frame_idx, frame in enumerate(input_container.decode(video=0)):
    # Convert the PyAV frame to a PIL image for drawing and to RGB explicitly (to avoid the warning)
    img = frame.to_image().convert("RGB")

    # Calculate the timestamp for the current frame
    current_frame_timestamp = video_start_timestamp + frame_idx * frame_duration_ns

    # Draw AOI for the current frame if available

    for i in range(2):
        timestamp = current_frame_timestamp + i * frame_duration_ns//2

        # Normalize timestamp relative to video_start_timestamp
        normalized_timestamp = timestamp - video_start_timestamp

        # Find gaze points for the current frame and calculate the average
        gaze_points = gaze_data[(gaze_data['timestamp [ns]'] >= current_frame_timestamp) &
                                (gaze_data['timestamp [ns]'] < current_frame_timestamp + frame_duration_ns)]

        if not gaze_points.empty:
            # Average the gaze points for this frame
            avg_gaze_x = int(gaze_points['gaze x [px]'].mean())
            avg_gaze_y = int(gaze_points['gaze y [px]'].mean())
            is_within_aoi = gaze_points['within_aoi'].mean() > 0.5  # If majority of points are inside AOI

            # Normalize gaze coordinates
            norm_gaze_x = avg_gaze_x / worldCameraVid_width
            norm_gaze_y = avg_gaze_y / worldCameraVid_height

            timestamp_ms = timestamp/1e6
            gaze_preprocessed.append({
                "timestamp": normalized_timestamp,
                "frame_idx": frame_idx,
                "confidence": 1.0,
                "norm_pos_x": norm_gaze_x,
                "norm_pos_y": norm_gaze_y,
            })

output_folder = 'preprocessing/pl_preprocessed_out'
# Save preprocessed gaze data to a CSV file
with open(os.path.join(output_folder, 'gaze_preprocessed.csv'), 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'frame_idx', 'confidence', 'norm_pos_x', 'norm_pos_y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for data in gaze_preprocessed:
        writer.writerow(data)
