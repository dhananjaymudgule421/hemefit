import streamlit as st
import cv2
import os
import numpy as np
import pickle
from pose_estimator import get_combined

# Constants (You can keep your existing constants here)
# ...

# Path to the video library directory
video_library_dir = "video_library"

# List video files in the directory
video_files = [f for f in os.listdir(video_library_dir) if f.endswith((".mp4", ".avi"))]

st.markdown('<style>h1{color: #007BFF;}</style>', unsafe_allow_html=True)
st.title('💪HemeHealth  F🏋️‍♂️T')
st.markdown("---")
st.sidebar.header("Settings")

# Add a dropdown to select a video from the video library
selected_video = st.sidebar.selectbox("Select an exercise from the library", video_files)

# Display the selected video in the sidebar
if selected_video:
    selected_video_path = os.path.join(video_library_dir, selected_video)
    st.sidebar.video(selected_video_path)

# Automatically choose the reference keypoints file based on the selected video's name
selected_keypoints = selected_video.replace(".mp4", ".pkl").replace(".avi", ".pkl")

# Initialize video capture and keypoints (set to None initially)
cap = None
prerecorded_keypoints = None

# Add a checkbox to start the pose matching
start_pose_matching = st.sidebar.button("START")

if start_pose_matching:
    if selected_video:
        # Construct the full path to the selected video and keypoints file
        selected_video_path = os.path.join(video_library_dir, selected_video)
        selected_keypoints_path = os.path.join(video_library_dir, selected_keypoints)


    if selected_video_path is None or selected_keypoints_path is None:
        st.warning("Please select a video from the library to begin.")
    else:
        video_frame = st.empty()

        for frame in get_combined(selected_video_path, selected_keypoints_path):
            video_frame.image(frame, use_column_width=True)

else:
    if cap is not None:
        cap.release()  # Release the video capture when pose matching is stopped.

    st.write("Click `START` to begin.")























