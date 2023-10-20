import streamlit as st
import os
import numpy as np
import pandas as pd
from BODY_pose_detector import *
from utils import *

# Constants
video_library_dir = "video_library"


def select_joints(joints_dict):
    """Function to select multiple joints from the given dictionary."""
    
    # Note: The returned list contains the names of the selected joints
    selected_joints = st.sidebar.multiselect("Select Joints", list(joints_dict.keys()))

    # Convert selected joint names to their corresponding points
    # selected_joints = [joints_dict[joint_name] for joint_name in selected_joints]
    
    return selected_joints

def select_distances(distances_dict):
    """Function to select multiple distances from the given dictionary."""
    # Note: The returned list contains the names of the selected distances
    selected_distances = st.sidebar.multiselect("Select Distances", list(distances_dict.keys()))
    return selected_distances


def get_video_files():
    """Get all video files from the video library directory."""
    return [f for f in os.listdir(video_library_dir) if f.endswith((".mp4", ".avi"))]

def display_data_as_table(data, index_name):
    """
    Converts the given data to a DataFrame, sets the column names, and displays it in Streamlit.
    Parameters:
    - data (dict): Data to be displayed.
    - index_name (str): Name to be set for the index column.
    Returns:
    - None
    """
    df = pd.DataFrame(data).T.reset_index()
    df = df.rename(columns={"index": index_name, "max": "Max", "min": "Min"})
    st.table(df)




def display_video_analysis():
    """Display functionality for video analysis."""
    selected_joints = select_joints(JOINTS_ORIGINAL)
    selected_distances = select_distances(DISTANCES_ORIGINAL)  # Added this line


    st.sidebar.subheader("Select Exercise")
    selected_video = st.sidebar.selectbox("Select an exercise from the library", video_files)
    if selected_video:
        selected_video_path = os.path.join(video_library_dir, selected_video)
        st.sidebar.video(selected_video_path)

    start_video = st.sidebar.button("`ANALYSE EXERCISE`")

    if start_video:
        if selected_video_path:
            video_frame = st.empty()
            angles_data = dict()
            distances_data = dict()
            for frame_video,angles_data, distances_data in video_pose_tracking(selected_joints,selected_distances, selected_video_path):
                
                video_frame.image(frame_video, use_column_width=True)

            # Display angles and distances data
            display_data_as_table(angles_data, "JOINT_ANGLES")
            display_data_as_table(distances_data, "JOINT_DISTANCES")

        else:
            st.warning("Please select a video from the library to begin.")
    else:
        st.write("Press `ANALYSE EXERCISE` to view pose tracking in action.")

def display_live_analysis():
    """Display functionality for live webcam analysis."""
    # In the Streamlit app file:
    duration = st.sidebar.number_input("Enter analysis duration (seconds):", min_value=1, max_value=600, value=30)
    selected_joints = select_joints(JOINTS_FLIPPED)
    selected_distances = select_distances(DISTANCES_FLIPPED)  # Added this line
    reps_joint = st.sidebar.selectbox("Select Joint for reps count", list(JOINTS_FLIPPED.keys()))



    max_angle = st.sidebar.number_input("Maximum Angle", min_value=0, max_value=180, value=160, step=1)
    min_angle = st.sidebar.number_input("Minimum Angle", min_value=0, max_value=180, value=40, step=1)
    upper_threshold = st.sidebar.slider('Upper Threshold (%)', 80, 100, 95)
    lower_threshold = st.sidebar.slider('Lower Threshold (%)', 0, 10, 5)

    start_pose_matching = st.sidebar.button("`START EXERCISE`")

    if start_pose_matching:
        webcam_frame = st.empty()
        angles_data = dict()
        distances_data = dict()
        for frame_webcam,angles_data, distances_data in live_pose_tracking(reps_joint,selected_joints,selected_distances,duration,min_angle, max_angle,upper_threshold,lower_threshold,):
            webcam_frame.image(frame_webcam, use_column_width=True) 

        # Display angles and distances data
        display_data_as_table(angles_data, "JOINT_ANGLES")
        display_data_as_table(distances_data, "JOINT_DISTANCES")  
    else:
        st.write("Click `START EXERCISE` to begin.")

# Main

st.markdown('<style>h1{color: #007BFF;}</style>', unsafe_allow_html=True)
st.title('💪HemeHealth  F🏋️‍♂️T')
st.markdown("---")
st.sidebar.header("`Contol Panel`")

# Sidebar setup
page = st.sidebar.radio("Choose an Action", ["Video Analysis", "Live Analysis"])

# Fetch video files
video_files = get_video_files()

# Depending on the user's choice, display the appropriate page
if page == "Video Analysis":
    display_video_analysis()
elif page == "Live Analysis":
    display_live_analysis()
