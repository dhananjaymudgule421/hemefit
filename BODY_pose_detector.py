import cv2
import numpy as np
import mediapipe as mp
import time

from PIL import Image
import io

from BODY_PoseModule import poseDetector
from feedback import *
from utils import *

def live_pose_tracking(reps_joint,selected_joints,selected_distances,duration,min_angle, max_angle,upper_threshold,lower_threshold):
   
    detector = poseDetector()
    cap = cv2.VideoCapture(0)  # default webcam

    rep_count = 0
    rep_direction = 0
    angles_data = {joint_name: {"max": float('-inf'), "min": float('inf')} for joint_name in selected_joints}
    distances_data = {joint_name: {"max": float('-inf'), "min": float('inf')} for joint_name in selected_distances}
    start_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            break

        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            break
        img = cv2.flip(img, 1)
        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            for joint_name in selected_joints:
                joint_points = JOINTS_FLIPPED[joint_name]
                pose_points = [lmList[p][1:] for p in joint_points]

                angleDeg = detector.findAngle(joint_points)

                if joint_name == reps_joint:
                    # print(joint_name)
                    percentage = np.interp(angleDeg, [min(min_angle, max_angle), max(min_angle, max_angle)], [100, 0])
                    bar = np.interp(angleDeg, [min(min_angle, max_angle), max(min_angle, max_angle)], [100, 650])

                    rep_direction, rep_count, color = detector.update_count_and_direction(upper_threshold,lower_threshold,rep_direction, rep_count, percentage)
                    # draw bar
                    img = draw_percentage_bar(img, bar, percentage, color)
                    # draw count
                    img = draw_count_circle(img, int(rep_count))
                    
                img = draw_poses_on_frame(img, pose_points, angleDeg)

                angles_data[joint_name]["max"] = max(angles_data[joint_name]["max"], angleDeg)
                angles_data[joint_name]["min"] = min(angles_data[joint_name]["min"], angleDeg)

            # display_angles_data = [(name, data["max"], data["min"]) for name, data in angles_data.items()]
            # img = draw_angles_on_frame(img, display_angles_data)

            for joint_name_d in selected_distances:
                joint_points_d = DISTANCES_FLIPPED[joint_name_d]
                joint_distance = detector.findDistance(joint_points_d)

                distances_data[joint_name_d]["max"] = max(distances_data[joint_name_d]["max"], joint_distance)
                distances_data[joint_name_d]["min"] = min(distances_data[joint_name_d]["min"], joint_distance)
                # print(distances_data)

        # Convert the frame to BytesIO object
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        yield image_bytes,angles_data,distances_data





def video_pose_tracking(reps_joint,selected_joints,selected_distances, video_path=None):
    if not video_path:
        print("Please provide a video path for this function.")
        return

    detector = poseDetector()
    cap = cv2.VideoCapture(video_path)
    
    angles_data = {joint_name: {"max": float('-inf'), "min": float('inf')} for joint_name in selected_joints}
    distances_data = {joint_name: {"max": float('-inf'), "min": float('inf')} for joint_name in selected_distances}

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            for joint_name in selected_joints:
                    
                joint_points = JOINTS_ORIGINAL[joint_name]
                pose_points = [lmList[p][1:] for p in joint_points]

                angleDeg = detector.findAngle(joint_points)
                # print(f"Joint name:{joint_name}, Angle: {angleDeg}")        
                img = draw_poses_on_frame(img, pose_points, angleDeg)

                angles_data[joint_name]["max"] = max(angles_data[joint_name]["max"], angleDeg)
                angles_data[joint_name]["min"] = min(angles_data[joint_name]["min"], angleDeg)

            display_angles_data = [(name, data["max"], data["min"]) for name, data in angles_data.items()]
            img = draw_angles_on_frame(img, display_angles_data)

            for joint_name_d in selected_distances:
                joint_points_d = DISTANCES_ORIGINAL[joint_name_d]
                joint_distance = detector.findDistance(joint_points_d)

                distances_data[joint_name_d]["max"] = max(distances_data[joint_name_d]["max"], joint_distance)
                distances_data[joint_name_d]["min"] = min(distances_data[joint_name_d]["min"], joint_distance)

        # Convert the frame to BytesIO object
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        yield image_bytes,angles_data,distances_data












