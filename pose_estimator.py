import cv2
import mediapipe as mp
import numpy as np
import pickle

from PIL import Image
import io




THRESHOLD = 0.3
MAX_DISTANCE = 0.5
CIRCLE_RADIUS = 100
MATCH_THRESHOLD = 0.1
STATUS_BAR_HEIGHT = 50

def compute_avg_distance(live_landmarks, prerecorded_landmarks):
    total_distance = 0
    landmark_count = len(live_landmarks.landmark)
    for i in range(landmark_count):
        live_landmark = live_landmarks.landmark[i]
        prerecorded_landmark = prerecorded_landmarks.landmark[i]
        dx = live_landmark.x - prerecorded_landmark.x
        dy = live_landmark.y - prerecorded_landmark.y
        dz = live_landmark.z - prerecorded_landmark.z
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        total_distance += distance
    return total_distance / landmark_count

def get_status_message(ratio, has_started):
    # if not has_started:
    #     return "Click START", (0, 0, 255)
    if ratio < MATCH_THRESHOLD:
        return "Please follow the video", (0, 0, 255)
    else:
        return "Great, Kepp Doing!", (0, 255, 0)

def add_status_bar(image, height, color):
    return cv2.copyMakeBorder(image, height, 0, 0, 0, cv2.BORDER_CONSTANT, value=color)

def load_keypoints(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print("Reference keypoints file not found. Exiting.")
        exit()


def concatenate_images(image1, image2):
    """
    Concatenate two images horizontally handling different heights.
    """
    height = max(image1.shape[0], image2.shape[0])
    width_total = image1.shape[1] + image2.shape[1]
    concatenated_image = np.zeros((height, width_total, 3), dtype=np.uint8)
    concatenated_image[:image1.shape[0], :image1.shape[1]] = image1
    concatenated_image[:image2.shape[0], image1.shape[1]:] = image2
    return concatenated_image

def get_feedback_circle_parameters(ratio):
    # Calculate the BGR values based on the ratio
    if ratio < 0.33:  # Going from red to orange
        red = 255
        green = int(165 * (ratio * 3))  # Scale ratio to [0, 1] for this third
    elif ratio < 0.66:  # Going from orange to yellow
        red = 255
        green = 165 + int((255-165) * ((ratio-0.33) * 3))  # Scale ratio to [0, 1] for this third
    else:  # Going from yellow to green
        red = int(255 * ((1 - ratio) * 3))  # Scale ratio to [0, 1] for this third
        green = 255
    
    color_value = (0, green, red)  # BGR instead of RGB
    
    # Calculate the arc's end angle based on the ratio
    end_angle = 360 * ratio

    return color_value, end_angle





def get_combined(selected_video_path, selected_keypoints_path):
    prerecorded_keypoints = load_keypoints(selected_keypoints_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    # prerecorded_video_path = 'ref_1.mp4'
    prerecorded_video = cv2.VideoCapture(selected_video_path)

    has_started = False
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        prerecorded_ret, prerecorded_frame = prerecorded_video.read()

        if not prerecorded_ret:
            prerecorded_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_num = 0
            prerecorded_ret, prerecorded_frame = prerecorded_video.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        annotated_image = frame.copy()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            ratio = 0
            status, color = get_status_message(ratio, has_started)
            if has_started and frame_num < len(prerecorded_keypoints):
                distance = compute_avg_distance(results.pose_landmarks, prerecorded_keypoints[frame_num])
                ratio = 1.0 - distance / MAX_DISTANCE
                ratio = min(1.0, max(0.0, ratio))
                status, color = get_status_message(ratio, has_started)


            annotated_image = add_status_bar(annotated_image, STATUS_BAR_HEIGHT, [0, 0, 0])

            # Feedback Circle Drawing
            center = (120, 120 + STATUS_BAR_HEIGHT)
            color_value, end_angle = get_feedback_circle_parameters(ratio)
            cv2.ellipse(annotated_image, center, (CIRCLE_RADIUS, CIRCLE_RADIUS), 0, 0, end_angle, color_value, -1)
            cv2.circle(annotated_image, center, CIRCLE_RADIUS, (255, 255, 255), 2)

            # Status message
            cv2.putText(annotated_image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        frame_num += 1

        prerecorded_frame = cv2.resize(prerecorded_frame, (frame.shape[1], frame.shape[0]))
        prerecorded_frame = add_status_bar(prerecorded_frame, STATUS_BAR_HEIGHT, [0, 0, 0])
        combined_frame = concatenate_images(annotated_image, prerecorded_frame)

        # Convert the frame to BytesIO object
        pil_image = Image.fromarray(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        yield image_bytes

