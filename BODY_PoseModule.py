import cv2 
import mediapipe as mp
import time 
import math
import numpy as np

class poseDetector():
    def __init__(self, mode=False, model_complexity=1, smooth=True, 
                 enable_segmentation=False, smooth_segmentation=True, 
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.model_complexity,
                                     smooth_landmarks=self.smooth,
                                     enable_segmentation=self.enable_segmentation,
                                     smooth_segmentation=self.smooth_segmentation,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self,img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw = True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)

        return self.lmList

    def findAngle(self, joint_points):
        p1,p2,p3 = joint_points
        # get the landmark
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        x3,y3 = self.lmList[p3][1:]
        # Calculate the angle using the arctangent of the slopes
        angleRad = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        angleDeg = abs(math.degrees(angleRad))
        # If you want to keep the angle within 0-180 degrees
        if angleDeg > 180.0:
            angleDeg = 360 - angleDeg

        return angleDeg

    def findDistance(self, joint_points_d):
        point1,point2 = joint_points_d
        x1, y1 = self.lmList[point1][1:]
        x2, y2 = self.lmList[point2][1:]
        
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance


    def update_count_and_direction1(self, movement_phase, count, percentage):
        color = (255, 0, 255)

        # Check with a range/buffer for percentage
        if 95 <= percentage <= 100:
            color = (0, 255, 0)
            if movement_phase == 0:
                count += 0.5
                dir = 1

        elif 0 <= percentage <= 5:
            color = (0, 255, 0)
            if movement_phase == 1:
                count += 0.5
                dir = 0

        return movement_phase, count, color

    def update_count_and_direction(self, upper_threshold,lower_threshold,rep_direction, rep_count, percentage):
        default_color = (255, 255, 255)  # Default color is white

        # Check with a range/buffer for percentage
        if upper_threshold <= percentage <= 100:
            color = (0, 255, 0)  # Green for upper threshold
            if rep_direction == 0:
                rep_count += 0.5
                rep_direction = 1

        elif 0 <= percentage <= lower_threshold:
            color = (0, 0, 255)  # Red for lower threshold
            if rep_direction == 1:
                rep_count += 0.5
                rep_direction = 0
        else:
            color = default_color  # Assign default color if no condition is met

        return rep_direction, rep_count, color




    




def main():
    # cap = cv2.VideoCapture('video_library/ref_1.mp4')
    cap = cv2.VideoCapture(0)

    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList):
            print(lmList[14])
            cv2.circle(img,(lmList[14][1],lmList[14][2]),25,(0,0,255),cv2.FILLED)
            # cv2.circle(img,(lmList[12][1],lmList[14][1]),25,(0,0,255),cv2.FILLED)
            # cv2.circle(img,(lmList[16][1],lmList[14][1]),25,(0,0,255),cv2.FILLED)

    
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        img = cv2.flip(img, 1)

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
