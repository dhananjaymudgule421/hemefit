import cv2 
import math
from utils import COLORS_BGR


def draw_count_circle(img, count, position=(170, 170), radius=120):
    """
    Draw the count in a circle on the image.

    Parameters:
    - img: The image on which to draw.
    - count: The count value to display.
    - position: The center position of the circle. Defaults to (120, 120) for top-left placement.
    - radius: The radius of the circle. Defaults to 120.

    Returns:
    - The image with the count drawn on it.
    """
    # Define the circle color based on the count value. Transitioning from light green to green.
    circle_color = (0, 255 - min(int(count) * 3, 255), 0)

    # Draw the circle
    cv2.circle(img, position, radius, circle_color, -1)

    # Define the font scale and thickness
    font_scale = 10
    font_thickness = 8

    # Get the size of the text
    text_size = cv2.getTextSize(str(int(count)), cv2.FONT_HERSHEY_PLAIN, font_scale, font_thickness)[0]

    # Calculate the bottom left corner of the text to place it centered inside the circle
    text_position = (position[0] - text_size[0] // 2, position[1] + text_size[1] // 2)

    # Draw the count text
    cv2.putText(img, str(int(count)), text_position, cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 0, 0), font_thickness)

    return img


def draw_percentage_bar(img, bar, percentage, color):
    """
    Draw a percentage bar on the image based on the given parameters.
    
    :param img: Image on which the bar is to be drawn
    :param bar: Length of the bar to be filled
    :param percentage: Percentage to be displayed near the bar
    :param color: Color of the bar
    :return: Image with the drawn bar
    """
     # To use a color:
    text_colour = COLORS_BGR["CYAN"]
    width = img.shape[1]
    bar_start_x = width - 200
    cv2.rectangle(img, (bar_start_x, 100), (bar_start_x + 75, 650), color, 3)
    cv2.rectangle(img, (bar_start_x, int(bar)), (bar_start_x + 75, 650), color, cv2.FILLED)
    cv2.putText(img, f'{int(percentage)}%', (bar_start_x, 75), cv2.FONT_HERSHEY_PLAIN, 4, text_colour, 4)
    
    return img


def draw_angles_on_frame(img, angles_data):
    """
    Draw the provided angles on the given image frame with distinct background rectangles.

    :param img: Image frame where angles will be drawn.
    :param angles_data: A list of tuples, where each tuple is (joint_name, max_angle, min_angle).
    :return: Image frame with drawn angles.
    """
    font_scale = 5  # Adjust this based on your needs
    font_thickness = 4  # Adjust this based on your needs
    font = cv2.FONT_HERSHEY_PLAIN
    padding = 15  # padding around the text

    # Define individual colors for max and min text and their backgrounds
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 255)  # Red

    # Initial positions
    x_start = 20
    y_start = 60
    y_gap = 20  # This is the vertical gap between the rectangles. Adjust this as needed.


    for joint_name, max_angle, min_angle in angles_data:
        max_angle_text = f"Max {joint_name}: {int(max_angle)}"
        min_angle_text = f"Min {joint_name}: {int(min_angle)}"
        (w_max, h_max), _ = cv2.getTextSize(max_angle_text, font, font_scale, font_thickness)
        (w_min, h_min), _ = cv2.getTextSize(min_angle_text, font, font_scale, font_thickness)

        # Compute max width for the rectangle
        max_width = max(w_max, w_min)

        # Draw rounded rectangles (background)
        cv2.rectangle(img, (x_start-padding, y_start-h_max-padding), (x_start+max_width+padding, y_start+padding), bg_color, -1)
        cv2.putText(img, max_angle_text, (x_start, y_start), font, font_scale, text_color, font_thickness)

        # Adjust starting point for the next angle
        y_start += h_max + y_gap

        cv2.rectangle(img, (x_start-padding, y_start-h_min-padding), (x_start+max_width+padding, y_start+padding), bg_color, -1)
        cv2.putText(img, min_angle_text, (x_start, y_start), font, font_scale, text_color, font_thickness)

        # Adjust starting point for the next joint's angles
        y_start += h_min + y_gap + 15  # Additional space between different joints

    return img




def draw_poses_on_frame(img, pose_points, angleDeg,joint_name, draw=True):
    """
    Draw poses and the angle on the given image frame.

    :param img: Image frame where poses will be drawn.
    :param pose_points: List of pose points as [(x1, y1), (x2, y2), (x3, y3)].
    :param angleDeg: Angle value to display.
    :param draw: If set to True, it draws the poses and angles on the frame.
    :return: Image frame with drawn poses and angle.
    """
    # Extract pose points
    (x1, y1), (x2, y2), (x3, y3) = pose_points

    # Drawing parameters
    line_color = COLORS_BGR["WHITE"]
    circle_color = COLORS_BGR["BLUE"]
    angle_text_color = COLORS_BGR["CYAN"]
    inner_circle_color = COLORS_BGR["ORANGE"]

    font_scale = 5
    font_thickness = 4
    
    if draw:
        # Drawing lines
        cv2.line(img, (x1, y1), (x2, y2), line_color, 3)
        cv2.line(img, (x3, y3), (x2, y2), line_color, 3)

        # Drawing circles
        cv2.circle(img, (x1, y1), 10, inner_circle_color, cv2.FILLED)
        cv2.circle(img, (x1, y1), 15, circle_color, 2)
        cv2.circle(img, (x2, y2), 10, inner_circle_color, cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, circle_color, 2)
        cv2.circle(img, (x3, y3), 10, inner_circle_color, cv2.FILLED)
        cv2.circle(img, (x3, y3), 15, circle_color, 2)

        # Drawing angle text
        if "LEFT" in joint_name:
            text_offset = (0, 50)
            cv2.putText(img, str(int(angleDeg)), (x2 - text_offset[0], y2 + text_offset[1]),
                        cv2.FONT_HERSHEY_PLAIN, font_scale, angle_text_color, font_thickness)
        else:
            text_offset = (150, 50)
            cv2.putText(img, str(int(angleDeg)), (x2 - text_offset[0], y2 + text_offset[1]),
                    cv2.FONT_HERSHEY_PLAIN, font_scale, angle_text_color, font_thickness)


    return img








