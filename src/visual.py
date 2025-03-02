import cv2
import numpy as np
from .result import GazeResultContainer


def draw_gaze(x, y, w, h, image, pitchyaw, thickness=2, color=(255, 255, 0), scale=2.0):
    """
    Draws a gaze vector on the image.

    Args:
        x, y (int): Top-left corner of the bounding box.
        w, h (int): Width and height of the bounding box.
        image (np.ndarray): Input image.
        pitchyaw (tuple): Gaze angles (pitch, yaw).
        thickness (int, optional): Line thickness. Defaults to 2.
        color (tuple, optional): RGB color for the gaze line. Defaults to yellow.
        scale (float, optional): Scale factor for gaze length. Defaults to 2.0.

    Returns:
        np.ndarray: Image with gaze overlay.
    """
    (h_img, w_img) = image.shape[:2]
    center = (int(x + w / 2.0), int(y + h / 2.0))
    length = w * scale  # Gaze length based on bbox width

    # Convert grayscale to BGR if needed
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Compute gaze direction
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])

    # Draw gaze vector
    cv2.arrowedLine(
        image,
        tuple(np.round(center).astype(np.int32)),
        tuple(np.round([center[0] + dx, center[1] + dy]).astype(int)),
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.18
    )

    return image


def draw_bbox(frame: np.ndarray, bbox: np.ndarray, color=(0, 255, 0), thickness=1):
    """
    Draws a bounding box around detected faces.

    Args:
        frame (np.ndarray): Image frame.
        bbox (np.ndarray): Bounding box coordinates (x_min, y_min, x_max, y_max).
        color (tuple, optional): Bounding box color (default: green).
        thickness (int, optional): Line thickness (default: 1).

    Returns:
        np.ndarray: Image with bounding box drawn.
    """
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Ensure coordinates are within frame bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

    return frame


def render(frame: np.ndarray, results: GazeResultContainer):
    """
    Draws bounding boxes and gaze vectors on the frame.

    Args:
        frame (np.ndarray): Image frame.
        results (GazeResultContainer): Contains bounding boxes, pitch, and yaw data.

    Returns:
        np.ndarray: Processed image with visualized gaze data.
    """
    # Draw bounding boxes
    for bbox in results.bboxes:
        frame = draw_bbox(frame, bbox)

    # Draw gaze vectors
    for i, bbox in enumerate(results.bboxes):
        pitch = results.pitch[i]
        yaw = results.yaw[i]

        # Compute bounding box size
        x_min, y_min, x_max, y_max = map(int, bbox)
        x_min, y_min = max(0, x_min), max(0, y_min)  # Ensure valid coordinates

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        draw_gaze(x_min, y_min, bbox_width, bbox_height, frame, (pitch, yaw), color=(0, 0, 255))

    return frame
