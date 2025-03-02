import argparse
import pathlib
import time
import cv2

import torch
import torch.backends.cudnn as cudnn

from src.pipeline import Pipeline
from src.visual import render

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

# Set Current Working Directory
CWD = pathlib.Path.cwd()

def parse_args():
    """
    Parse input arguments for gaze estimation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Gaze evaluation using L2CS-Net trained on Gaze360.'
    )
    parser.add_argument(
        '--model_path', help='Path to model checkpoint.', 
        default=str(CWD / 'models' / 'L2CSNet_gaze360.pkl'), type=str
    )
    parser.add_argument(
        '--video_path', help='Path to video file or camera device ID (default: 0 for webcam).', 
        default='0', type=str
    )
    return parser.parse_args()

def main():
    """Main function for real-time gaze tracking."""
    args = parse_args()

    # Enable cuDNN for optimized performance
    cudnn.enabled = True

    # Initialize gaze estimation pipeline
    gaze_pipeline = Pipeline(
        weights=args.model_path,
        arch='ResNet50'
    )

    # Open video or webcam
    cap = cv2.VideoCapture(int(args.video_path) if args.video_path.isdigit() else args.video_path)

    if not cap.isOpened():
        raise IOError("ERROR: Cannot open video or webcam.")

    print("Video/Webcam successfully opened. Press 'q' to exit.")

    with torch.no_grad():
        while True:
            # Get frame
            success, frame = cap.read()
            start_time = time.time()

            if not success:
                print("Warning: Failed to obtain frame. Retrying...")
                time.sleep(0.1)
                continue  # Skip to next iteration

            # Process frame
            results = gaze_pipeline.step(frame)

            # Render visualization
            frame = render(frame, results)

            # Display FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(
                frame, f'FPS: {fps:.1f}', (10, 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA
            )

            # Show output
            cv2.imshow("Gaze Tracking Demo", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
