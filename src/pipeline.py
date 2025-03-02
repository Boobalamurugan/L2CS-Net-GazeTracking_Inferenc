import pathlib
from typing import Union, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from face_detection import RetinaFace

from .utils import prep_input_numpy, getArch
from .result import GazeResultContainer


class Pipeline:
    def __init__(
        self, 
        weights: pathlib.Path, 
        arch: str,
        # device: str = "cpu", 
        include_detector: bool = True,
        confidence_threshold: float = 0.5
    ) -> None:
        """
        Initializes the gaze estimation pipeline.

        Args:
            weights (pathlib.Path): Path to the model weights.
            arch (str): Model architecture.
            device (str): 'cpu' or 'cuda' for GPU acceleration.
            include_detector (bool): Whether to use face detection.
            confidence_threshold (float): Minimum confidence score for face detection.
        """
        self.weights: pathlib.Path = weights
        self.include_detector: bool = include_detector
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔹 Running on: {'GPU' if self.device == 'cuda' else 'CPU'}")
        self.confidence_threshold: float = confidence_threshold

        # Create L2CS model
        self.model: nn.Module = getArch(arch, 90)
        self.model.load_state_dict(torch.load(self.weights, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        # Create RetinaFace if face detection is enabled
        self.detector: Optional[RetinaFace] = None
        if self.include_detector:
            self.detector = RetinaFace(gpu_id=0 if self.device == "cuda" else -1)


        # Softmax for probability conversion
        self.softmax: nn.Softmax = nn.Softmax(dim=1)
        
        # Create tensor index for gaze bins (0-89)
        self.idx_tensor: torch.Tensor = torch.arange(90, dtype=torch.float32, device=self.device)

    def step(self, frame: np.ndarray) -> GazeResultContainer:
        """
        Processes a single frame to detect faces and estimate gaze.

        Args:
            frame (np.ndarray): Input image (BGR format).

        Returns:
            GazeResultContainer: Contains gaze angles, bounding boxes, and landmarks.
        """
        face_imgs: List[np.ndarray] = []
        bboxes: List[np.ndarray] = []
        landmarks: List[np.ndarray] = []
        scores: List[float] = []

        # Run face detection if enabled
        if self.include_detector and self.detector is not None:
            faces = self.detector(frame)
            if faces is not None:
                for box, landmark, score in faces:
                    if score < self.confidence_threshold:
                        continue

                    # Extract and ensure non-negative bounding box coordinates
                    x_min, y_min = max(0, int(box[0])), max(0, int(box[1]))
                    x_max, y_max = int(box[2]), int(box[3])

                    # Crop face region
                    img: np.ndarray = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    # Store bounding box, landmarks, and confidence score
                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

                # If faces were detected, predict gaze
                pitch, yaw = self.predict_gaze(np.stack(face_imgs)) if face_imgs else (np.empty((0, 1)), np.empty((0, 1)))

        else:
            # Directly predict gaze without face detection
            pitch, yaw = self.predict_gaze(frame)

        # Package results
        results = GazeResultContainer(
            pitch=pitch,
            yaw=yaw,
            bboxes=np.array(bboxes) if bboxes else np.empty((0, 4)),
            landmarks=np.array(landmarks) if landmarks else np.empty((0, 5, 2)),
            scores=np.array(scores) if scores else np.empty((0,))
        )

        return results

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts gaze (yaw & pitch) from an input image or tensor.

        Args:
            frame (Union[np.ndarray, torch.Tensor]): Input image(s) as numpy array or tensor.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted pitch and yaw angles in radians.
        """
        # Convert input to a torch tensor if it's a numpy array
        if isinstance(frame, np.ndarray):
            img: torch.Tensor = prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame
        else:
            raise TypeError("Invalid input type. Expected np.ndarray or torch.Tensor.")

        # Forward pass through L2CS model
        gaze_pitch, gaze_yaw = self.model(img)

        # Convert model output using softmax
        pitch_prob: torch.Tensor = self.softmax(gaze_pitch)
        yaw_prob: torch.Tensor = self.softmax(gaze_yaw)

        # Compute weighted sum to get continuous angles
        pitch_predicted: torch.Tensor = torch.sum(pitch_prob * self.idx_tensor, dim=1) * 4 - 180
        yaw_predicted: torch.Tensor = torch.sum(yaw_prob * self.idx_tensor, dim=1) * 4 - 180

        # Convert degrees to radians
        pitch_final: np.ndarray = (pitch_predicted.cpu().detach().numpy() * np.pi / 180.0)
        yaw_final: np.ndarray = (yaw_predicted.cpu().detach().numpy() * np.pi / 180.0)

        return pitch_final, yaw_final
