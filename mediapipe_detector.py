"""
MediaPipe Hands detector for better hand keypoint detection.
This replaces DWPose for hand-only images.
"""

import numpy as np
from PIL import Image

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not installed. Run: pip install mediapipe")


class MediaPipeHandDetector:
    """
    MediaPipe Hands detector wrapper for hand keypoint detection.
    Returns 21 hand landmarks in the same format as DWPose.
    """

    def __init__(self,
                 model_path=None,
                 num_hands=2,
                 min_detection_confidence=0.35,
                 min_tracking_confidence=0.35):
        """
        Initialize MediaPipe Hands detector.

        Args:
            model_path: Path to hand_landmarker.task model (None = download default)
            num_hands: Maximum number of hands to detect (default: 2)
            min_detection_confidence: Minimum confidence for detection (default: 0.35, optimized from 0.5)
            min_tracking_confidence: Minimum confidence for tracking (default: 0.35, optimized from 0.5)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not installed. Run: pip install mediapipe")

        # Download model if needed
        if model_path is None:
            model_path = self._download_model()

        # Create detector
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _download_model(self):
        """
        Download MediaPipe hand_landmarker.task model.
        Returns path to downloaded model.
        """
        import os
        import urllib.request

        # Model URL (MediaPipe official)
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

        # Download to same directory as this file
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "hand_landmarker.task")

        if not os.path.exists(model_path):
            print(f"Downloading MediaPipe hand model to {model_path}...")
            urllib.request.urlretrieve(model_url, model_path)
            print("Download complete!")

        return model_path

    def detect_hands(self, image, return_format="dwpose"):
        """
        Detect hands in image and return keypoints.

        Args:
            image: PIL Image or numpy array (RGB)
            return_format: "dwpose" (list of [x,y]) or "mediapipe" (landmark objects)

        Returns:
            dict with keys:
                - "left": List of 21 [x, y] keypoints for left hand (or None)
                - "right": List of 21 [x, y] keypoints for right hand (or None)
                - "handedness": List of detected hand types ["Left", "Right"]
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Ensure RGB format (HWC)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[-1] == 4:  # RGBA
            image_np = image_np[:, :, :3]

        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np.copy())

        # Detect hands
        detection_result = self.detector.detect(mp_image)

        # Extract results
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness

        # Get image dimensions
        H, W = image_np.shape[:2]

        # Organize by left/right hand
        result = {
            "left": None,
            "right": None,
            "handedness": []
        }

        if len(hand_landmarks_list) == 0:
            return result

        for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
            # Get hand type ("Left" or "Right")
            hand_type = handedness[0].category_name  # "Left" or "Right"
            result["handedness"].append(hand_type)

            # Convert landmarks to keypoints
            if return_format == "dwpose":
                # DWPose format: list of [x, y] in pixel coordinates
                keypoints = []
                for landmark in hand_landmarks:
                    x = landmark.x * W
                    y = landmark.y * H
                    keypoints.append([x, y])
            else:
                # MediaPipe format: landmark objects with x, y, z (normalized 0-1)
                keypoints = hand_landmarks

            # Store by hand type
            if hand_type == "Left":
                result["left"] = keypoints
            elif hand_type == "Right":
                result["right"] = keypoints

        return result

    def detect_single_hand(self, image, side="left", return_format="dwpose"):
        """
        Detect a single hand (left or right) in image.

        Args:
            image: PIL Image or numpy array (RGB)
            side: "left" or "right"
            return_format: "dwpose" or "mediapipe"

        Returns:
            List of 21 keypoints or None if not detected
        """
        result = self.detect_hands(image, return_format=return_format)
        return result[side]

    def close(self):
        """Close the detector and free resources."""
        if hasattr(self, 'detector'):
            self.detector.close()
