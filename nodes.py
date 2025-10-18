import torch
import numpy as np
from PIL import Image, ImageDraw
import math
import traceback
from typing import Dict, List

# --- Optional Imports ---
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from .mediapipe_detector import MediaPipeHandDetector
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[Smart Hands Replace] MediaPipe not available. Install with: pip install mediapipe")

try:
    from .hand_anatomy import HandEstimator
    HAND_ESTIMATOR_AVAILABLE = True
except ImportError:
    HAND_ESTIMATOR_AVAILABLE = False
    print("[Smart Hands Replace] HandEstimator not available")
# --------------------------

# --- Constants ---
# Full-body detection: bbox_ratio > threshold indicates full-body skeleton (not hand-only)
# Typical full-body hand bbox is 3-5x smaller than actual hand
FULLBODY_DETECTION_THRESHOLD = 3.0

# Baseline hand coverage: DWPose keypoint bbox typically covers 10% of actual hand
# Used to estimate actual hand size from keypoint bbox (multiply by 10x)
BASELINE_HAND_COVERAGE = 0.1

# Hand size multiplier: Apply to final hand area for size adjustment
# 1.0 = exact base pose size, >1.0 = larger, <1.0 = smaller
HAND_SIZE_MULTIPLIER = 1.2
# --------------------------


class ComposeMultipleImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "cell_size": ("INT", {"default": 256, "min": 64, "max": 1024}),
                "padding": ("INT", {"default": 10, "min": 0, "max": 100}),
                "background_color": (["black", "white", "gray"],),
                "auto_canvas_size": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "grid_columns": ("INT", {"default": 0, "min": 0, "max": 20}),
                "grid_rows": ("INT", {"default": 0, "min": 0, "max": 20}),
                "canvas_width": ("INT", {"default": 512, "min": 128, "max": 4096}),
                "canvas_height": ("INT", {"default": 512, "min": 128, "max": 4096}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compose"
    CATEGORY = "image/hands"

    def compose(self, images, cell_size, padding, background_color, auto_canvas_size,
                grid_columns=0, grid_rows=0, canvas_width=512, canvas_height=512):

        num_images = images.shape[0]
        if num_images == 0:
            return (torch.zeros((1, canvas_height, canvas_width, 3)),)

        if grid_columns == 0 and grid_rows == 0:
            grid_columns = int(np.ceil(np.sqrt(num_images)))
            grid_rows = int(np.ceil(num_images / grid_columns))
        elif grid_columns == 0:
            grid_columns = int(np.ceil(num_images / grid_rows))
        elif grid_rows == 0:
            grid_rows = int(np.ceil(num_images / grid_columns))

        if auto_canvas_size:
            canvas_width = grid_columns * (cell_size + padding) + padding
            canvas_height = grid_rows * (cell_size + padding) + padding

        bg_colors = {"black": (0,0,0), "white": (255,255,255), "gray": (127,127,127)}
        canvas = Image.new("RGB", (canvas_width, canvas_height), bg_colors[background_color])

        for idx, image_tensor in enumerate(images):
            img_pil = self.tensor_to_pil(image_tensor)
            img_pil = img_pil.resize((cell_size, cell_size), Image.LANCZOS)

            col = idx % grid_columns
            row = idx // grid_columns

            x = padding + col * (cell_size + padding)
            y = padding + row * (cell_size + padding)

            if x + cell_size <= canvas_width and y + cell_size <= canvas_height:
                canvas.paste(img_pil, (x, y), img_pil if img_pil.mode == 'RGBA' else None)

        return (self.pil_to_tensor(canvas),)

    def tensor_to_pil(self, tensor):
        return Image.fromarray((tensor.cpu().numpy() * 255).astype(np.uint8))

    def pil_to_tensor(self, pil_img):
        return torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0).unsqueeze(0)

class CoordinateTransform:
    """
    Coordinate system conversion helper.
    Handles canvas space ↔ image space ↔ relative (0.0-1.0) space.
    """

    def __init__(self, canvas_size, image_size):
        """
        Args:
            canvas_size: (width, height) tuple - DWPose coordinate system
            image_size: (width, height) tuple - Actual PIL image size
        """
        self.canvas_size = canvas_size
        self.image_size = image_size
        self.scale_x = image_size[0] / canvas_size[0] if canvas_size[0] > 0 else 1.0
        self.scale_y = image_size[1] / canvas_size[1] if canvas_size[1] > 0 else 1.0

    def canvas_to_relative(self, kp):
        """Convert keypoint from canvas coordinates to relative (0.0-1.0)."""
        return [kp[0] / self.canvas_size[0], kp[1] / self.canvas_size[1]]

    def relative_to_image(self, kp_rel):
        """Convert keypoint from relative (0.0-1.0) to image pixel coordinates."""
        return [kp_rel[0] * self.image_size[0], kp_rel[1] * self.image_size[1]]

    def canvas_to_image(self, kp):
        """Direct conversion: canvas → image coordinates."""
        return [kp[0] * self.scale_x, kp[1] * self.scale_y]


class SmartHandsReplace:
    """
    (V11 - Simplified Interface) Hand composition with automatic keypoint detection.
    Uses normalized (0.0-1.0) coordinate space for resolution-independent transformations.

    V11 Features:
    - Automatic hand keypoint detection (no manual keypoint input required)
    - 3-stage detection pipeline: MediaPipe → Phase2 → Basepose
    - Resolution-independent coordinate transformations
    - Simplified workflow: only hand images + base keypoints needed
    """

    def __init__(self):
        """Initialize detectors for internal hand keypoint detection."""
        self.mediapipe_detector = None
        self.hand_estimator = None

        # Initialize MediaPipe detector
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mediapipe_detector = MediaPipeHandDetector(
                    min_detection_confidence=0.35,
                    min_tracking_confidence=0.35
                )
            except Exception as e:
                print(f"[Smart Hands Replace] Failed to initialize MediaPipe: {e}")

        # Initialize Phase 2 estimator
        if HAND_ESTIMATOR_AVAILABLE:
            try:
                self.hand_estimator = HandEstimator()
            except Exception as e:
                print(f"[Smart Hands Replace] Failed to initialize HandEstimator: {e}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "base_keypoints": ("POSE_KEYPOINT",),
                "left_hand_image": ("IMAGE",),
                "right_hand_image": ("IMAGE",),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "erase_expansion": ("INT", {"default": 10, "min": 0, "max": 50}),
                "line_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
                "enable_debug_logging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine"
    CATEGORY = "image/hands"

    def combine(self, base_image, base_keypoints, left_hand_image, right_hand_image,
                blend_strength, erase_expansion, line_thickness, enable_debug_logging):

        batch_size = base_image.shape[0]
        results = []
        for i in range(batch_size):
            base_pil = self.tensor_to_pil(base_image[i])
            result_pil = base_pil.copy()

            # Convert hand images to PIL
            left_hand_pil = self.tensor_to_pil(left_hand_image[i])
            right_hand_pil = self.tensor_to_pil(right_hand_image[i])

            # Auto-detect hand keypoints using internal detection pipeline
            # Detect with actual side (MediaPipe/Phase2 detect what's in the image)
            left_hand_kps = self._detect_hand_keypoints(
                left_hand_pil, side="left", base_keypoints=base_keypoints,
                enable_debug_logging=enable_debug_logging
            )

            right_hand_kps = self._detect_hand_keypoints(
                right_hand_pil, side="right", base_keypoints=base_keypoints,
                enable_debug_logging=enable_debug_logging
            )

            # Process both hands with detected keypoints
            # No swapping needed - detection already uses correct sides
            for side, hand_pil, hand_kps in [("left", left_hand_pil, left_hand_kps),
                                             ("right", right_hand_pil, right_hand_kps)]:
                try:
                    result_pil = self.composite_hand(
                        result_pil, hand_pil, base_keypoints, hand_kps, side,
                        blend_strength, erase_expansion, line_thickness, enable_debug_logging
                    )
                except Exception as e:
                    print(f"[ERROR] {side.upper()} hand composite failed: {e}")
                    traceback.print_exc()

            results.append(self.pil_to_tensor(result_pil))

        return (torch.cat(results, dim=0),)

    def _detect_hand_keypoints(self, hand_pil, side, base_keypoints, enable_debug_logging):
        """
        Internal hand keypoint detection pipeline with fallback chain.

        Pipeline:
        1. MediaPipe (0.35 threshold) - Best accuracy for hand-only images
        2. Phase 2 Estimation - Uses wrist + MCPs from base_keypoints
        3. Basepose Fallback - Uses base_keypoints directly

        Args:
            hand_pil: PIL Image of hand
            side: "left" or "right"
            base_keypoints: Full-body skeleton for fallback
            enable_debug_logging: Debug output

        Returns:
            POSE_KEYPOINT format dict or None
        """
        if enable_debug_logging:
            print(f"\n[{side.upper()} HAND DETECTION] Image: {hand_pil.width}×{hand_pil.height}")

        # --- Step 1: Try MediaPipe Detection ---
        if self.mediapipe_detector is not None:
            try:
                if enable_debug_logging:
                    print(f"  [1/3] Trying MediaPipe (threshold=0.35)...")

                result = self.mediapipe_detector.detect_single_hand(
                    hand_pil, side=side, return_format="dwpose"
                )

                if enable_debug_logging:
                    if result is not None:
                        if len(result) == 21:
                            print(f"  ✅ MediaPipe SUCCESS: 21/21 keypoints detected")
                        else:
                            print(f"  ❌ MediaPipe FAILED: Got {len(result)} keypoints instead of 21")
                    else:
                        # Check if MediaPipe detected ANY hands (wrong side?)
                        all_hands_result = self.mediapipe_detector.detect_hands(hand_pil, return_format="dwpose")
                        detected_sides = [k for k, v in all_hands_result.items() if v is not None and k in ['left', 'right']]

                        if detected_sides:
                            print(f"  ⚠️ MediaPipe detected {detected_sides} but looking for '{side}'")
                        else:
                            print(f"  ❌ MediaPipe FAILED: No hand detected at all")

                if result is not None and len(result) == 21:
                    # Convert to POSE_KEYPOINT format
                    return self._format_as_pose_keypoint(result, hand_pil.width, hand_pil.height, side)

            except Exception as e:
                if enable_debug_logging:
                    print(f"  ❌ MediaPipe ERROR: {e}")
                    traceback.print_exc()

        # --- Step 2: Try Phase 2 Estimation ---
        if self.hand_estimator is not None and base_keypoints is not None:
            try:
                if enable_debug_logging:
                    print(f"  [2/3] Trying Phase 2 Estimation (wrist + MCPs)...")

                # Extract wrist + 3 MCPs from base_keypoints
                base_hand_kps = self._get_hand_kps(base_keypoints, side)
                if base_hand_kps and len(base_hand_kps) >= 21:
                    wrist = base_hand_kps[0]
                    index_mcp = base_hand_kps[5]
                    middle_mcp = base_hand_kps[9]
                    pinky_mcp = base_hand_kps[17]

                    # Check if anchor points are valid
                    valid_anchors = [
                        self.is_valid_keypoint(wrist),
                        self.is_valid_keypoint(index_mcp),
                        self.is_valid_keypoint(middle_mcp),
                        self.is_valid_keypoint(pinky_mcp)
                    ]

                    if sum(valid_anchors) >= 2:  # Need at least 2 valid anchors
                        # Run Phase 2 estimation
                        # Create partial detection with available anchors (21 keypoints)
                        partial_kps = [wrist if valid_anchors[0] else None] + [None] * 4  # 0-4: wrist, thumb
                        partial_kps += [index_mcp if valid_anchors[1] else None] + [None] * 3  # 5-8: index
                        partial_kps += [middle_mcp if valid_anchors[2] else None] + [None] * 3  # 9-12: middle
                        partial_kps += [None] * 4  # 13-16: ring
                        partial_kps += [pinky_mcp if valid_anchors[3] else None] + [None] * 3  # 17-20: pinky

                        estimated_kps = self.hand_estimator.estimate_missing_keypoints(
                            detected_keypoints=partial_kps,
                            basepose_keypoints=base_hand_kps,
                            side=side
                        )

                        if estimated_kps and len(estimated_kps) == 21:
                            if enable_debug_logging:
                                valid_count = sum(1 for kp in estimated_kps if self.is_valid_keypoint(kp))
                                print(f"  ✅ Phase 2 SUCCESS: {valid_count}/21 keypoints estimated")

                            return self._format_as_pose_keypoint(estimated_kps, hand_pil.width, hand_pil.height, side)

                if enable_debug_logging:
                    print(f"  ❌ Phase 2 FAILED: Insufficient anchor points")

            except Exception as e:
                if enable_debug_logging:
                    print(f"  ❌ Phase 2 ERROR: {e}")

        # --- Step 3: Fallback to Basepose ---
        if base_keypoints is not None:
            base_hand_kps = self._get_hand_kps(base_keypoints, side)

            if base_hand_kps:
                if enable_debug_logging:
                    valid_count = sum(1 for kp in base_hand_kps if self.is_valid_keypoint(kp))
                    print(f"  [3/3] Using Basepose fallback...")
                    print(f"  ⚠️ Basepose FALLBACK: {valid_count}/21 keypoints from full-body skeleton")

                # CRITICAL: Basepose keypoints are in base image coordinate system
                # Keep the base image canvas_size for correct coordinate transformation
                # The CoordinateTransform in composite_hand() will handle the resolution difference
                base_data = base_keypoints[0]
                base_canvas_w = base_data.get("canvas_width", hand_pil.width)
                base_canvas_h = base_data.get("canvas_height", hand_pil.height)

                return self._format_as_pose_keypoint(base_hand_kps, base_canvas_w, base_canvas_h, side)

            # If extraction failed, return None
            return None

        # --- All methods failed ---
        if enable_debug_logging:
            print(f"  ❌ ALL DETECTION METHODS FAILED")

        return None

    def _format_as_pose_keypoint(self, keypoints_list, canvas_width, canvas_height, side):
        """
        Convert keypoint list to POSE_KEYPOINT format compatible with DWPose.

        Args:
            keypoints_list: List of 21 [x, y] keypoints
            canvas_width: Canvas width for metadata
            canvas_height: Canvas height for metadata
            side: "left" or "right"

        Returns:
            List with single dict in POSE_KEYPOINT format
        """
        # Flatten keypoints to [x, y, confidence, x, y, confidence, ...] format
        flat_keypoints = []
        for kp in keypoints_list:
            if kp is not None and len(kp) >= 2:
                flat_keypoints.extend([kp[0], kp[1], 1.0])  # confidence = 1.0
            else:
                flat_keypoints.extend([0.0, 0.0, 0.0])  # invalid keypoint

        # Create POSE_KEYPOINT structure matching DWPose format
        pose_keypoint = [{
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "people": [{
                f"hand_{side}_keypoints_2d": flat_keypoints
            }]
        }]

        return pose_keypoint

    def composite_hand(self, base_pil, hand_pil, base_keypoints, hand_keypoints, side,
                       blend_strength, erase_expansion, line_thickness, enable_debug_logging):
        if enable_debug_logging:
            print(f"\n{'='*60}")
            print(f"[{side.upper()} HAND]")

        # 1. Get Target Hand Details from Base Pose
        base_info = self._get_hand_details(base_keypoints, side, enable_debug_logging, base_pil)
        if not base_info:
            if enable_debug_logging:
                print(f"[{side}] ❌ Failed to get base hand details (side={side})")
            return base_pil
        base_wrist_raw, base_mcp_raw, base_angle, base_bbox_area, base_canvas_size = base_info

        # 2. Get Source Hand Details from Hand Image
        source_info = self._get_hand_details(hand_keypoints, side, enable_debug_logging, hand_pil)
        if not source_info:
            if enable_debug_logging:
                print(f"[{side}] ❌ Failed to get source hand details (hand_keypoints side={side})")
            return base_pil
        source_wrist_raw, source_mcp_raw, source_angle, source_bbox_area, source_canvas_size = source_info

        # 3. Erase old hand from base image (including skeleton lines)
        self._erase_hand_from_base(base_pil, base_keypoints, side, erase_expansion, enable_debug_logging)

        # 4. Calculate Transformations
        # CRITICAL FIX: Canvas size mismatch auto-correction
        # Problem: DWPose may output canvas_size that doesn't match actual image size
        # This causes incorrect scale factors (e.g., 0.5 instead of 1.0)
        # Solution: If canvas doesn't match image, use image size as ground truth

        # Detect canvas mismatch (tolerance: 1px for rounding)
        base_canvas_mismatch = abs(base_pil.width - base_canvas_size[0]) > 1 or abs(base_pil.height - base_canvas_size[1]) > 1
        source_canvas_mismatch = abs(hand_pil.width - source_canvas_size[0]) > 1 or abs(hand_pil.height - source_canvas_size[1]) > 1

        # Use actual image size if mismatch detected
        effective_base_canvas = (base_pil.width, base_pil.height) if base_canvas_mismatch else base_canvas_size
        effective_source_canvas = (hand_pil.width, hand_pil.height) if source_canvas_mismatch else source_canvas_size

        actual_base_scale_x = base_pil.width / effective_base_canvas[0] if effective_base_canvas[0] > 0 else 1.0
        actual_base_scale_y = base_pil.height / effective_base_canvas[1] if effective_base_canvas[1] > 0 else 1.0
        actual_source_scale_x = hand_pil.width / effective_source_canvas[0] if effective_source_canvas[0] > 0 else 1.0
        actual_source_scale_y = hand_pil.height / effective_source_canvas[1] if effective_source_canvas[1] > 0 else 1.0

        if enable_debug_logging:
            if base_canvas_mismatch:
                print(f"[{side}] ⚠️ Base canvas mismatch: img={base_pil.width}×{base_pil.height} vs canvas={base_canvas_size}")
            if source_canvas_mismatch:
                print(f"[{side}] ⚠️ Source canvas mismatch: img={hand_pil.width}×{hand_pil.height} vs canvas={source_canvas_size}")

        # Scale bbox areas to actual image sizes
        scaled_base_bbox_area = base_bbox_area * actual_base_scale_x * actual_base_scale_y
        scaled_source_bbox_area = source_bbox_area * actual_source_scale_x * actual_source_scale_y

        # Principle: Full-body skeleton hand bbox typically captures only 15-20% of actual hand
        # Reason: DWPose detects wrist region accurately, but misses finger tips in full-body pose
        # 3D→2D Note: Hand extended forward (closer to camera) can appear >20%, up to 50%+
        # Solution: Use 10% baseline for keypoint bbox coverage (multiply by 10x for full-body)

        bbox_ratio = scaled_source_bbox_area / scaled_base_bbox_area if scaled_base_bbox_area > 0 else 1.0

        if bbox_ratio > FULLBODY_DETECTION_THRESHOLD:
            # Full-body skeleton detected - apply 10x correction
            estimated_actual_hand_bbox = scaled_base_bbox_area / BASELINE_HAND_COVERAGE

            # Calculate base pose actual hand size (손목→MCP 거리 기반)
            # CRITICAL: base_wrist_raw and base_mcp_raw are in CANVAS coordinates
            # Need to scale to actual image size FIRST
            base_wrist_img = [base_wrist_raw[0] * actual_base_scale_x, base_wrist_raw[1] * actual_base_scale_y]
            base_mcp_img = [base_mcp_raw[0] * actual_base_scale_x, base_mcp_raw[1] * actual_base_scale_y]

            # Now calculate distance in IMAGE pixel space
            base_hand_distance_img = math.sqrt((base_mcp_img[0] - base_wrist_img[0])**2 +
                                               (base_mcp_img[1] - base_wrist_img[1])**2)

            # 손 크기 추정: (손목→MCP 거리 × 6)² (손가락 길이 고려)
            base_hand_area_estimate = (base_hand_distance_img * 6) ** 2

            # Also calculate source hand distance for comparison
            source_wrist_img = [source_wrist_raw[0] * actual_source_scale_x, source_wrist_raw[1] * actual_source_scale_y]
            source_mcp_img = [source_mcp_raw[0] * actual_source_scale_x, source_mcp_raw[1] * actual_source_scale_y]
            source_hand_distance_img = math.sqrt((source_mcp_img[0] - source_wrist_img[0])**2 +
                                                 (source_mcp_img[1] - source_wrist_img[1])**2)

            # Calculate source hand area estimate (same method as base)
            source_hand_area_estimate = (source_hand_distance_img * 6) ** 2

            # Use base hand size as target with size multiplier
            adjusted_target_bbox = base_hand_area_estimate * HAND_SIZE_MULTIPLIER

            if enable_debug_logging:
                print(f"[{side}] Full-body (ratio={bbox_ratio:.1f}x): keypoint={scaled_base_bbox_area:.0f} × 10 = {estimated_actual_hand_bbox:.0f}")
                print(f"[{side}] Base hand distance: {base_hand_distance_img:.1f}px (canvas={math.sqrt((base_mcp_raw[0]-base_wrist_raw[0])**2+(base_mcp_raw[1]-base_wrist_raw[1])**2):.1f}, scale={actual_base_scale_x:.3f})")
                print(f"[{side}] Source hand distance: {source_hand_distance_img:.1f}px")
                print(f"[{side}] Base hand area: {base_hand_area_estimate:.0f} (from wrist→MCP distance)")
                print(f"[{side}] Source hand area: {source_hand_area_estimate:.0f} (from wrist→MCP distance)")
                print(f"[{side}] Source keypoint bbox: {scaled_source_bbox_area:.0f} (old method, for comparison)")
                print(f"[{side}] → Target bbox: {adjusted_target_bbox:.0f}")
        else:
            # Hand-only skeleton - also use distance-based calculation for BOTH base and source
            # Calculate base hand distance
            base_wrist_img = [base_wrist_raw[0] * actual_base_scale_x, base_wrist_raw[1] * actual_base_scale_y]
            base_mcp_img = [base_mcp_raw[0] * actual_base_scale_x, base_mcp_raw[1] * actual_base_scale_y]
            base_hand_distance_img = math.sqrt((base_mcp_img[0] - base_wrist_img[0])**2 +
                                               (base_mcp_img[1] - base_wrist_img[1])**2)
            base_hand_area_estimate = (base_hand_distance_img * 6) ** 2

            # Calculate source hand distance
            source_wrist_img = [source_wrist_raw[0] * actual_source_scale_x, source_wrist_raw[1] * actual_source_scale_y]
            source_mcp_img = [source_mcp_raw[0] * actual_source_scale_x, source_mcp_raw[1] * actual_source_scale_y]
            source_hand_distance_img = math.sqrt((source_mcp_img[0] - source_wrist_img[0])**2 +
                                                 (source_mcp_img[1] - source_wrist_img[1])**2)
            source_hand_area_estimate = (source_hand_distance_img * 6) ** 2

            # Use base hand size as target with size multiplier (same as full-body case)
            adjusted_target_bbox = base_hand_area_estimate * HAND_SIZE_MULTIPLIER

            if enable_debug_logging:
                print(f"[{side}] Hand-only (ratio={bbox_ratio:.1f}x): keypoint bbox={scaled_base_bbox_area:.0f}")
                print(f"[{side}] Base hand distance: {base_hand_distance_img:.1f}px (canvas={math.sqrt((base_mcp_raw[0]-base_wrist_raw[0])**2+(base_mcp_raw[1]-base_wrist_raw[1])**2):.1f}, scale={actual_base_scale_x:.3f})")
                print(f"[{side}] Source hand distance: {source_hand_distance_img:.1f}px")
                print(f"[{side}] Base hand area: {base_hand_area_estimate:.0f} (from wrist→MCP distance)")
                print(f"[{side}] Source hand area: {source_hand_area_estimate:.0f} (from wrist→MCP distance)")
                print(f"[{side}] → Target bbox: {adjusted_target_bbox:.0f}")

        # Calculate scale using distance-based source area
        size_scale_factor = math.sqrt(adjusted_target_bbox / source_hand_area_estimate) if source_hand_area_estimate > 0 else 1.0

        # Detect if hand needs horizontal flip based on arm direction (BEFORE rotation calculation)
        # Principle: Compare arm direction (elbow→wrist) with hand direction (wrist→MCP)
        # Why: Hand image may be captured from opposite direction compared to base pose arm
        # Method: If arm and hand point in opposite horizontal directions, flip is needed
        needs_flip = False
        body_kps = self._get_body_keypoints(base_keypoints)

        if body_kps and len(body_kps) > 7:  # Need at least 8 body keypoints (0-7)
            # Get elbow and wrist indices based on side
            # Left arm: shoulder(5) → elbow(6) → wrist(7)
            # Right arm: shoulder(2) → elbow(3) → wrist(4)
            if side == "left":
                elbow_idx = 6
                wrist_idx = 7
            else:
                elbow_idx = 3
                wrist_idx = 4

            elbow_kp = body_kps[elbow_idx]
            wrist_kp = body_kps[wrist_idx]

            # Only calculate flip if both elbow and wrist are valid
            if self.is_valid_keypoint(elbow_kp) and self.is_valid_keypoint(wrist_kp):
                # Arm direction: elbow → wrist
                arm_dx = wrist_kp[0] - elbow_kp[0]

                # Hand direction: wrist → MCP (from hand image)
                hand_dx = source_mcp_raw[0] - source_wrist_raw[0]

                # Flip if arm and hand point in opposite horizontal directions
                needs_flip = (arm_dx * hand_dx) < 0

                if enable_debug_logging:
                    print(f"[{side}] Arm direction (elbow→wrist): dx={arm_dx:.1f}")
                    print(f"[{side}] Hand direction (wrist→MCP): dx={hand_dx:.1f}")
                    print(f"[{side}] Flip needed: {needs_flip} (dot_product={arm_dx * hand_dx:.1f})")
            elif enable_debug_logging:
                print(f"[{side}] ⚠️ Elbow or wrist invalid, flip detection skipped")
        elif enable_debug_logging:
            print(f"[{side}] ⚠️ Body keypoints unavailable, flip detection skipped")

        # Calculate rotation angle AFTER flip detection
        # CRITICAL: If flipping, adjust source angle for horizontal mirror
        # Horizontal flip transforms angle: θ → (180° - θ)
        adjusted_source_angle = source_angle
        if needs_flip:
            adjusted_source_angle = 180.0 - source_angle
            if enable_debug_logging:
                print(f"[{side}] Angle adjustment for flip: {source_angle:.1f}° → {adjusted_source_angle:.1f}°")

        # CRITICAL: Always rotate to match base pose angle, ignore blend_strength for rotation
        # Rotation must match skeleton direction regardless of blend setting
        rotation_needed = base_angle - adjusted_source_angle

        # 5. Transform hand keypoints using normalized coordinates (resolution-independent)
        # NEW APPROACH V10: Canvas → Relative (0.0-1.0) → Transform → Image coordinates
        # This handles multi-resolution pipelines (1024px base + 512px hand) correctly

        # Get all hand keypoints
        hand_kps_list = self._get_hand_kps(hand_keypoints, side)
        if not hand_kps_list or len(hand_kps_list) < 21:
            return base_pil

        # Create coordinate transformers for resolution-independent processing
        source_transform = CoordinateTransform(source_canvas_size, (hand_pil.width, hand_pil.height))
        target_transform = CoordinateTransform(effective_base_canvas, (base_pil.width, base_pil.height))

        if enable_debug_logging:
            # Convert to relative coordinates for display
            source_wrist_display = source_transform.canvas_to_relative(source_wrist_raw)
            target_wrist_display = target_transform.canvas_to_relative(base_wrist_raw)

            print(f"[{side}] Source wrist: canvas={source_wrist_raw} → relative=({source_wrist_display[0]:.3f}, {source_wrist_display[1]:.3f})")
            print(f"[{side}] Target wrist: canvas={base_wrist_raw} → relative=({target_wrist_display[0]:.3f}, {target_wrist_display[1]:.3f})")
            print(f"[{side}] Flip: {needs_flip}, Scale: {size_scale_factor:.3f}, Rotation: {rotation_needed:.1f}°")

        # Removed verbose keypoint logging for cleaner output

        # Convert anchors to relative coordinates (resolution-independent)
        source_wrist_rel = source_transform.canvas_to_relative(source_wrist_raw)
        target_wrist_rel = target_transform.canvas_to_relative(base_wrist_raw)

        # Transform each keypoint in normalized space
        transformed_kps = []
        for kp in hand_kps_list:
            if not self.is_valid_keypoint(kp):
                transformed_kps.append(None)
                continue

            # Step 1: Canvas → Relative (0.0-1.0) space
            kp_rel = source_transform.canvas_to_relative(kp)

            # Step 2: Translate to origin (pivot at wrist, BEFORE flip/rotate/scale)
            rel_x = kp_rel[0] - source_wrist_rel[0]
            rel_y = kp_rel[1] - source_wrist_rel[1]

            # Step 3: Flip FIRST (in wrist-centered space)
            if needs_flip:
                # Mirror X coordinate around wrist (which is now at origin)
                old_rel_x = rel_x
                rel_x = -rel_x
                if enable_debug_logging and kp == hand_kps_list[5]:  # Log only for index MCP
                    print(f"[{side}] 🔄 FLIP APPLIED: Index MCP rel_x: {old_rel_x:.3f} → {rel_x:.3f}")

            # Step 4: Rotate (mathematically correct order: flip → rotate → scale)
            angle_rad = math.radians(rotation_needed)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            rotated_x = rel_x * cos_a - rel_y * sin_a
            rotated_y = rel_x * sin_a + rel_y * cos_a

            # Step 5: Scale AFTER rotation (preserves rotation geometry)
            scaled_x = rotated_x * size_scale_factor
            scaled_y = rotated_y * size_scale_factor

            # Step 6: Translate to target position in relative space
            final_rel_x = scaled_x + target_wrist_rel[0]
            final_rel_y = scaled_y + target_wrist_rel[1]

            # Step 7: Relative → Image pixel coordinates
            final_kp = target_transform.relative_to_image([final_rel_x, final_rel_y])
            transformed_kps.append(tuple(final_kp))

            # Debug: Log Index MCP final position
            if enable_debug_logging and kp == hand_kps_list[5]:
                print(f"[{side}] 📍 Index MCP final position: ({final_kp[0]:.1f}, {final_kp[1]:.1f})")

        # Removed verbose transformed keypoints logging

        # Draw transformed keypoints on base_pil
        draw = ImageDraw.Draw(base_pil)

        # Use bright colors for visibility
        colors = [
            (255, 100, 100),  # Bright red (thumb)
            (255, 200, 100),  # Orange (index)
            (255, 255, 100),  # Yellow (middle)
            (100, 255, 100),  # Green (ring)
            (100, 200, 255)   # Cyan (pinky)
        ]

        # Draw bone connections including wrist→palm base
        bone_connections = [
            # Wrist to palm base (connects wrist to each finger base)
            (0,1), (0,5), (0,9), (0,13), (0,17),
            # Thumb
            (1,2), (2,3), (3,4),
            # Index finger
            (5,6), (6,7), (7,8),
            # Middle finger
            (9,10), (10,11), (11,12),
            # Ring finger
            (13,14), (14,15), (15,16),
            # Pinky finger
            (17,18), (18,19), (19,20)
        ]

        lines_drawn = 0
        for i, (p1_idx, p2_idx) in enumerate(bone_connections):
            if p1_idx < len(transformed_kps) and p2_idx < len(transformed_kps):
                p1 = transformed_kps[p1_idx]
                p2 = transformed_kps[p2_idx]
                if p1 is not None and p2 is not None:
                    # Determine finger color based on bone connection
                    # Indices: 0-4 wrist connections, 5-7 thumb, 8-10 index, 11-13 middle, 14-16 ring, 17-19 pinky
                    if i < 5:
                        # Wrist to palm base - use corresponding finger color
                        finger_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}  # (0,1)→thumb, (0,5)→index, etc.
                        finger_idx = finger_map.get(i, 0)
                    else:
                        # Finger bones - group by 3
                        finger_idx = (i - 5) // 3

                    # Safety check: clamp to valid color range
                    finger_idx = min(finger_idx, len(colors) - 1)
                    color = colors[finger_idx]
                    draw.line([p1, p2], fill=color, width=line_thickness)
                    lines_drawn += 1

        # Draw keypoint circles (joints) - CRITICAL for Qwen recognition
        # DWPose format includes circles at each keypoint for better visibility
        keypoint_radius = 3  # 6px diameter circles for balanced visibility and precision
        keypoint_color = (0, 0, 255)  # Blue circles (standard DWPose color)
        circles_drawn = 0
        for kp in transformed_kps:
            if kp is not None:
                # Draw circle at keypoint
                x, y = kp
                # Draw filled circle
                draw.ellipse(
                    [(x - keypoint_radius, y - keypoint_radius),
                     (x + keypoint_radius, y + keypoint_radius)],
                    fill=keypoint_color,
                    outline=keypoint_color
                )
                circles_drawn += 1

        if enable_debug_logging:
            print(f"[{side}] Drew {lines_drawn}/{len(bone_connections)} bone connections")
            print(f"[{side}] Drew {circles_drawn}/21 keypoint circles")
            print(f"{'='*60}\n")

        return base_pil

    def _get_hand_kps(self, dwpose_data, side):
        if not dwpose_data: return None
        data = dwpose_data[0]

        # Handle nested `people` structure first (more common in newer versions)
        if "people" in data and data["people"]:
            person = data["people"][0]
            hand_key = f"hand_{side}_keypoints_2d"
            hand_kps_flat = person.get(hand_key)
            if not hand_kps_flat: return None
            # Convert flat list [x,y,c,x,y,c] to list of lists [[x,y],[x,y]]
            return [ [hand_kps_flat[i], hand_kps_flat[i+1]] for i in range(0, len(hand_kps_flat), 3) ]

        # Fallback to flat `keypoints` list
        all_keypoints = data.get("keypoints")
        if all_keypoints:
            # Corrected indices for flat keypoint array
            # Body: 0-53 (18 points), Face: 54-263 (70 points), Left hand: 264-326, Right hand: 327-389
            hand_start_idx = 264 if side == "left" else 327
            hand_end_idx = hand_start_idx + 63  # 21 points × 3 values
            if len(all_keypoints) < hand_end_idx: return None
            # Extract and convert to list of [x,y] pairs
            hand_flat = all_keypoints[hand_start_idx:hand_end_idx]
            return [ [hand_flat[i], hand_flat[i+1]] for i in range(0, len(hand_flat), 3) ]

        return None

    def _get_body_keypoints(self, dwpose_data):
        """
        Extract body keypoints from DWPose data.

        Body keypoint indices (DWPose format):
        - 0: nose, 1: neck
        - 2: right_shoulder, 3: right_elbow, 4: right_wrist
        - 5: left_shoulder, 6: left_elbow, 7: left_wrist
        - 8-17: hips, knees, ankles, eyes, ears

        Returns:
            List of body keypoints [[x,y], [x,y], ...] or None if unavailable
        """
        if not dwpose_data: return None
        data = dwpose_data[0]

        # Handle nested `people` structure first (more common in newer versions)
        if "people" in data and data["people"]:
            person = data["people"][0]
            body_kps_flat = person.get("pose_keypoints_2d")
            if not body_kps_flat: return None
            # Convert flat list [x,y,c,x,y,c] to list of lists [[x,y],[x,y]]
            return [ [body_kps_flat[i], body_kps_flat[i+1]] for i in range(0, len(body_kps_flat), 3) ]

        # Fallback to flat `keypoints` list
        all_keypoints = data.get("keypoints")
        if all_keypoints:
            # Body keypoints: indices 0-53 (18 points × 3 values)
            body_end_idx = 54  # 18 points × 3 values
            if len(all_keypoints) < body_end_idx: return None
            # Extract and convert to list of [x,y] pairs
            body_flat = all_keypoints[0:body_end_idx]
            return [ [body_flat[i], body_flat[i+1]] for i in range(0, len(body_flat), 3) ]

        return None

    def _get_hand_details(self, dwpose_data, side, enable_debug_logging, image_pil=None):
        try:
            if not dwpose_data: return None
            data = dwpose_data[0]
            canvas_size = (data.get("canvas_width", 512), data.get("canvas_height", 512))

            hand_kps_list = self._get_hand_kps(dwpose_data, side)
            if not hand_kps_list or len(hand_kps_list) < 21: return None

            wrist_kp = tuple(hand_kps_list[0])
            if not self.is_valid_keypoint(wrist_kp): return None

            # Find second anchor point with fallback chain
            # Priority: Middle(9) → Index(5) → Ring(13) → Pinky(17) → Thumb(1)
            # Reason: Middle finger is most stable and central
            mcp_kp = None
            for mcp_idx in [9, 5, 13, 17, 1]:
                candidate = tuple(hand_kps_list[mcp_idx])
                if self.is_valid_keypoint(candidate):
                    mcp_kp = candidate
                    if enable_debug_logging and mcp_idx != 9:
                        print(f"[{side}] Using fallback anchor: keypoint {mcp_idx} (middle=9 unavailable)")
                    break

            if mcp_kp is None:
                if enable_debug_logging:
                    print(f"[{side}] No valid MCP keypoint found in fallback chain")
                return None

            dx, dy = mcp_kp[0] - wrist_kp[0], mcp_kp[1] - wrist_kp[1]
            angle = np.degrees(np.arctan2(dy, dx))

            valid_points = [p for p in hand_kps_list if self.is_valid_keypoint(p)]
            if len(valid_points) < 3: return None

            # Calculate bbox from keypoints
            min_x = min(p[0] for p in valid_points)
            max_x = max(p[0] for p in valid_points)
            min_y = min(p[1] for p in valid_points)
            max_y = max(p[1] for p in valid_points)

            # If image provided, calculate actual skeleton pixel area
            if image_pil is not None:
                # Scale keypoint bbox to image coordinates
                scale_x = image_pil.width / canvas_size[0] if canvas_size[0] > 0 else 1.0
                scale_y = image_pil.height / canvas_size[1] if canvas_size[1] > 0 else 1.0

                img_min_x = int(min_x * scale_x)
                img_max_x = int(max_x * scale_x)
                img_min_y = int(min_y * scale_y)
                img_max_y = int(max_y * scale_y)

                # Clamp to image bounds
                img_min_x = max(0, img_min_x)
                img_max_x = min(image_pil.width, img_max_x)
                img_min_y = max(0, img_min_y)
                img_max_y = min(image_pil.height, img_max_y)

                # Crop hand region and count non-black pixels
                if img_max_x > img_min_x and img_max_y > img_min_y:
                    hand_region = image_pil.crop((img_min_x, img_min_y, img_max_x, img_max_y))
                    # Convert to grayscale and count bright pixels (skeleton lines)
                    gray = hand_region.convert('L')
                    pixels = np.array(gray)
                    # Count pixels brighter than threshold (skeleton is white/bright)
                    skeleton_pixels = np.sum(pixels > 30)

                    # Calculate actual bbox area from pixel count
                    # Instead of estimating from pixel count, use the actual cropped region dimensions
                    # This represents the area that contains all hand keypoints
                    region_width = img_max_x - img_min_x
                    region_height = img_max_y - img_min_y

                    # Use region dimensions directly as bbox
                    bbox_area = region_width * region_height
                    # Removed verbose skeleton pixel logging
                else:
                    bbox_area = (max_x - min_x) * (max_y - min_y)
            else:
                # No image provided, use keypoint bbox
                bbox_area = (max_x - min_x) * (max_y - min_y)

            if bbox_area < 1: return None

            # Return wrist, mcp, angle, bbox_area, canvas_size (added mcp for flip detection)
            return wrist_kp, mcp_kp, angle, bbox_area, canvas_size
        except Exception as e:
            if enable_debug_logging: print(f"[{side}] ERROR in _get_hand_details: {e}")
            traceback.print_exc()
            return None

    def _erase_hand_from_base(self, image, keypoints, side, expansion, enable_debug_logging):
        try:
            if not OPENCV_AVAILABLE:
                return

            hand_kps = self._get_hand_kps(keypoints, side)
            if not hand_kps or len(hand_kps) < 21:
                return

            canvas_size = (keypoints[0].get("canvas_width", image.width), keypoints[0].get("canvas_height", image.height))
            scale_x = image.width / canvas_size[0]
            scale_y = image.height / canvas_size[1]

            valid_points = [[p[0] * scale_x, p[1] * scale_y] for p in hand_kps if self.is_valid_keypoint(p)]
            if len(valid_points) < 3:
                return

            # Create convex hull and erase mask
            hull = cv2.convexHull(np.array(valid_points, dtype=np.float32))
            erase_mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(erase_mask).polygon([tuple(p[0]) for p in hull.tolist()], fill=255)

            # Expand mask if needed
            if expansion > 0:
                kernel = np.ones((expansion * 2 + 1, expansion * 2 + 1), np.uint8)
                erase_mask = Image.fromarray(cv2.dilate(np.array(erase_mask), kernel, iterations=1))

            image.paste((0, 0, 0), mask=erase_mask)
        except Exception as e:
            if enable_debug_logging: print(f"[{side}] Failed to erase hand: {e}")

    def is_valid_keypoint(self, kp):
        # DWPose marks invalid keypoints as either -1 or exactly (0.0, 0.0)
        # Use eps = 0.01 threshold to match DWPose validation (util.py Line 188)
        # This allows keypoints very close to edges while filtering exact (0.0, 0.0)
        eps = 0.01
        return kp is not None and len(kp) >= 2 and kp[0] > eps and kp[1] > eps and kp[0] != -1 and kp[1] != -1

    def tensor_to_pil(self, tensor):
        return Image.fromarray((tensor.cpu().numpy() * 255).astype(np.uint8))

    def pil_to_tensor(self, pil_img):
        return torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0).unsqueeze(0)


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "ComposeMultipleImages": ComposeMultipleImages,
    "SmartHandsReplace": SmartHandsReplace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComposeMultipleImages": "Compose Multiple Images (Grid)",
    "SmartHandsReplace": "Smart Hands Replace",
}
