"""
Hand anatomical structure and proportions for keypoint estimation.

Based on standard hand anatomy:
- 21 keypoints (same as MediaPipe/DWPose)
- Anatomically accurate bone length ratios
- Standard finger spread angles
"""

import numpy as np
import math

# Hand keypoint indices (MediaPipe/DWPose standard)
KEYPOINT_NAMES = [
    "wrist",           # 0
    "thumb_cmc",       # 1 (Carpometacarpal)
    "thumb_mcp",       # 2 (Metacarpophalangeal)
    "thumb_ip",        # 3 (Interphalangeal)
    "thumb_tip",       # 4
    "index_mcp",       # 5
    "index_pip",       # 6 (Proximal interphalangeal)
    "index_dip",       # 7 (Distal interphalangeal)
    "index_tip",       # 8
    "middle_mcp",      # 9
    "middle_pip",      # 10
    "middle_dip",      # 11
    "middle_tip",      # 12
    "ring_mcp",        # 13
    "ring_pip",        # 14
    "ring_dip",        # 15
    "ring_tip",        # 16
    "pinky_mcp",       # 17
    "pinky_pip",       # 18
    "pinky_dip",       # 19
    "pinky_tip",       # 20
]

# Finger structure: [finger_name, [kp_indices]]
FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}

# MCP (Metacarpophalangeal) joint indices
MCP_INDICES = [5, 9, 13, 17]  # index, middle, ring, pinky
THUMB_BASE = 1

# Standard hand proportions (relative to total hand length)
# Based on Leonardo da Vinci's anatomical studies
HAND_PROPORTIONS = {
    # Hand length from wrist to middle fingertip = 1.0
    "palm_length": 0.55,          # Wrist to MCP knuckles

    # Finger lengths (relative to hand length)
    "thumb_length": 0.40,
    "index_length": 0.45,
    "middle_length": 0.50,
    "ring_length": 0.47,
    "pinky_length": 0.38,

    # Phalanx ratios (proximal:middle:distal)
    "thumb_phalanges": [0.50, 0.30, 0.20],   # Thumb has 2 phalanges (CMC acts as base)
    "index_phalanges": [0.40, 0.30, 0.30],
    "middle_phalanges": [0.40, 0.30, 0.30],
    "ring_phalanges": [0.40, 0.30, 0.30],
    "pinky_phalanges": [0.40, 0.30, 0.30],
}

# Finger spread angles (degrees from middle finger axis)
# For relaxed open hand pose
FINGER_SPREAD_ANGLES = {
    "thumb": -50,    # Thumb spreads outward significantly
    "index": -10,
    "middle": 0,     # Reference axis
    "ring": 10,
    "pinky": 20
}

# MCP knuckle positions (relative to palm width)
# In relaxed pose, MCPs form an arc
MCP_POSITIONS = {
    "index": -0.3,   # Left of center
    "middle": 0.0,   # Center
    "ring": 0.3,     # Right of center
    "pinky": 0.55    # Far right
}


class HandAnatomy:
    """
    Hand anatomical model for keypoint estimation.
    """

    @staticmethod
    def estimate_hand_size(wrist, mcp):
        """
        Estimate hand size from wrist and MCP positions.

        Args:
            wrist: [x, y] wrist position
            mcp: [x, y] MCP position (any finger)

        Returns:
            float: Estimated hand length (wrist to middle fingertip)
        """
        if wrist is None or mcp is None:
            return None

        wrist = np.array(wrist)
        mcp = np.array(mcp)

        # Distance from wrist to MCP ≈ palm_length
        palm_length = np.linalg.norm(mcp - wrist)

        # Total hand length = palm_length / 0.55
        hand_length = palm_length / HAND_PROPORTIONS["palm_length"]

        return hand_length

    @staticmethod
    def estimate_palm_orientation(wrist, middle_mcp, index_mcp=None):
        """
        Estimate palm orientation (angle and normal vector).

        Args:
            wrist: [x, y] wrist position
            middle_mcp: [x, y] middle finger MCP
            index_mcp: [x, y] index finger MCP (optional, for better accuracy)

        Returns:
            dict: {
                "palm_vector": np.array,  # Wrist to middle MCP
                "palm_angle": float,      # Angle in degrees
                "palm_width_vector": np.array  # Perpendicular to palm
            }
        """
        wrist = np.array(wrist)
        middle_mcp = np.array(middle_mcp)

        # Primary palm direction: wrist to middle MCP
        palm_vector = middle_mcp - wrist
        palm_angle = math.degrees(math.atan2(palm_vector[1], palm_vector[0]))

        # Palm width direction (perpendicular)
        palm_width_vector = np.array([-palm_vector[1], palm_vector[0]])
        palm_width_vector = palm_width_vector / np.linalg.norm(palm_width_vector)

        return {
            "palm_vector": palm_vector,
            "palm_angle": palm_angle,
            "palm_width_vector": palm_width_vector
        }

    @staticmethod
    def estimate_finger_direction(wrist, finger_mcp, palm_orientation):
        """
        Estimate finger direction vector from wrist and MCP.

        Args:
            wrist: [x, y] wrist position
            finger_mcp: [x, y] finger MCP position
            palm_orientation: dict from estimate_palm_orientation()

        Returns:
            np.array: Normalized finger direction vector
        """
        wrist = np.array(wrist)
        finger_mcp = np.array(finger_mcp)

        # Finger direction: from wrist through MCP
        finger_vector = finger_mcp - wrist
        finger_vector = finger_vector / np.linalg.norm(finger_vector)

        return finger_vector

    @staticmethod
    def calculate_phalanx_positions(mcp, finger_direction, finger_name, hand_length):
        """
        Calculate PIP, DIP, and TIP positions from MCP.

        Args:
            mcp: [x, y] MCP position
            finger_direction: np.array normalized direction vector
            finger_name: str ("thumb", "index", "middle", "ring", "pinky")
            hand_length: float estimated hand length

        Returns:
            list: [[x,y], [x,y], [x,y]] for PIP, DIP, TIP
        """
        mcp = np.array(mcp)

        # Get finger length and phalanx ratios
        finger_length = hand_length * HAND_PROPORTIONS[f"{finger_name}_length"]
        phalanx_ratios = HAND_PROPORTIONS[f"{finger_name}_phalanges"]

        # Calculate joint positions
        # For 4-joint fingers: MCP -> PIP -> DIP -> TIP
        # Each phalanx length is proportional to finger length

        positions = []
        current_pos = mcp.copy()

        for ratio in phalanx_ratios:
            phalanx_length = finger_length * ratio
            current_pos = current_pos + finger_direction * phalanx_length
            positions.append(current_pos.tolist())

        return positions

    @staticmethod
    def estimate_mcp_position(wrist, palm_orientation, finger_name, hand_length):
        """
        Estimate MCP position from wrist and palm orientation.
        Used when MCP is not detected.

        Args:
            wrist: [x, y] wrist position
            palm_orientation: dict from estimate_palm_orientation()
            finger_name: str finger name
            hand_length: float estimated hand length

        Returns:
            [x, y]: Estimated MCP position
        """
        wrist = np.array(wrist)
        palm_vector = palm_orientation["palm_vector"]
        palm_width_vector = palm_orientation["palm_width_vector"]

        # Palm length
        palm_length = hand_length * HAND_PROPORTIONS["palm_length"]

        # MCP is at the end of palm
        palm_vector_norm = palm_vector / np.linalg.norm(palm_vector)
        mcp_base = wrist + palm_vector_norm * palm_length

        # Offset perpendicular based on finger position
        if finger_name in MCP_POSITIONS:
            offset_ratio = MCP_POSITIONS[finger_name]
            palm_width = hand_length * 0.4  # Palm width ≈ 40% of hand length
            offset = palm_width_vector * (offset_ratio * palm_width)
            mcp = mcp_base + offset
        else:
            mcp = mcp_base

        return mcp.tolist()


class HandEstimator:
    """
    Phase 2: Vector-based hand keypoint estimation.
    Estimates missing keypoints from partial detections using anatomical proportions.
    """

    def __init__(self):
        self.anatomy = HandAnatomy()

    def estimate_missing_keypoints(self, detected_keypoints, basepose_keypoints, side="left"):
        """
        Complete hand keypoint detection and estimation pipeline.

        Args:
            detected_keypoints: list[21] of [x,y] or None - DWPose/MediaPipe detection results
            basepose_keypoints: list[21] of [x,y] - Basepose original hand skeleton
            side: str "left" or "right"

        Returns:
            dict: {
                "keypoints": list[21] of [x,y],
                "confidence": list[21] of float,
                "method": str,
                "message": str,
                "status": str  # "success" | "warning" | "error"
            }
        """
        # Step 1: Analyze detected keypoints
        analysis = self._analyze_detection(detected_keypoints)

        # Case E: No keypoints detected at all
        if analysis["wrist"] is None:
            return {
                "keypoints": basepose_keypoints,
                "confidence": [0.0] * 21,
                "method": "basepose_fallback",
                "message": "❌ No hand keypoints detected! Using base pose hand. STRONGLY recommend replacing image.",
                "status": "error"
            }

        # Case D: Wrist only, no MCPs
        if analysis["detected_mcp_count"] == 0:
            return {
                "keypoints": basepose_keypoints,
                "confidence": [0.0] * 21,
                "method": "basepose_fallback",
                "message": "⚠️ Insufficient detection (wrist only). Using base pose hand. Consider replacing image.",
                "status": "warning"
            }

        # Case B/C: Wrist + 1+ MCPs → Perform Phase 2 estimation
        try:
            # Step 2: Estimate hand size and orientation
            hand_info = self._estimate_hand_info(analysis)

            # Step 3: Estimate missing MCPs
            all_mcps = self._estimate_mcps(analysis, hand_info)

            # Step 4: Estimate finger joints
            full_keypoints = self._estimate_finger_joints(analysis, all_mcps, hand_info)

            # Step 5: Calculate confidence scores
            confidence_scores = self._calculate_confidence(analysis, full_keypoints)

            # Step 6: Generate status message
            message, status = self._generate_status_message(analysis, confidence_scores)

            return {
                "keypoints": full_keypoints,
                "confidence": confidence_scores,
                "method": "vector_based_estimation",
                "message": message,
                "status": status
            }

        except Exception as e:
            # Estimation failed, fallback to basepose
            return {
                "keypoints": basepose_keypoints,
                "confidence": [0.0] * 21,
                "method": "basepose_fallback",
                "message": f"⚠️ Estimation failed: {str(e)}. Using base pose hand.",
                "status": "warning"
            }

    def _analyze_detection(self, detected_keypoints):
        """
        Analyze which keypoints were detected.

        Returns:
            dict: {
                "wrist": [x,y] or None,
                "detected_mcps": dict {5: [x,y], 9: [x,y], ...},
                "detected_mcp_count": int,
                "detected_count": int,
                "all_detected": dict {kp_idx: [x,y]}
            }
        """
        analysis = {
            "wrist": None,
            "detected_mcps": {},
            "detected_mcp_count": 0,
            "detected_count": 0,
            "all_detected": {}
        }

        if detected_keypoints is None or len(detected_keypoints) != 21:
            return analysis

        for idx, kp in enumerate(detected_keypoints):
            if kp is not None and len(kp) == 2:
                x, y = kp
                # Check if valid (not [0.0, 0.0])
                if abs(x) > 0.01 or abs(y) > 0.01:
                    analysis["all_detected"][idx] = kp
                    analysis["detected_count"] += 1

                    if idx == 0:
                        analysis["wrist"] = kp
                    elif idx in MCP_INDICES:
                        analysis["detected_mcps"][idx] = kp
                        analysis["detected_mcp_count"] += 1

        return analysis

    def _estimate_hand_info(self, analysis):
        """
        Estimate hand size and palm orientation from detected keypoints.

        Returns:
            dict: {
                "hand_length": float,
                "palm_orientation": dict,
                "reference_mcp": [x,y]
            }
        """
        wrist = np.array(analysis["wrist"])

        # Use the first detected MCP as reference
        if analysis["detected_mcp_count"] > 0:
            reference_mcp_idx = list(analysis["detected_mcps"].keys())[0]
            reference_mcp = np.array(analysis["detected_mcps"][reference_mcp_idx])
        else:
            raise ValueError("No MCP detected")

        # Estimate hand size
        hand_length = self.anatomy.estimate_hand_size(wrist, reference_mcp)

        # Estimate palm orientation
        # If middle MCP (kp[9]) detected, use it; otherwise use any detected MCP
        if 9 in analysis["detected_mcps"]:
            middle_mcp = np.array(analysis["detected_mcps"][9])
        else:
            middle_mcp = reference_mcp

        palm_orientation = self.anatomy.estimate_palm_orientation(
            wrist, middle_mcp
        )

        return {
            "hand_length": hand_length,
            "palm_orientation": palm_orientation,
            "reference_mcp": reference_mcp.tolist()
        }

    def _estimate_mcps(self, analysis, hand_info):
        """
        Estimate missing MCP positions.

        Returns:
            dict: {5: [x,y], 9: [x,y], 13: [x,y], 17: [x,y]}
        """
        all_mcps = {}

        # Add detected MCPs
        all_mcps.update(analysis["detected_mcps"])

        # Estimate missing MCPs
        finger_names = {5: "index", 9: "middle", 13: "ring", 17: "pinky"}

        for mcp_idx in MCP_INDICES:
            if mcp_idx not in all_mcps:
                finger_name = finger_names[mcp_idx]
                estimated_mcp = self.anatomy.estimate_mcp_position(
                    analysis["wrist"],
                    hand_info["palm_orientation"],
                    finger_name,
                    hand_info["hand_length"]
                )
                all_mcps[mcp_idx] = estimated_mcp

        return all_mcps

    def _estimate_finger_joints(self, analysis, all_mcps, hand_info):
        """
        Estimate all finger joints (PIP, DIP, TIP) from MCPs.

        Returns:
            list[21]: Complete keypoints array
        """
        full_keypoints = [None] * 21

        # Set wrist
        full_keypoints[0] = analysis["wrist"]

        # Set MCPs
        for mcp_idx, mcp_pos in all_mcps.items():
            full_keypoints[mcp_idx] = mcp_pos

        wrist = np.array(analysis["wrist"])

        # Estimate finger joints for each finger
        finger_data = {
            "index": {"mcp": 5, "joints": [6, 7, 8]},
            "middle": {"mcp": 9, "joints": [10, 11, 12]},
            "ring": {"mcp": 13, "joints": [14, 15, 16]},
            "pinky": {"mcp": 17, "joints": [18, 19, 20]}
        }

        for finger_name, data in finger_data.items():
            mcp_idx = data["mcp"]
            mcp = np.array(all_mcps[mcp_idx])

            # Calculate finger direction
            finger_direction = self.anatomy.estimate_finger_direction(
                wrist, mcp, hand_info["palm_orientation"]
            )

            # Calculate phalanx positions
            phalanx_positions = self.anatomy.calculate_phalanx_positions(
                mcp, finger_direction, finger_name, hand_info["hand_length"]
            )

            # Assign to keypoints
            for i, joint_idx in enumerate(data["joints"]):
                full_keypoints[joint_idx] = phalanx_positions[i]

        # Estimate thumb (special case)
        full_keypoints = self._estimate_thumb(full_keypoints, analysis, hand_info)

        return full_keypoints

    def _estimate_thumb(self, keypoints, analysis, hand_info):
        """
        Estimate thumb keypoints (kp[1-4]).
        Thumb is special because it spreads differently.
        """
        wrist = np.array(analysis["wrist"])

        # Check if thumb is already detected
        thumb_detected = all(
            idx in analysis["all_detected"] for idx in [1, 2, 3, 4]
        )

        if thumb_detected:
            for idx in [1, 2, 3, 4]:
                keypoints[idx] = analysis["all_detected"][idx]
            return keypoints

        # Estimate thumb base (CMC - kp[1])
        palm_vector = hand_info["palm_orientation"]["palm_vector"]
        palm_width_vector = hand_info["palm_orientation"]["palm_width_vector"]
        palm_length = hand_info["hand_length"] * HAND_PROPORTIONS["palm_length"]

        # Thumb CMC is offset from wrist
        palm_vector_norm = palm_vector / np.linalg.norm(palm_vector)
        thumb_cmc = wrist + palm_vector_norm * (palm_length * 0.3) + palm_width_vector * (-0.2 * hand_info["hand_length"])
        keypoints[1] = thumb_cmc.tolist()

        # Thumb direction (spreads outward)
        thumb_angle = hand_info["palm_orientation"]["palm_angle"] + FINGER_SPREAD_ANGLES["thumb"]
        thumb_direction = np.array([
            math.cos(math.radians(thumb_angle)),
            math.sin(math.radians(thumb_angle))
        ])

        # Calculate thumb joints
        thumb_positions = self.anatomy.calculate_phalanx_positions(
            thumb_cmc, thumb_direction, "thumb", hand_info["hand_length"]
        )

        keypoints[2] = thumb_positions[0]  # MCP
        keypoints[3] = thumb_positions[1]  # IP
        keypoints[4] = thumb_positions[2]  # TIP

        return keypoints

    def _calculate_confidence(self, analysis, keypoints):
        """
        Calculate confidence score for each keypoint.

        Returns:
            list[21]: Confidence scores
        """
        confidence = [0.0] * 21

        # Detected keypoints = 1.0
        for idx in analysis["all_detected"]:
            confidence[idx] = 1.0

        # Estimated MCPs
        if analysis["detected_mcp_count"] > 0:
            base_mcp_confidence = 0.7 if analysis["detected_mcp_count"] > 1 else 0.6
            for mcp_idx in MCP_INDICES:
                if mcp_idx not in analysis["all_detected"]:
                    confidence[mcp_idx] = base_mcp_confidence

        # Estimated finger joints
        joint_confidence = 0.6 if analysis["detected_mcp_count"] > 1 else 0.5
        for idx in range(1, 21):
            if confidence[idx] == 0.0:  # Not yet assigned
                confidence[idx] = joint_confidence

        return confidence

    def _generate_status_message(self, analysis, confidence_scores):
        """
        Generate user-facing status message.

        Returns:
            (message: str, status: str)
        """
        detected_count = analysis["detected_count"]
        estimated_count = 21 - detected_count
        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        if avg_confidence >= 0.7:
            return (
                f"✅ Hand estimated from DWPose (wrist + {analysis['detected_mcp_count']} MCPs detected, {estimated_count} keypoints estimated)",
                "success"
            )
        else:
            return (
                f"⚠️ Hand estimated with low confidence (wrist + {analysis['detected_mcp_count']} MCP detected, {estimated_count} keypoints estimated)",
                "warning"
            )
