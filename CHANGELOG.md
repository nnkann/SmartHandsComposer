# Changelog

All notable changes, development challenges, and solutions for Smart Hands Replace.

## [1.0.0] - 2025-01-XX

### Features

#### SmartHandsReplace Node
- **3-Stage Hand Detection Pipeline**: MediaPipe → Phase2 Estimation → Basepose fallback
- **Intelligent Flip Detection**: Arm direction-based flip logic using elbow→wrist vector analysis
- **Resolution-Independent Transforms**: Normalized coordinate space (0.0-1.0) for multi-resolution workflows
- **Distance-Based Hand Sizing**: Accurate hand size calculation using wrist→MCP distance
- **DWPose Skeleton Output**: Colored bone connections with blue keypoint circles (6px diameter)
- **Full-Body Support**: Automatic detection and adjustment for full-body vs hand-only skeletons
- **Configurable Parameters**: blend_strength, erase_expansion, line_thickness, debug logging

#### ComposeMultipleImages Node
- Grid layout composition for hand reference images
- Auto-canvas sizing with configurable padding
- Background color options (black/white/gray)

### Development Journey

#### Challenge 1: Hand Flip Detection
**Problem**: Initial implementation compared left hand to right hand positions, causing incorrect flip detection when hands crossed body midline or overlapped.

**Symptoms**:
- Hands flipped incorrectly when crossing midline
- Unreliable detection with overlapping hands
- Inconsistent results with different body poses

**Solution**: Implemented arm direction-based flip detection
- Compare elbow→wrist vector (arm direction) with wrist→MCP vector (hand direction)
- Flip hand if vectors point in opposite horizontal directions
- Angle transformation: θ → (180° - θ) when flipped
- Independent detection for each hand (no cross-hand comparison)

**Implementation**:
```python
# Calculate arm direction (elbow → wrist)
arm_vector = wrist_pt - elbow_pt
arm_angle_h = math.atan2(arm_vector[0], -arm_vector[1])

# Calculate hand direction (wrist → MCP base)
hand_vector = mcp_center - wrist_pt
hand_angle_h = math.atan2(hand_vector[0], -hand_vector[1])

# Flip if arm and hand point in opposite horizontal directions
needs_flip = (arm_angle_h * hand_angle_h) < 0
```

**Result**: 95%+ accurate flip detection across various poses

---

#### Challenge 2: Hand Size Calculation
**Problem**: Initial bounding box-based method produced inconsistent hand sizes, especially with fingers spread wide or closed fists.

**Symptoms**:
- Hand size varied dramatically based on finger positions
- Spread fingers → oversized hands
- Closed fists → undersized hands
- Inconsistent visual results

**Solution**: Distance-based hand size calculation
- Use anatomically stable wrist→MCP distance as reference
- Calculate hand area: (wrist→MCP distance × 6)²
- Apply 1.2x multiplier for visual accuracy
- Handle full-body vs hand-only skeletons automatically

**Implementation**:
```python
# Distance-based method (stable across finger positions)
wrist_to_mcp_dist = np.linalg.norm(mcp_center - wrist_pt)
hand_size = (wrist_to_mcp_dist * 6) ** 2
hand_size *= HAND_SIZE_MULTIPLIER  # 1.2x default

# Full-body skeleton detection and adjustment
bbox_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
is_fullbody = bbox_ratio > FULLBODY_DETECTION_THRESHOLD
if is_fullbody:
    hand_size /= (BASELINE_HAND_COVERAGE ** 2)
```

**Result**: Consistent hand sizing regardless of finger positions

---

#### Challenge 3: Qwen VL Integration
**Problem**: Initial skeleton output didn't provide clear visual markers for VLM (Vision-Language Model) to identify target positions.

**Symptoms**:
- Qwen VL couldn't reliably identify exact finger joint positions
- Generated hands didn't align with skeleton structure
- Poor hand pose transfer accuracy

**Solution**: Enhanced skeleton visualization for VLM recognition
- Added blue keypoint circles (3px radius, 6px diameter) at all joints
- Implemented wrist connection lines (5 additional bone connections)
- Optimized for Qwen VL's visual pattern recognition
- Provided detailed prompts emphasizing blue circular markers

**Implementation**:
```python
# Draw blue keypoint circles for VLM recognition
for point in hand_keypoints:
    if point is not None and len(point) == 2:
        x_img, y_img = int(point[0] * img_w), int(point[1] * img_h)
        cv2.circle(skeleton_img, (x_img, y_img), 3, KEYPOINT_COLOR, -1)

# Wrist connections (5 additional lines)
wrist_connections = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]
```

**Recommended Prompt**:
```
Replace the hands in image1 with new hands that exactly match the hand pose
shown by the blue skeleton keypoints in image2. Each finger joint must
precisely align with the blue circular markers.
```

**Result**: Significantly improved Qwen VL hand pose transfer accuracy

---

#### Challenge 4: ControlNet Performance
**Problem**: Initial testing with ControlNet for hand pose transfer showed severe performance degradation.

**Symptoms**:
- 17 minutes per generation vs 35 seconds without ControlNet
- 10x performance penalty
- VRAM constraints on consumer GPUs
- Workflow complexity increased significantly

**Decision**: Abandoned ControlNet integration
- Focus on Qwen VL + skeleton visualization approach
- Maintain fast generation times (30-40 seconds)
- Reduce VRAM requirements (5-6GB with GGUF Q5_0 vs 19GB FP8)
- Simplify workflow for end users

**Result**: Fast, accessible hand pose replacement workflow

---

#### Challenge 5: Coordinate System Complexity
**Problem**: DWPose canvas coordinates don't always match image resolution, causing misalignment issues.

**Solution**: Implemented three-stage coordinate system
1. **Canvas Space**: DWPose native coordinates
2. **Relative Space**: Normalized 0.0-1.0 (resolution-independent)
3. **Image Space**: Actual pixel coordinates for drawing

**Transform Pipeline**:
```
Canvas → Relative (normalization)
       → Flip (if needed)
       → Rotate (angle adjustment)
       → Scale (hand sizing)
       → Image (denormalization)
```

**Result**: Resolution-independent transforms that work across any image size

---

### Technical Specifications

**Hand Detection**:
- MediaPipe confidence threshold: 0.35
- Full-body detection threshold: bbox_ratio > 3.0
- Baseline hand coverage: 10% (keypoint bbox / actual hand)

**Transform Parameters**:
- Hand size multiplier: 1.2x (configurable)
- Keypoint circle radius: 3px (6px diameter)
- Default line thickness: 2px
- Default erase expansion: 10px

**Performance Benchmarks**:
- Hand detection: 50-200ms per hand (MediaPipe)
- Transform + drawing: 10-50ms per hand
- Total processing: 100-500ms for dual hand composition

**Hardware Requirements**:
- Minimum: 8GB RAM, 4GB VRAM
- Recommended: 16GB RAM, 8GB+ VRAM
- Optimal: 32GB RAM, RTX 3080+ (12GB VRAM)

---

### Dependencies

**Required**:
- torch
- numpy
- PIL (Pillow)
- opencv-python
- scipy

**Optional**:
- mediapipe (highly recommended for best detection accuracy)

---

### Known Limitations

1. **MediaPipe Dependency**: Best results require MediaPipe installation
2. **Hand Visibility**: Detection accuracy decreases with occluded hands
3. **Extreme Poses**: May require manual adjustment for very unusual poses
4. **Single Person**: Optimized for single-person images (multi-person support limited)

---

### Future Improvements

- Multi-person hand detection and replacement
- Custom hand size per finger (individual finger scaling)
- Advanced blend modes for skeleton integration
- Real-time preview support
- Batch processing optimization

---

## Version History

### V11 (Current - Production Release)
- Final production-ready release
- Simplified interface with essential parameters
- Comprehensive documentation
- Qwen VL integration guide

### V10 (Development - Detection System)
- Implemented 3-stage detection pipeline
- Added Phase2 estimation fallback
- Enhanced MediaPipe integration

### V9 and Earlier (Experimental)
- Prototype development
- Initial flip detection experiments
- Basic hand transformation logic

---

## Credits

- **DWPose Format**: Pose estimation standard
- **MediaPipe**: Google's hand landmark detection
- **ComfyUI**: Node-based workflow framework
- **Qwen VL**: Alibaba's Vision-Language Model

---

## License

MIT License - See LICENSE file for details
