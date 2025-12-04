# Research Methodology: Elderly Bed-Fall Detection

## 1. Domain Shift Analysis

The transition from the KTH Action Recognition dataset to Elderly Bed-Fall Detection introduces significant domain shifts that must be addressed:

*   **Subject Demographics**: KTH features young, healthy actors performing exaggerated, clean actions. The target domain involves elderly patients with potentially limited mobility, frailty, and different body shapes.
*   **Environment**: KTH is filmed in controlled outdoor/indoor settings with stable lighting and clean backgrounds. Bed-fall videos are likely in hospital rooms or bedrooms, with clutter, occlusion (bed rails, blankets), and variable lighting (night mode).
*   **Motion Dynamics**: KTH actions (boxing, running) are high-energy and repetitive. A fall is a singular, non-repetitive, and often slow-onset event followed by rapid descent, or a "crumple" rather than a clean ballistic trajectory.
*   **Viewpoint**: KTH has consistent camera angles. Bed-fall videos may have overhead CCTV angles or side views from bedside monitors.

**Why Skeleton-based Features?**
Using OpenPose-extracted skeletons mitigates visual domain shifts (lighting, background, clothing) by abstracting the input to pure geometric pose data. The model learns *motion patterns* of joints rather than pixel-level features, making it more robust to environmental differences.

## 2. Transfer Learning Plan

To leverage the 3D CNN trained on KTH:

1.  **Pretraining**: Use the KTH-trained model as a feature extractor. The model has learned to recognize temporal relationships between joint movements (e.g., leg lifting, arm swinging).
2.  **Architecture Adaptation**:
    *   **Freeze Layers**: Freeze the convolutional blocks (Conv1, Conv2) to retain low-level motion features.
    *   **Fine-tuning**: Unfreeze the final convolutional block (Conv3) and the Global Average Pooling layer to adapt to the specific dynamics of a fall.
    *   **Head Replacement**: Replace the 6-class classification head (`Linear(128, 6)`) with a binary classification head (`Linear(128, 2)`) for "Fall" vs. "Non-Fall".
3.  **Training Strategy**:
    *   **Positive Class (Fall)**: The 10 elderly bed-fall videos.
    *   **Negative Class (Non-Fall)**: A subset of KTH videos (walking, standing) and potentially "normal activity" segments extracted from the start of the bed-fall videos (before the fall occurs).

## 3. Data Scarcity Strategy

With only 10 fall videos, overfitting is a major risk. We will employ aggressive augmentation:

*   **Temporal Augmentation**:
    *   **Time Warping**: Speed up and slow down the fall sequence (0.8x to 1.2x speed) to simulate different fall velocities.
    *   **Window Slicing**: Extract multiple overlapping windows around the fall event. If a fall happens at frame 50, take windows [20-52], [25-57], [30-62] as positive samples.
*   **Spatial/Skeleton Augmentation**:
    *   **Rotation**: Rotate the entire skeleton by ±15 degrees to simulate different camera angles.
    *   **Scaling**: Randomly scale the skeleton size to simulate different distances from the camera.
    *   **Noise Injection**: Add small Gaussian noise to joint coordinates to simulate sensor jitter.
    *   **Mirroring**: Flip skeletons horizontally (left/right swap) to double the dataset.
*   **Synthetic Falls (Concept)**:
    *   Interpolate between an upright pose and a prone pose to generate synthetic fall trajectories, though this is complex to make realistic.

## 4. Evaluation Method

Due to the small sample size, a standard train/test split is unreliable.

*   **Leave-One-Video-Out Cross-Validation (LOOCV)**:
    *   Train on 9 fall videos (+ negative samples), test on the 1 remaining fall video.
    *   Repeat 10 times, once for each video.
    *   Report average metrics across all 10 folds.
*   **Metrics**:
    *   **Recall (Sensitivity)**: Crucial for safety. We must detect every fall. Target > 95%.
    *   **Precision**: Important to reduce false alarms for nurses.
    *   **F1-Score**: Harmonic mean of Precision and Recall.
    *   **Latency**: Time from fall onset to detection trigger.
*   **Temporal Smoothing**:
    *   During inference, run the model on a sliding window.
    *   Trigger an alarm only if the probability of "Fall" > Threshold for $N$ consecutive frames (e.g., 5 frames) to avoid flickering false positives.

## 5. Model Deployment Considerations

*   **Privacy**: Skeleton extraction happens at the edge (on the device). No raw video needs to be stored or transmitted, preserving patient privacy.
*   **Real-time Constraints**: The 3D CNN is lightweight. OpenPose is the bottleneck. For deployment, switch to lighter pose estimators like **MediaPipe Pose** or **MoveNet** which run faster on CPU/mobile devices.
*   **Threshold Tuning**: Implement a sensitivity dial. In high-risk wards, lower the detection threshold (higher Recall, more False Alarms). In general wards, raise it.
