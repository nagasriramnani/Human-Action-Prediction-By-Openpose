import torch
import numpy as np
import cv2
import os
import shutil
import glob
from model import ActionResNet3D
from video_to_keypoints import run_openpose, json_to_numpy, normalize_skeleton

# Configuration
MODEL_PATH = "best_model.pth" # Path to trained model
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu") # Force CPU for debugging crash
CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

def interpolate(data):
    """
    Fill zero values (missing keypoints) using linear interpolation.
    data: (T, J, C)
    """
    T, J, C = data.shape
    for j in range(J):
        for c in range(C):
            # Find valid (non-zero) indices
            valid_indices = np.where(data[:, j, c] != 0)[0]
            
            # If no valid points, skip (can't interpolate)
            if len(valid_indices) == 0:
                continue
                
            # If some missing, interpolate
            if len(valid_indices) < T:
                data[:, j, c] = np.interp(
                    np.arange(T), 
                    valid_indices, 
                    data[valid_indices, j, c]
                )
    return data

def render_video(video_path, keypoints, output_path):
    """
    Render skeleton on video and save as MP4.
    keypoints: (T, 25, 2)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # MP4 codec (avc1 is widely supported, mp4v is backup)
    # WebM codec (vp80) is supported by browsers and doesn't require OpenH264
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # BODY_25 Pairs
    pairs = [
        (1,8), (1,2), (1,5), (2,3), (3,4), (5,6), (6,7), 
        (8,9), (9,10), (10,11), (8,12), (12,13), (13,14),
        (1,0), (0,15), (15,17), (0,16), (16,18), 
        (14,19), (19,20), (14,21), (11,22), (22,23), (11,24)
    ]
    color = (0, 255, 0) # Green
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx < len(keypoints):
            kp = keypoints[frame_idx]
            # Draw connections
            for p1, p2 in pairs:
                if p1 < 25 and p2 < 25:
                    x1, y1 = int(kp[p1, 0]), int(kp[p1, 1])
                    x2, y2 = int(kp[p2, 0]), int(kp[p2, 1])
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw points
            for i in range(25):
                x, y = int(kp[i, 0]), int(kp[i, 1])
                if x > 0 and y > 0:
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()

class ActionRecognizer:
    def __init__(self, model_path=None):
        self.device = DEVICE
        self.model = ActionResNet3D(num_classes=len(CLASSES)).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: No model found, using random weights for testing.")
            
        self.model.eval()

    def predict_video(self, video_path):
        """
        Full pipeline: Video -> OpenPose -> Skeleton -> Model -> Probabilities
        """
        # 1. Create temp dir for OpenPose output
        # CRITICAL FIX: Use absolute path so OpenPose can find it from any working directory
        temp_dir = os.path.abspath("temp_inference")
        
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        try:
            # 2. Run OpenPose
            # Note: This calls the function from video_to_keypoints.py
            # Ensure OpenPose is configured there or it will run in mock mode.
            
            # We don't ask OpenPose to write video anymore, we do it ourselves for MP4 support
            json_out_dir = run_openpose(video_path, temp_dir)
            
            # 3. Get Video Info
            cap = cv2.VideoCapture(video_path)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # 4. Preprocess
            # Check if OpenPose actually produced output
            json_files = glob.glob(os.path.join(json_out_dir, "*.json"))
            if not json_files:
                raise RuntimeError("OpenPose failed to generate keypoints. Check if OpenPoseDemo.exe path is correct and accessible.")
                
            skeleton_seq = json_to_numpy(json_out_dir, num_frames)
            
            # Interpolate missing points (CRITICAL: Matches training logic)
            skeleton_seq = interpolate(skeleton_seq)
            
            # RENDER VIDEO HERE (Before normalization)
            processed_video_name = "processed_" + os.path.splitext(os.path.basename(video_path))[0] + ".webm"
            temp_video_path = os.path.join(temp_dir, processed_video_name)
            render_video(video_path, skeleton_seq, temp_video_path)
            
            norm_seq = normalize_skeleton(skeleton_seq, width, height)
            
            # Pad/Crop to 32 frames
            T, J, C = norm_seq.shape
            target_T = 32
            if T < target_T:
                padding = np.zeros((target_T - T, J, C), dtype=norm_seq.dtype)
                norm_seq = np.concatenate((norm_seq, padding), axis=0)
            elif T > target_T:
                start = (T - target_T) // 2
                norm_seq = norm_seq[start : start + target_T]
            
            # Transpose to (C, T, J)
            input_tensor = torch.from_numpy(norm_seq.transpose(2, 0, 1)).float()
            input_tensor = input_tensor.unsqueeze(0).to(self.device) # Add batch dim: (1, C, T, J)
            
            # 5. Inference
            try:
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            except Exception as e:
                print(f"Error during inference: {e}")
                raise e
            
            # Move processed video to static folder
            static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "static")
            os.makedirs(static_dir, exist_ok=True)
            final_video_path = os.path.join(static_dir, processed_video_name)
            
            video_url = None
            if os.path.exists(temp_video_path):
                shutil.move(temp_video_path, final_video_path)
                # URL relative to backend mount
                video_url = f"/static/{processed_video_name}"
            
            # 6. Format Result
            result = {
                "classes": CLASSES,
                "probabilities": probs.tolist(),
                "top_class": CLASSES[np.argmax(probs)],
                "top_probability": float(np.max(probs)),
                "video_url": video_url
            }
            return result
            
        finally:
            # Cleanup
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Test
    recognizer = ActionRecognizer()
    # Create a dummy video file for testing if needed
    if not os.path.exists("test.avi"):
        # Create a dummy video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('test.avi', fourcc, 20.0, (160, 120))
        for _ in range(50):
            frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
    res = recognizer.predict_video("test.avi")
    print(res)
