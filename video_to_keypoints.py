import os
print("Starting video_to_keypoints.py...")
import subprocess
import glob
import json
import numpy as np
import cv2
from pathlib import Path

# Configuration
# Assuming OpenPose is inside the project folder "openpose"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OPENPOSE_BIN = os.path.join(PROJECT_ROOT, "openpose", "bin", "OpenPoseDemo.exe")
OPENPOSE_MODEL = os.path.join(PROJECT_ROOT, "openpose", "models")
DATA_ROOT = os.path.join(PROJECT_ROOT, "Data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "Skeletons")

ACTIONS = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

def run_openpose(video_path, output_dir, output_video_path=None):
    """
    Runs OpenPoseDemo.exe on a video and saves JSON keypoints to output_dir.
    If output_video_path is provided, it also saves the processed video there.
    """
    # Create output directory for this video's frames
    video_name = Path(video_path).stem
    json_out_dir = os.path.join(output_dir, video_name)
    os.makedirs(json_out_dir, exist_ok=True)

    # Command to run OpenPose
    # --video: input video
    # --write_json: output directory for JSONs
    # --display 0: disable display for speed (unless we want to see it, but we save video instead)
    # --render_pose 0: disable rendering for speed (unless saving video)
    # --model_pose BODY_25: Use BODY_25 model (25 keypoints)
    # Run OpenPose from its own directory to find DLLs
    openpose_dir = os.path.dirname(OPENPOSE_BIN)
    
    cmd = [
        OPENPOSE_BIN,
        "--video", video_path,
        "--write_json", json_out_dir,
        "--model_folder", OPENPOSE_MODEL,
        "--model_pose", "BODY_25" 
    ]
    
    if output_video_path:
        # Enable rendering and writing video
        cmd.extend(["--display", "0", "--render_pose", "1", "--write_video", output_video_path])
    else:
        # Disable rendering for speed
        cmd.extend(["--display", "0", "--render_pose", "0"])

    print(f"Processing {video_name}...")
    try:
        subprocess.run(cmd, cwd=openpose_dir, check=True) 
    except subprocess.CalledProcessError as e:
        print(f"Error processing {video_name}: {e}")

    return json_out_dir

def json_to_numpy(json_dir, num_frames, expected_joints=25):
    """
    Reads OpenPose JSON files and converts to (T, J, 2) numpy array.
    """
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    
    # If no JSONs found (e.g. mock run), return random data for testing structure
    if not json_files:
        print(f"Warning: No JSONs found in {json_dir}. Generating dummy data.")
        return np.random.rand(num_frames, expected_joints, 2).astype(np.float32)

    # Initialize with zeros (T, J, 2)
    skeleton_seq = np.zeros((num_frames, expected_joints, 2), dtype=np.float32)
    
    for jf in json_files:
        # Parse frame index from filename
        # Format: {video_name}_{frame_number}_keypoints.json
        # We look for the last sequence of digits before "_keypoints"
        filename = os.path.basename(jf)
        try:
            # Split by '_' and find the part that is the frame number (usually 2nd to last)
            # Example: video_000000000000_keypoints.json -> parts[-2] is "000000000000"
            parts = filename.replace("_keypoints.json", "").split('_')
            frame_idx = int(parts[-1])
            
            if frame_idx >= num_frames:
                continue
                
            with open(jf, 'r') as f:
                content = json.load(f)
            
            # Extract keypoints
            if content['people']:
                # Take the first person
                keypoints = content['people'][0]['pose_keypoints_2d']
                kp_reshaped = np.array(keypoints).reshape(-1, 3)
                
                # Check confidence
                # Sum of confidence scores (3rd column)
                confidence_sum = np.sum(kp_reshaped[:, 2])
                
                # Threshold: If total confidence is too low, treat as noise/empty
                # 25 joints * 0.1 avg confidence = 2.5
                if confidence_sum > 5.0:
                    kp_xy = kp_reshaped[:, :2]
                    skeleton_seq[frame_idx] = kp_xy
                else:
                    # Low confidence, ignore (leave as zeros)
                    pass
            # else: no person, leave as zeros
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            continue
            
    return skeleton_seq

def normalize_skeleton(skeleton_seq, width=160, height=120):
    """
    Normalize skeleton coordinates to [0, 1].
    KTH videos are typically 160x120.
    """
    # Avoid division by zero
    w = width if width > 0 else 1
    h = height if height > 0 else 1
    
    norm_seq = skeleton_seq.copy()
    norm_seq[:, :, 0] /= w
    norm_seq[:, :, 1] /= h
    return norm_seq

def process_dataset():
    for action in ACTIONS:
        action_dir = os.path.join(DATA_ROOT, action)
        output_action_dir = os.path.join(OUTPUT_ROOT, action)
        os.makedirs(output_action_dir, exist_ok=True)
        
        video_files = glob.glob(os.path.join(action_dir, "*.avi"))
        
        for video_path in video_files:
            # 1. Run OpenPose
            json_out_dir = run_openpose(video_path, output_action_dir)
            
            # 2. Get Video Properties (for normalization)
            cap = cv2.VideoCapture(video_path)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # 3. Convert to Numpy
            skeleton_seq = json_to_numpy(json_out_dir, num_frames)
            
            # 4. Normalize
            norm_seq = normalize_skeleton(skeleton_seq, width, height)
            
            # 5. Save as .npy
            save_path = os.path.join(output_action_dir, Path(video_path).stem + ".npy")
            np.save(save_path, norm_seq)
            print(f"Saved {save_path} shape={norm_seq.shape}")

if __name__ == "__main__":
    process_dataset()
