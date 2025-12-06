import numpy as np
import glob
import os

def inspect_data():
    # Load a sample from 'jogging'
    files = glob.glob(r"F:\KTP-CNN-PROJECT\Skeletons\jogging\*.npy")
    if not files:
        print("No jogging files found!")
        return

    sample_path = files[0]
    data = np.load(sample_path) # (T, 25, 2)
    
    print(f"Inspecting: {os.path.basename(sample_path)}")
    print(f"Shape: {data.shape}")
    
    # Check leg keypoints for BODY_25
    # 8: MidHip, 9: RHip, 10: RKnee, 11: RAnkle, 12: LHip, 13: LKnee, 14: LAnkle
    leg_indices = [9, 10, 11, 12, 13, 14]
    
    # Check if they are all zeros
    legs = data[:, leg_indices, :]
    non_zero_legs = np.count_nonzero(legs)
    total_legs = legs.size
    
    print(f"Non-zero leg values: {non_zero_legs} / {total_legs}")
    if non_zero_legs == 0:
        print("❌ CRITICAL: No leg detection found! The model cannot see running/walking.")
    else:
        print("✅ Legs detected.")
        print("Sample frame 0 legs:\n", legs[0])

if __name__ == "__main__":
    inspect_data()
