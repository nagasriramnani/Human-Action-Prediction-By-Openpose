import torch
import time

def check_cuda():
    print("="*40)
    print("CUDA DIAGNOSTIC TOOL")
    print("="*40)

    # 1. Check Availability
    is_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_available}")
    
    if not is_available:
        print("❌ CUDA is NOT available. Training will be slow (CPU only).")
        return

    # 2. Device Info
    device_count = torch.cuda.device_count()
    print(f"GPU Count: {device_count}")
    
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Current GPU: {device_name}")

    # 3. Memory Test
    print("\n[Running Tensor Test...]")
    try:
        # Create a large tensor on GPU
        x = torch.rand(10000, 10000).cuda()
        y = torch.rand(10000, 10000).cuda()
        
        start_time = time.time()
        # Perform matrix multiplication
        z = torch.matmul(x, y)
        end_time = time.time()
        
        print(f"✅ Success! Matrix multiplication on GPU took {end_time - start_time:.4f} seconds.")
        print("CUDA is correctly configured and working.")
        
    except Exception as e:
        print(f"❌ Error using GPU: {e}")

    print("="*40)

if __name__ == "__main__":
    check_cuda()
