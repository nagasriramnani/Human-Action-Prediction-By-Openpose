@echo off
echo Creating Conda Environment KTP-TASK...

call conda create -n KTP-TASK python=3.9 -y
call conda activate KTP-TASK

echo Installing PyTorch with CUDA 11.8...
call conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo Installing other dependencies...
call pip install numpy opencv-python scikit-learn matplotlib pandas tqdm

echo Setup Complete!
echo To activate: conda activate KTP-TASK
pause
