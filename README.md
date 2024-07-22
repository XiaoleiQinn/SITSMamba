# SITSMamba

##environment
conda create -n mamba python==3.8
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
wget https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2+cu122torch2.2cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm-1.2.2+cu122torch2.2cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
