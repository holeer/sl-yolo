df -h
apt install nvidia-cuda-toolkit
nvcc --version

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init

conda create -n sunli python=3.8
conda activate sunli
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.5 -c pytorch -c nvidia
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu115