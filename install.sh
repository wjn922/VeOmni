# python 3.11, cuda 12, torch2.8
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# uv
pip3 install uv==0.9.8

# veomni
uv pip install -v -e .[gpu]

# video
conda install -c conda-forge "ffmpeg>=6" -y

# lmms-eval
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv pip install -e ".[all]"
cd ..