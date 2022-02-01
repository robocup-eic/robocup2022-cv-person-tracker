# robocup2022-cv-person-tracker
python 3.7 <br>
download other files from https://drive.google.com/file/d/17N64aBPytocPqlA_pp7DCRruDQDeg3Ir/view?usp=sharing <br>
ref: https://youtu.be/FuvQ8Melz1o <br>

Conda <br>
```
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

Pip <br>
```
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
