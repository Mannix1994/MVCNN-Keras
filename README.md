# HI

this is the MVCNN model coding with keras.  
Inspired by WeiTang114's project [MVCNN-TensorFlow](https://github.com/WeiTang114/MVCNN-TensorFlow).

# Requirements
* CUDA 9.0 (if you have NVIDIA GPU)
* python 2.7 or python 3.5+
* tensorflow 1.12.0
* tensorflow-gpu 1.12.0 (if you have NVIDIA GPU)
* nvidia-ml-py(for python 2.7)
* nvidia-ml-py3(for python 3.5+)
* some other python packages

# Dataset
In the original [MVCNN](https://github.com/suhangpro/mvcnn) project page


# Train
```bash
# for having zero gpu or one gpu
python train.py
# for having more than one gpu
python train.py -u
```

# Evaluate
```bash
pthon evaluate.py
```

# Predict 
```bash
python predict.py
```

# Note
there still have some overfitting problem