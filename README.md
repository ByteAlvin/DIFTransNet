# DIFTransNet: Dual-Branch Interactive Fusion Network with CNN and Multi-Scale Transformer for Infrared Small Target Detection

Code repository will be publicly released upon paper acceptance !

## Prerequisite

* Tested on CentOS 7, with Python 3.7, PyTorch 1.7, Torchvision 0.11.2, CUDA 11.1, and 1x NVIDIA 2080Ti and also 
* Tested on Windows 11, with Python 3.11, PyTorch 2.2, Torchvision 0.17.1, CUDA 12.1, and 1x NVIDIA 4060.
* **NUDT-SIRST** &nbsp; [[download]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/abstract/document/9864119)
* **IRSTD-1K** &nbsp; [[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)

## Usage

#### 1. Data

* **Our project has the following structure:**
  ```
  ├──./datasets/
  │    ├── IRSTD-1K
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_IRSTD-1K.txt
  │    │    │    ├── test_IRSTD-1K.txt
  │    ├── NUDT-SIRST
  │    │    ├── images
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_NUDT-SIRST.txt
  │    │    │    ├── test_NUDT-SIRST.txt
  │    ├── ...
  │    ├── ...
  │    ├── SIRST3
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_SIRST3.txt
  │    │    │    ├── test_SIRST3.txt
  
  ```


##### 2. Train.
```bash
CUDA_VISIBLE_DEVICES=1  python train.py ----dataset_names NUDT-SIRST --patchSize 256
```
```bash
CUDA_VISIBLE_DEVICES=1  python train.py ----dataset_names IRSTD-1K --patchSize 512
```

#### 3. Test and demo.
```bash
python test.py
```
## Results and Trained Models

#### Quantitative Results on NUDT-SIRST, and IRSTD-1K

| Model         | pixAcc (x10(-2)) | mIoU (x10(-2)) | Pd (x10(-2))|Fa (x10(-6))| Weights|
| :-------------: |:-------------:|:-----:|:-----:|:-----:|:-----:|
| NUDT-SIRST    | 96.69  |  95.15 |99.05 | 3.88  |[[best.pt]](https://pan.baidu.com/s/1J2Ibn-3vyWobxD6vybWUAA?pwd=urky)|
| IRSTD-1K      | 76.31  |  66.79 |  92.26 | 17.61 |[[best.pt]](https://pan.baidu.com/s/1aIG00HSpLmzGzIobXOuhNQ?pwd=ci4j)|

*This code is highly borrowed from [SCTransNet](https://github.com/YimianDai/open-acm). Thanks to Shuai Yuan.

*The overall repository style is highly borrowed from [BasicIRSTD](https://github.com/YimianDai/open-acm). Thanks to Xinyi Ying.

## Contact
**Welcome to raise issues or email to [liushenao23@mails.ucas.ac.cn](liushenao23@mails.ucas.ac.cn) or [shenaoliu@163.com](shenaoliu@163.com) for any question regarding our HESformer.**
