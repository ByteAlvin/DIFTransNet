# STFUNet：Swin Transformer Unet with Hierarchical Fusion Module for Infrared Small Target Detection


## Algorithm Introduction



We propose a dense nested attention network (DNANet) to achieve accurate single-frame infrared small target detection and develop an open-sourced infrared small target dataset (namely, NUDT-SIRST) in this paper. Experiments on both public (e.g., NUAA-SIRST, NUST-SIRST) and our self-developed datasets demonstrate the effectiveness of our method. The contribution of this paper are as follows:

1. We propose a dense nested attention network (namely, DNANet) to maintain small targets in deep layers.

2. An open-sourced dataset (i.e., NUDT-SIRST) with rich targets.

3. Performing well on all existing SIRST datasets.




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
| NUDT-SIRST    | 94.09  |  94.38 |98.62 | 4.29  |[[best.pt]](https://drive.google.com/file/d/1Kxs2wKG2uq2YiGJOBGWoVz7B1-8DJoz3/view?usp=sharing)|
| IRSTD-1K      | 68.03  |  68.15 |  93.27 | 10.74 |[[best.pt]](https://drive.google.com/file/d/1Kxs2wKG2uq2YiGJOBGWoVz7B1-8DJoz3/view?usp=sharing)|

*This code is highly borrowed from [SCTransNet](https://github.com/YimianDai/open-acm). Thanks to Shuai Yuan.

*The overall repository style is highly borrowed from [BasicIRSTD]([https://github.com/YeRen123455/Infrared-Small-Target-Detection](https://github.com/XinyiYing/BasicIRSTD)). Thanks to Xinyi Ying.

## Contact
**Welcome to raise issues or email to [liushenao23@mails.ucas.ac.cn](liushenao23@mails.ucas.ac.cn) or [shenaoliu@163.com](shenaoliu@163.com) for any question regarding our STFUNet.**
