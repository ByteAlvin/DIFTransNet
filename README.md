# STFUNet：Swin Transformer Unet with Hierarchical Fusion Module for Infrared Small Target Detection


## Algorithm Introduction

Dense Nested Attention Network for Infrared Small Target Detection, Boyang Li, Chao Xiao, Longguang Wang, and Yingqian Wang, arxiv 2021 [[Paper]](https://arxiv.org/pdf/2106.00487.pdf)

We propose a dense nested attention network (DNANet) to achieve accurate single-frame infrared small target detection and develop an open-sourced infrared small target dataset (namely, NUDT-SIRST) in this paper. Experiments on both public (e.g., NUAA-SIRST, NUST-SIRST) and our self-developed datasets demonstrate the effectiveness of our method. The contribution of this paper are as follows:

1. We propose a dense nested attention network (namely, DNANet) to maintain small targets in deep layers.

2. An open-sourced dataset (i.e., NUDT-SIRST) with rich targets.

3. Performing well on all existing SIRST datasets.




## Prerequisite

* Tested on Ubuntu 16.04, with Python 3.7, PyTorch 1.7, Torchvision 0.8.1, CUDA 11.1, and 1x NVIDIA 3090 and also 

* Tested on Windows 10  , with Python 3.6, PyTorch 1.1, Torchvision 0.3.0, CUDA 10.0, and 1x NVIDIA 1080Ti.

* [The NUDT-SIRST download dir](https://pan.baidu.com/s/1WdA_yOHDnIiyj4C9SbW_Kg?pwd=nudt) (Extraction Code: nudt)

* [The NUAA-SIRST download dir](https://github.com/YimianDai/sirst) [[ACM]](https://arxiv.org/pdf/2009.14530.pdf)

* [The NUST-SIRST download dir](https://github.com/wanghuanphd/MDvsFA_cGAN) [[MDvsFA]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Miss_Detection_vs._False_Alarm_Adversarial_Learning_for_Small_Object_ICCV_2019_paper.pdf)

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

#### Quantitative Results on Mixed SIRST, NUDT-SIRST, and IRSTD-1K

| Model         | mIoU (x10(-2)) | nIoU (x10(-2)) | F-measure (x10(-2))| Pd (x10(-2))|Fa (x10(-6))| Weights|
| ------------- |:-------------:|:-----:|:-----:|:-----:|:-----:|:-----:|
| NUDT-SIRST    | 94.09  |  94.38 | 96.95 | 98.62 | 4.29  |[[best.pt]](https://drive.google.com/file/d/1Kxs2wKG2uq2YiGJOBGWoVz7B1-8DJoz3/view?usp=sharing)|
| IRSTD-1K      | 68.03  |  68.15 | 80.96 | 93.27 | 10.74 |[[best.pt]](https://drive.google.com/file/d/1Kxs2wKG2uq2YiGJOBGWoVz7B1-8DJoz3/view?usp=sharing)|

*This code is highly borrowed from [SCTransNet](https://github.com/YimianDai/open-acm). Thanks to Shuai Yuan.

*The overall repository style is highly borrowed from [DNA-Net](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Thanks to Boyang Li.

## Contact
**Welcome to raise issues or email to [liushenao23@mails.ucas.ac.cn](liushenao23@mails.ucas.ac.cn) or [shenaoliu@163.com](shenaoliu@163.com) for any question regarding our STFUNet.**
