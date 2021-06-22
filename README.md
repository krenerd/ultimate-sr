# One-to-Many Approach for Improving Perceptual Super-Resolution :satisfied: [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/one-to-many-approach-for-improving-super/image-super-resolution-on-div8k-val-16x)](https://paperswithcode.com/sota/image-super-resolution-on-div8k-val-16x?p=one-to-many-approach-for-improving-super)

Official Implementation of **Compatible Training Objective for Improving Perceptual Super-Resolution** in Tensorflow 2.0+. 

This repository contains the implementation and training of the methods proposed in the paper Compatible Training Objective for Improving Perceptual Super-Resolution.(Link)

![Diagram of our method](./readme/diagram.png)

The methods presented in our paper were implemented with the ESRGAN network from ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks by Xintao Wang et al. In our work we propose the following:

* We provide weigthed random noise to the generator to provide it with the ability to generate diverse outputs.
* We propose a weaker content loss that is compatible with the multiple outputs of the generator, and does not contradict the adversarial loss.
* We improve the SR quality by filtering blurry regions in the training data using Laplacian activation.
* We additionally provide the LR image to the discriminator as a reference image to give better gradient feedback to the generator.


## Training and Testing

### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Configuration File
You can modify the configurations of our models in [./configs/*.yaml](https://github.com/krenerd/ultimate-sr/tree/master/configs) for training and testing, which like below.

```python
# general setting
batch_size: 16
input_size: 32
gt_size: 128
ch_size: 3
scale: 4
log_dir: '/content/drive/MyDrive/ESRGAN'
pretrain_dir: '/content/drive/MyDrive/ESRGAN-MSE-500K-DIV2K'  # directory to load from at initial training
cycle_mse: True
# generator setting
network_G:
    name: 'RRDB'
    nf: 64
    nb: 23
    apply_noise: True
# discriminator setting
network_D:
    nf: 64

# dataset setting
train_dataset:
    path: '/content/drive/MyDrive/data/div2k_hr/DIV2K_train_HR'
    num_samples: 32208
    using_bin: True
    using_flip: True
    using_rot: True
    detect_blur: True
    buffer_size: 1024           # max size of buffer
    patch_per_image: 128        # number of patches to extract from each image
test_dataset:
    set5: './test data/Set5'
    set14: './test data/Set14'

# training setting
niter: 400000

lr_G: !!float 1e-4
lr_D: !!float 1e-4
lr_steps: [50000, 100000, 200000, 300000]
lr_rate: 0.5

adam_beta1_G: 0.9
adam_beta2_G: 0.99
adam_beta1_D: 0.9
adam_beta2_D: 0.99

w_pixel: !!float 1e-2
pixel_criterion: l1

w_feature: 1.0
feature_criterion: l1

w_gan: !!float 5e-3
gan_type: ragan  # gan | ragan
refgan: True       # provide reference image

save_steps: 5000

# logging settings
logging:
    psnr: True
    lpips: True
    ssim: True
    plot_samples: True
```

`cycle_mse`: use cycle-consistent content loss

`network_G/apply_noise`: provide random noise to the generator network

`train_dataset/detect_blur`: filter blurry images in the training dataset

`refgan`: provide reference image to the discriminator network

Explanation of *config* files:
- `esrgan.yaml`: baseline ESRGAN (configuration(c))
- `esrrefgan.yaml`: +refgan (configuration(d))
- `use_noise.yaml`: +use noise (configuration(e))
- `cyclegan.yaml`: +cycle loss (configuration(f))
- `cyclegan_only.yaml`: -perceptual loss (configuration(g))

### Training
The training process is divided into two parts;
pretraining the model with pixel-wise loss, and training the pretrained PSNR model with ESRGAN loss.

#### Pretrain PSNR
Pretrain the PSNR RDDB model.
```bash
python train_psnr.py --cfg_path="./configs/psnr.yaml" --gpu=0
```

#### ESRGAN
Train the ESRGAN model with the pretrain PSNR model.
```bash
python train_esrgan.py --cfg_path="./configs/esrgan.yaml" --gpu=0
```
Configure the dataset directory and log directory in the config file before training. The DIV2K dataset is available [here](https://drive.google.com/drive/folders/1jgvj8oBpYBwK6S2xe2gX9LF50r5a2pYX?usp=sharing) and the DIV8K dataset is avilable [here](https://drive.google.com/drive/folders/1WuwWfc0X5ORF3zT7Z-5Soisbpyh-LDy_?usp=sharing).

## Evaluation

```
python test.py --model=weights/ESRGAN-cyclemixing --gpu=0 --img_path=photo/baby.png --down_up=True --scale=4(optional)
```
When the `down_up` option is `True`, the image will be arbitrarily downsampled and processed through the network. For real use cases, the option must be marked `False` for the model to upsample the image.  
## Pre-trained models and logs

All our training checkpoints and `tensorboard` logs in the experiment can be downloaded [here](https://drive.google.com/drive/folders/1AmsOyI1hf0jJBY1WvZJIaj1aDobfXM-G?usp=sharing). Model `.tb` files can be downloded [here](https://drive.google.com/drive/folders/13WOQc15styMJNfTAmoKASlub5Yd5vebj?usp=sharing), and four trained models are included in the repository in `./weights/*`.

## Results

Our methods were evaluated on LPIPS, PSNR, and SSIM using the Set5, Set14, BSD100, Urban100, and Manga109 dataset. The scores are displayed in the tables below, in the order LPIPS/PSNR/SSIM.

### Pretrained PSNR Network

| <sub>Method</sub> | <sub>Set5</sub> | <sub>Set14</sub> | <sub>BSD100</sub> | <sub>Urban100</sub> |
|:---:|:---:|:---:|:---:|:---:|
| <sub>Baseline PSNR</sub> | <sub>0.1341 / 30.3603 / ****0.8679**** </sub> |<sub>****0.2223**** / 26.7608 / 0.7525</sub>|<sub>0.2705 / 27.2264 / 0.7461</sub>|<sub>0.1761 / 24.8770 / 0.7764</sub>|
| <sub>+Blur detection</sub> | <sub>****0.1327**** / ****30.4582**** / 0.7525</sub> | <sub>0.2229 / ****26.8448**** / ****0.7547****</sub> | <sub>****0.2684**** / ****27.2545**** / **0.7473**</sub> | <sub>****0.1744**** / ****25.0816**** / ****0.7821****</sub> |

### X4 super-resolution

| <sub>Method</sub> | <sub>Set5</sub> | <sub>Set14</sub> | <sub>BSD100</sub> | <sub>Urban100</sub> |
|:---:|:---:|:---:|:---:|:---:|
| <sub>ESRGAN (Official)</sub> | <sub>0.0597 / 28.4362 / 0.8145</sub> |<sub>0.1129 / 23.4729 / 0.6276</sub>|<sub>0.1285 / 23.3657 / 0.6108</sub>|<sub>0.1025 / 22.7912 / 0.7058</sub>|
| <sub>ESRGAN (Baseline)</sub> | <sub> 0.0538 / 27.9285 / 0.7968 </sub> |<sub>0.1117 / 24.5264 / 0.6602</sub>|<sub>0.1256 / 24.6554 / 0.6447</sub>|<sub>0.1026 / **23.2829** / 0.7137</sub>|
| <sub>+refGAN</sub> | <sub>0.0536 / 27.9871 / 0.8014</sub> | <sub>0.1157 / 24.4505 / 0.6611</sub> | <sub>0.1275 / 24.5896 / 0.6470</sub> | <sub>0.1027 / 23.0496 / 0.7103</sub> |
| <sub>+Add noise</sub> | <sub>**0.04998** / **28.23** / **0.8081**</sub> | <sub>0.1104 / 24.48 / 0.6626</sub> | <sub>**0.1209** / **24.8439** / **0.6577**</sub> | <sub>**0.1007** / 23.2204 / **0.7203**</sub>|
|<sub>+Cycle loss</sub> | <sub>0.0524 / 28.1322 / 0.8033</sub> |<sub>**0.1082** / **24.5802** / **0.6634**</sub> |<sub>0.1264 / 24.6180 / 0.6468</sub> |<sub>0.1015 / 23.1363 / 0.7103</sub> |
|<sub>-Perceptual loss</sub> | <sub>0.2690 / 23.4608 / 0.6312</sub> |<sub>0.2727 / 22.2703 / 0.5685</sub> |<sub>0.2985 / 24.1648 / 0.5859</sub> |<sub>0.2411 / 20.8169 / 0.6244</sub> |


![Comparison of results](./readme/result_comparison.png)
* Our work was made upon [this](https://github.com/peteryuX/esrgan-tf2) Tensorflow 2.x implementation of ESRGAN. Specail thanks to the creators of the repository.
