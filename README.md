# Compatible Training Objective for Improving Perceptual Super-Resolution-Tensorflow 2

Compatible Training Objective for Improving Perceptual Super-Resolution implemented in Tensorflow 2.0+.

This repository contains the implementation and training of the methods proposed in the paper Compatible Training Objective for Improving Perceptual Super-Resolution.(Link)

The methods presented in our paper were implemented with the ESRGAN network from ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks by Xintao Wang et al. In our work we propose the following:

* We provide weigthed random noise to the generator to provide it with the ability to generate diverse outputs.
* We propose a weaker content loss that is compatible with the multiple outputs of the generator, and does not contradict the adversarial loss.
* We improve the SR quality by filtering blurry regions in the training data using Laplacian activation.
* We additionally provide the LR image to the discriminator as a reference image to give better gradient feedback to the generator.


Paper:     &nbsp; [Arxiv](https://arxiv.org/abs/1809.00219) &nbsp; [ECCV2018](http://openaccess.thecvf.com/content_eccv_2018_workshops/w25/html/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.html)

:: Results from our work. ::
<img src="photo/baboon_cover.png">

## Training and Testing

### Config File
You can modify your own dataset path or other settings of model in [./configs/*.yaml](https://github.com/krenerd/ultimate-sr/tree/master/configs) for training and testing, which like below.

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

`apply_noise`: provide random noise to the generator network

`detect_blur`: filter blurry images in the training dataset

`refgan`: provide reference image to the discriminator network


### Training
The training process is divided into two parts;
pretraining the PSNR RRDB model, and training the ESRGAN model with the pretrained PSNR model.
#### Pretrain PSNR
Pretrain the PSNR RDDB model.
```bash
python train_psnr.py --cfg_path="./configs/psnr.yaml" --gpu=0
```

#### ESRGAN
Train the ESRGAN model with the pretrain PSNR model.
```bash
python train_esrgan.py --cfg_path="./configs/esrgan_*.yaml" --gpu=0
```

## Results

### Table

| <sub>Method</sub> | <sub>Set5</sub> | <sub>Set14</sub> | <sub>BSD100</sub> | <sub>Urban100</sub> | <sub>Manga109</sub> |
|:---:|:---:|:---:|:---:|:---:|:---:|
| <sub>Baseline ESRGAN</sub> | <sub>30.48/0.8628</sub> |<sub>27.50/0.7513</sub>|<sub>26.90/0.7101</sub>|<sub>24.52/0.7221</sub>|<sub>27.58/0.8555</sub>|
| <sub>+refGAN</sub> | <sub>32.46/0.8968</sub> | <sub>28.80/0.7876</sub> | <sub>27.71/0.7420</sub> | <sub>26.64/0.8033</sub> | <sub>31.02/0.9148</sub> |
| <sub>+Add noise</sub> | <sub>32.63/0.9002</sub> | <sub>28.87/0.7889</sub> | <sub>27.77/0.7436</sub> | <sub>26.82/ 0.8087</sub>| <sub>31.22/ 0.9173</sub>|
|<sub>+Cycle loss</sub> | <sub>**32.73/0.9011**</sub> |<sub>**28.99/0.7917**</sub> |<sub>**27.85/0.7455**</sub> |<sub>**27.03/0.8153**</sub> |<sub>**31.66/0.9196**</sub>|
|<sub>-Perceptual loss</sub> | <sub>**32.73/0.9011**</sub> |<sub>**28.99/0.7917**</sub> |<sub>**27.85/0.7455**</sub> |<sub>**27.03/0.8153**</sub> |<sub>**31.66/0.9196**</sub>|

### **Set5**
<table>
    <thead>
        <tr>
            <th>Image Name</th>
            <th>Bicubic</th>
            <th>PSNR (pretrain)</th>
            <th>ESRGAN</th>
            <th>Ground Truth</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center" rowspan=2>baby</td>
            <td align="center" colspan=4><img src="./photo/table_baby.png"></td>
        </tr>
            <td align="center">31.96 / 0.85</td>
            <td align="center">33.86 / 0.89</td>
            <td align="center">31.36 / 0.83</td>
            <td align="center">-</td>
        <tr>
        </tr>
        <tr>
            <td align="center" rowspan=2>bird</td>
            <td align="center" colspan=4><img src="./photo/table_bird.png"></td>
        </tr>
            <td align="center">30.27 / 0.87</td>
            <td align="center">35.00 / 0.94</td>
            <td align="center">32.22 / 0.90</td>
            <td align="center">-</td>
        <tr>
        </tr>
        <tr>
            <td align="center" rowspan=2>butterfly</td>
            <td align="center" colspan=4><img src="./photo/table_butterfly.png"></td>
        </tr>
            <td align="center">22.25 / 0.72</td>
            <td align="center">28.56 / 0.92</td>
            <td align="center">26.66 / 0.88</td>
            <td align="center">-</td>
        <tr>
        </tr>
        <tr>
            <td align="center" rowspan=2>head</td>
            <td align="center" colspan=4><img src="./photo/table_head.png"></td>
        </tr>
            <td align="center">32.01 / 0.76</td>
            <td align="center">33.18 / 0.80</td>
            <td align="center">30.19 / 0.70</td>
            <td align="center">-</td>
        </tr>
        <tr>
            <td align="center" rowspan=2>woman</td>
            <td align="center" colspan=4><img src="./photo/table_woman.png"></td>
        </tr>
            <td align="center">26.44 / 0.83</td>
            <td align="center">30.42 / 0.92</td>
            <td align="center">28.50 / 0.88</td>
            <td align="center">-</td>
        </tr>
    </tbody>
</table>

### **Set14 (Partial)**
<table>
    <thead>
        <tr>
            <th>Image Name</th>
            <th>Bicubic</th>
            <th>PSNR (pretrain)</th>
            <th>ESRGAN</th>
            <th>Ground Truth</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center" rowspan=2>baboon</td>
            <td align="center" colspan=4><img src="./photo/table_baboon.png"></td>
        </tr>
            <td align="center">22.06 / 0.45</td>
            <td align="center">22.77 / 0.54</td>
            <td align="center">20.73 / 0.44</td>
            <td align="center">-</td>
        <tr>
        </tr>
        <tr>
            <td align="center" rowspan=2>comic</td>
            <td align="center" colspan=4><img src="./photo/table_comic.png"></td>
        </tr>
            <td align="center">21.69 / 0.59</td>
            <td align="center">23.46 / 0.74</td>
            <td align="center">21.08 / 0.64</td>
            <td align="center">-</td>
        <tr>
        </tr>
        <tr>
            <td align="center" rowspan=2>lenna</td>
            <td align="center" colspan=4><img src="./photo/table_lenna.png"></td>
        </tr>
            <td align="center">29.67 / 0.80</td>
            <td align="center">32.06 / 0.85</td>
            <td align="center">28.96 / 0.80</td>
            <td align="center">-</td>
        <tr>
        </tr>
        <tr>
            <td align="center" rowspan=2>monarch</td>
            <td align="center" colspan=4><img src="./photo/table_monarch.png"></td>
        </tr>
            <td align="center">27.60 / 0.88</td>
            <td align="center">33.27 / 0.94</td>
            <td align="center">31.49 / 0.92</td>
            <td align="center">-</td>
        </tr>
        <tr>
            <td align="center" rowspan=2>zebra</td>
            <td align="center" colspan=4><img src="./photo/table_zebra.png"></td>
        </tr>
            <td align="center">24.15 / 0.68</td>
            <td align="center">27.29 / 0.78</td>
            <td align="center">24.86 / 0.67</td>
            <td align="center">-</td>
        </tr>
    </tbody>
</table>

