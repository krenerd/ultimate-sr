import tensorflow as tf
import numpy as np
import lpips
import torch
import matplotlib.pyplot as plt
from modules.utils import plot_to_image
lpips_alex = lpips.LPIPS(net='alex')

def evaluate_lpips(sr, hr):
    sr, hr=tf.expand_dims(tf.transpose(sr, [2, 0, 1]), axis=0), tf.expand_dims(tf.transpose(hr, [2, 0, 1]), axis=0)
    res=lpips_alex.forward(torch.Tensor(hr.numpy()), torch.Tensor(sr.numpy())) #Calculate LPIPS Similarity
    return res.detach().numpy().flatten()[0]

def evaluate_psnr(sr, hr):
    def PSNR(y_true,y_pred, image_range = 1):
        mse=tf.reduce_mean( (y_true - y_pred) ** 2 )
        return 20 * log10(image_range / (mse ** 0.5))

    def log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    res=PSNR(hr, sr)
    
    return res.numpy()

def evaluate_ssim(sr, hr):
    return tf.image.ssim(sr, hr, max_val=1, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)

def plot_examples(data_list, plot_size=4):
    # returns a image with all (SR, HR) pair visualized
    plt.close()
    num_data = len(data_list)
    fig=plt.figure(figsize=(plot_size * num_data, plot_size * 2))

    for idx,x in enumerate(data_list):
        plt.subplot(2, num_data, 1 + idx)
        plt.imshow(np.clip(x[0],0,1))
        plt.axis('off')

        plt.subplot(2, num_data, 1 + idx + num_data)
        plt.imshow(x[1])
        plt.axis('off')

    return plot_to_image(fig)

def evaluate_dataset(dataset, model, cfg):
    # evalutaes ever
    sum_LPIPS, sum_PSNR, sum_SSIM = 0, 0, 0
    data_list = []    # save all SR image for plotting

    for lr, hr in dataset:
        sr = model(lr[np.newaxis,:], training=False)[0] #Generate SR image
        
        if cfg['logging']['psnr']:
            sum_PSNR += evaluate_psnr(sr, hr)
        if cfg['logging']['ssim']:
            sum_SSIM += evaluate_ssim(sr, hr)
        if cfg['logging']['lpips']:
            sum_LPIPS += evaluate_lpips(sr, hr)
        
        if cfg['logging']['plot_samples']:
            data_list.append((sr, hr))

    num_data = len(dataset)
    logs={}
    if cfg['logging']['psnr']:
        logs['psnr'] = sum_PSNR / num_data
    if cfg['logging']['ssim']:
        logs['ssim'] = sum_SSIM / num_data
    if cfg['logging']['lpips']:
        logs['lpips'] = sum_LPIPS / num_data
    if cfg['logging']['plot_samples']:
        logs['samples'] = plot_examples(data_list)

    return logs
