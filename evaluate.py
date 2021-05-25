import tensorflow as tf
import numpy as np
import lpips
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.dataset import load_valid_dataset
from modules.models import RRDB_Model, RRDB_Model_16x, RFB_Model_16x
from modules.utils import plot_to_image,load_yaml
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

    data_list = data_list[:min(14, len(data_list))]       # plot first 14 images if more data exists
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

def evaluate_dataset(dataset, model, cfg, return_sum=True):
    # evalutaes ever
    sum_LPIPS, sum_PSNR, sum_SSIM = 0, 0, 0
    list_PSNR, list_SSIM, list_LPIPS = [], [], []
    data_list = []    # save all SR image for plotting

    print("Evaluating on test dataset.")
    for lr, hr in tqdm(dataset, position=0, leave=True):
        sr = model(lr[np.newaxis,:], training=False)[0] #Generate SR image
        
        if return_sum:
            if cfg['logging']['psnr']:
                sum_PSNR += evaluate_psnr(sr, hr)
            if cfg['logging']['ssim']:
                sum_SSIM += evaluate_ssim(sr, hr)
            if cfg['logging']['lpips']:
                sum_LPIPS += evaluate_lpips(sr, hr)
        else:
            if cfg['logging']['psnr']:
                list_PSNR.append(evaluate_psnr(sr, hr))
            if cfg['logging']['ssim']:
                list_SSIM.append(evaluate_ssim(sr, hr))
            if cfg['logging']['lpips']:
                list_LPIPS.append(evaluate_lpips(sr,hr))
        
        if cfg['logging']['plot_samples']:
            data_list.append((sr, hr))

    num_data = len(dataset)
    logs={}
    if return_sum:
        if cfg['logging']['psnr']:
            logs['psnr'] = sum_PSNR / num_data
        if cfg['logging']['ssim']:
            logs['ssim'] = sum_SSIM / num_data
        if cfg['logging']['lpips']:
            logs['lpips'] = sum_LPIPS / num_data
    else:
        if cfg['logging']['psnr']:
            logs['psnr'] = list_PSNR 
        if cfg['logging']['ssim']:
            logs['ssim'] = list_SSIM 
        if cfg['logging']['lpips']:
            logs['lpips'] = list_LPIPS  
    if cfg['logging']['plot_samples']:
        logs['samples'] = plot_examples(data_list)

    return logs

def get_noise_layers(generator, plot_layer_wise=True):
    noise_feature=[]
    def recurse_all_layer(layer):
        try:  # pass only when layer is tf.keras.Model
            _=layer.layers
        except: # if layer is tf.keras.Layer
            if hasattr(layer,'applynoise'):     # if layer has noise layer
                noise_feature.append(layer.applynoise.weights[0])
            return
        
        for sublayer in layer.layers:
            recurse_all_layer(sublayer)
    
    recurse_all_layer(generator)

    if plot_layer_wise:
        max_arr, mean_arr, noise_arr=[],[],[]
        for x in noise_feature:
            max_arr.append(max(np.abs(x)))
            mean_arr.append(np.median(np.abs(x)))
            noise_arr.append(np.abs(x))


        plt.boxplot(noise_arr)
        plt.plot(np.arange(1, len(max_arr)+1),max_arr, c='black')
        plt.plot(np.arange(1, len(max_arr)+1),mean_arr, c='orange')
        plt.xticks([])
        plt.xlabel('# layer')
        plt.show()

    return noise_feature

def evaluate_with_path(cfg_path, dataset_path, scale=4, return_sum=True):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    cfg=load_yaml(cfg_path)
    if cfg['network_G']['name']=='RRDB':    # ESRGAN 4x
        model = RRDB_Model(None, cfg['ch_size'], cfg['network_G'])
    elif cfg['network_G']['name']=='RRDB_CIPLAB':
        model = RRDB_Model_16x(None, cfg['ch_size'], cfg['network_G'])
    elif cfg['network_G']['name']=='RFB_ESRGAN':
        model = RFB_Model_16x(None, cfg['ch_size'], cfg['network_G'])
    
    checkpoint_dir = cfg['log_dir'] + '/checkpoints/'
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'), model=model)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        print(f"Checkpoint doesn't exist in {checkpoint_dir}.")

    cfg_logging={'logging': {'lpips':True, 'psnr':True, 'ssim': True, 'plot_samples': False}}
    dataset = load_valid_dataset(dataset_path, scale, crop_centor=0)

    logs = evaluate_dataset(dataset, model, cfg_logging, return_sum=return_sum)
    return logs