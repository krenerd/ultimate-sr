import tensorflow as tf
import numpy as np
import lpips

def evaluate_lpips(sr, hr):
    loss_fn_alex = lpips.LPIPS(net='alex')
    pass

def evaluate_psnr(sr, hr):
    pass

def evaluate_ssim(sr, hr):
    pass

def plot_examples(sr, hr):
    pass

def evaluate_dataset(data_path, model, lpips=True, psnr=True, 
                    plot=True):
    


