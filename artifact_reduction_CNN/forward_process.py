from utils.basic_blocks import *
from utils.optimizer import Optimizer
from utils.metrics import structural_similarity, peak_snr
from configuration.configuration import Configuration
import tensorflow as tf
import numpy as np
from dataset.dataset_generator import Dataset_Generator
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image


class Forward:
    def __init__(self, cf):
        self.cf = cf
        self.beta_start = cf.beta_start
        self.beta_end = cf.beta_end
        self.time_steps = cf.time_steps

    def linear_beta_schedule(self, time_steps=200):
        return tf.linspace(self.beta_start, self.beta_end, time_steps)

    def reverse_transform(self,t):
        # 将数据从[-1, 1]转换到[0, 1]
        t = (t + 1) / 2
        return t

    def calculate(self):
        betas = self.linear_beta_schedule(time_steps=self.time_steps)

        # define alphas
        alphas = 1. - betas
        alphas_cumprod = tf.math.cumprod(alphas, axis=0)
        alphas_cumprod_prev = tf.pad(alphas_cumprod[:-1], [[1, 0]], constant_values=1.0)
        sqrt_recip_alphas = tf.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = tf.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = tf.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod


    def extract(self, a, t, x_shape):
        batch_size = tf.shape(t)[0]
        # TensorFlow中不需要.cpu()和.to(device)调用
        out = tf.gather(a, t, axis=-1, batch_dims=0)
        return tf.reshape(out, [batch_size] + [1] * (len(x_shape) - 1))


    def q_sample(self, x_start, t, noise=None):
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = self.calculate()
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(x_start))

        sqrt_alphas_cumprod_t = self.extract(sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def get_noisy_image(self, x_start, t):
        original_shape = tf.shape(x_start)
        if len(x_start.shape) == 3:
            x_start = np.array(x_start)
            x_start = tf.convert_to_tensor(x_start, dtype=tf.float32) / 255.0
            # add noise
            x_noisy = self.q_sample(x_start, t=t)
            # turn back into PIL image
            noisy_image = self.reverse_transform(tf.squeeze(x_noisy))
            return noisy_image.numpy().astype(np.float32)  # Ensure numpy array with correct type

        elif len(x_start.shape) == 5:
            # 确保x_start是一个Tensor,转换为三维
            x_start = tf.convert_to_tensor(x_start, dtype=tf.float32)
            x_start_3d = tf.reshape(x_start,(original_shape[1], original_shape[2], original_shape[3]))

            # 计算最大值和最小值
            min_val = tf.reduce_min(x_start_3d)
            max_val = tf.reduce_max(x_start_3d)

            # 将强度值转换为-1到1之间
            x_start_3d = 2 * (x_start_3d - min_val) / (max_val - min_val) - 1

            # 添加噪声
            x_noisy_merged = self.q_sample(x_start_3d, t=t)
            x_noisy_merged = self.reverse_transform(tf.squeeze(x_noisy_merged))

            # 将噪声数据的形状恢复到五维
            noisy_image = tf.reshape(x_noisy_merged, original_shape)

            # 确保数据类型为float32，并转换为Tensor
            noisy_image = tf.convert_to_tensor(noisy_image)
            return noisy_image


# Load configuration files
dataset_path = 'E:/Documents/GitHub/CNN-MoCo/artifact_reduction_CNN/data/'
experiments_path = 'E:/Documents/GitHub/CNN-MoCo/artifact_reduction_CNN/experiments/'
config_file = 'E:/Documents/GitHub/CNN-MoCo/artifact_reduction_CNN/demo_config.py'
configuration = Configuration(config_file, dataset_path, experiments_path)
cf = configuration.load()
forward_process = Forward(cf)

train_dataset, val_dataset, test_dataset, pred_dataset = Dataset_Generator().make(cf)

for data in train_dataset.take(1):
    inputs, targets = data  # 假设数据集中的每个元素都是一个元组(input, target)
    inputs_itk = sitk.GetImageFromArray(inputs)
    # sitk.Show(inputs_itk, "Image Display")
    print(f"Image size: {inputs_itk.GetSize()}")
    print(f"Image spacing: {inputs_itk.GetSpacing()}")
    print(f"Image depth: {inputs_itk.GetDepth()}")
    print(f"Number of components per pixel: {inputs_itk.GetNumberOfComponentsPerPixel()}")

    noisy_image = forward_process.get_noisy_image(inputs, tf.constant([30]))  # 打印数据
    noisy_image = sitk.GetImageFromArray(noisy_image)
    print(f"Image size: {noisy_image.GetSize()}")
    print(f"Image spacing: {noisy_image.GetSpacing()}")
    print(f"Image depth: {noisy_image.GetDepth()}")
    print(f"Number of components per pixel: {noisy_image.GetNumberOfComponentsPerPixel()}")
    sitk.Show(noisy_image[0, :, :, :, 0], "Image Display")




# image_path = r'E:\Downloads\MonteCarloDatasets\MonteCarloDatasets\Training\P1\MC_T_P1_NS\FDKRecon\FDK4D_01.mha'
# image = sitk.ReadImage(image_path)
#
# print(f"Image size: {image.GetSize()}")
# print(f"Image spacing: {image.GetSpacing()}")
# print(f"Image depth: {image.GetDepth()}")
# print(f"Number of components per pixel: {image.GetNumberOfComponentsPerPixel()}")
# sitk.Show(image, "Image Display")
#
# x_start = np.array(image)
# print(x_start.shape)
#
# noisy_image = forward_process.get_noisy_image(x_start, tf.constant([2]))
# noisy_image = sitk.GetImageFromArray(noisy_image)
# print(f"Image size: {noisy_image.GetSize()}")
# print(f"Image spacing: {noisy_image.GetSpacing()}")
# print(f"Image depth: {noisy_image.GetDepth()}")
# print(f"Number of components per pixel: {noisy_image.GetNumberOfComponentsPerPixel()}")
#
# sitk.Show(noisy_image, "Image Display")








