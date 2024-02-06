from utils.basic_blocks import *
from utils.optimizer import Optimizer
from utils.metrics import structural_similarity, peak_snr
from configuration.configuration import Configuration
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests


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

    def get_noisy_image(self,x_start, t):
        # add noise
        x_noisy = self.q_sample(x_start, t=t)

        # turn back into PIL image
        noisy_image = self.reverse_transform(tf.squeeze(x_noisy))

        return noisy_image



# Load configuration files
dataset_path = 'E:/Documents/GitHub/CNN-MoCo/artifact_reduction_CNN/experiments/'
experiments_path = 'E:/Documents/GitHub/CNN-MoCo/artifact_reduction_CNN/data/'
config_file = 'E:/Documents/GitHub/CNN-MoCo/artifact_reduction_CNN/demo_config.py'
configuration = Configuration(config_file, dataset_path, experiments_path)
cf = configuration.load()


forward_process = Forward(cf)

image_path = r'E:\Desktop\output_cats_verify.png'
image = Image.open(image_path)


# 将图像转换为适合处理的格式
image = np.array(image)
image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0  # 确保图像数据在[0, 1]范围
# 假设你的 get_noisy_image 方法已经返回一个适合显示的张量
noisy_image_tensor = forward_process.get_noisy_image(image, tf.constant([100]))

# 由于 noisy_image_tensor 可能不再是 uint8 类型，我们需要将其转换回 [0, 1] 范围的浮点数以便显示
noisy_image_tensor = tf.cast(noisy_image_tensor, dtype=tf.float32)  # 确保为 float 类型
noisy_image_tensor = (noisy_image_tensor - tf.reduce_min(noisy_image_tensor)) / (tf.reduce_max(noisy_image_tensor) - tf.reduce_min(noisy_image_tensor))

# 使用 matplotlib 显示加噪声后的图像
plt.imshow(noisy_image_tensor.numpy())
plt.show()








