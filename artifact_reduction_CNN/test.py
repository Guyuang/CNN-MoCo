import SimpleITK as sitk

# 替换此路径为您的实际文件路径
file_path = 'E:\Downloads\MonteCarloDatasets\MonteCarloDatasets\Training\P1\MC_T_P1_NS\FDKRecon\FDK4D_01.mha'

# 使用SimpleITK读取.mha文件
image = sitk.ReadImage(file_path)

print(f"Image size: {image.GetSize()}")
print(f"Image spacing: {image.GetSpacing()}")
print(f"Image depth: {image.GetDepth()}")
print(f"Number of components per pixel: {image.GetNumberOfComponentsPerPixel()}")
image_np = sitk.GetArrayFromImage(image)
# 显示图像，需要安装有适合的图像查看器
sitk.Show(image, "Image Display")

# 如果您只想读取图像而不显示，可以在此处添加您的处理代码
