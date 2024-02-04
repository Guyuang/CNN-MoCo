import SimpleITK as sitk
import numpy as np
import os

def convert_mha_to_npy(mha_file_path, npy_file_path):
    # 读取MHA文件
    image = sitk.ReadImage(mha_file_path)

    # 将SimpleITK图像转换为NumPy数组
    array = sitk.GetArrayFromImage(image)

    # 保存NumPy数组到NPY文件
    np.save(npy_file_path, array)

def convert_all_mha_in_folder(mha_folder, npy_folder):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(mha_folder):
        if filename.endswith('.mha'):
            # 构建完整的文件路径
            mha_file_path = os.path.join(mha_folder, filename)
            npy_file_path = os.path.join(npy_folder, filename.replace('.mha', '.npy'))

            # 转换文件
            convert_mha_to_npy(mha_file_path, npy_file_path)
            print(f"Converted {mha_file_path} to {npy_file_path}")


def npy_to_mha(npy_file_path, mha_file_path):
    # Load the numpy array from the NPY file
    array = np.load(npy_file_path)

    # Convert the numpy array back to a SimpleITK image
    image = sitk.GetImageFromArray(array)

    # Write the image to an MHA file
    sitk.WriteImage(image, mha_file_path)

def convert_all_npy_in_folder(npy_folder, mha_folder):
    # Iterate through all files in the folder
    for filename in os.listdir(npy_folder):
        if filename.endswith('.npy'):
            # Build the full file paths
            npy_file_path = os.path.join(npy_folder, filename)
            mha_file_path = os.path.join(mha_folder, filename.replace('.npy', '.mha'))

            # Convert the file
            npy_to_mha(npy_file_path, mha_file_path)
            print(f"Converted {npy_file_path} to {mha_file_path}")

# Example usage
npy_folder = 'E:\Documents\GitHub\CNN-MoCo/artifact_reduction_CNN\data\SPARE_test_npy_3D\V3'  # Change to your NPY folder path
mha_folder = 'E:\Documents\GitHub\CNN-MoCo/artifact_reduction_CNN\data\SPARE_test_npy_3D\V3'  # Folder path where you want to save MHA files
convert_all_npy_in_folder(npy_folder, mha_folder)


# # 使用示例
# mha_folder = 'E:\Documents\GitHub\CNN-MoCo/artifact_reduction_CNN\data\Val_npy_3D\V3'  # 更改为您的MHA文件夹路径
# npy_folder = 'E:\Documents\GitHub\CNN-MoCo/artifact_reduction_CNN\data\Val_npy_3D\V3'  # 您想要保存NPY文件的文件夹路径
# convert_all_mha_in_folder(mha_folder, npy_folder)
