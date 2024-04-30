from PIL import Image
import numpy as np

def mse(image1, image2):
    """计算两张图片的均方误差"""
    # 确保两张图片大小相同
    assert image1.size == image2.size, "图片大小不一致"

    # 将图片转换为灰度图像
    image1_gray = image1.convert("L")
    image2_gray = image2.convert("L")

    # 将灰度图像转换为numpy数组
    array1 = np.array(image1_gray)
    array2 = np.array(image2_gray)

    # 计算均方误差
    mse = np.mean((array1 - array2) ** 2)

    return mse

if __name__ == "__main__":
    # 两张图片的路径
    image_path1 = "/workspace/trial_new/undistorted/chunk_0/images/frame_00001.png" 
    image_path2 = "/workspace/trial_new/distorted/chunk_0/images/frame_00001.png" 

    # 加载图片
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # 计算均方误差
    mse_value = mse(image1, image2)
    print("MSE:", mse_value)
