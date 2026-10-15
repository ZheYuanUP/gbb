import cv2
import numpy as np

def gaussian_blur_opencv(input_path, output_path, kernel_size=(5,5), sigmaX=0):
    """
    :param input_path: 输入图片路径（如'./input.jpg'）
    :param output_path: 模糊后图片保存路径（如'./blur_output.jpg'）
    :param kernel_size: 高斯核大小，越大模糊效果越强
    :param sigmaX: X方向高斯核标准差
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法读取图片，请检查路径：{input_path}")
    
    blurred_img = cv2.GaussianBlur(img, kernel_size, sigmaX)
    
    cv2.imwrite(output_path, blurred_img)
    print(f"高斯模糊完成，结果已保存至：{output_path}")

# 示例调用
if __name__ == "__main__":
    input_img = "/home/jojo/pythonproject/MetaUAS-main/images/20251022/3-2.png"
    output_img = "/home/jojo/pythonproject/MetaUAS-main/images/20251022/3-2-1.png"
    gaussian_blur_opencv(input_img, output_img, kernel_size=(35,35), sigmaX=5)