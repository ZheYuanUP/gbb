from PIL import Image
import os
from typing import List

def images_to_gif(
    image_paths: List[str],
    output_gif_path: str = "output_animation.gif",
    duration: int = 500,
    loop: int = 0,
    resize: tuple = None
) -> None:
    """
    将多张图片合并为GIF动图

    参数说明：
    - image_paths: 图片文件路径列表（按GIF播放顺序排列）
    - output_gif_path: 输出GIF的文件路径（默认：output_animation.gif）
    - duration: 每张图片的显示时长（毫秒，默认：500ms）
    - loop: GIF循环次数（0表示无限循环，默认：0；大于0表示指定循环次数）
    - resize: 图片统一尺寸（可选，格式：(宽度, 高度)，默认：None，不调整尺寸）
    """
    # 存储加载并处理后的图片对象
    frame_list = []

    try:
        # 遍历所有图片路径，加载并处理图片
        for img_path in image_paths:
            # 检查图片文件是否存在
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图片文件不存在：{img_path}")
            
            # 打开图片（自动兼容JPG/PNG/BMP等常见格式）
            with Image.open(img_path) as img:
                # 统一图片格式（RGBA转RGB，避免透明通道导致的显示问题）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 可选：统一调整所有图片尺寸
                if resize is not None and isinstance(resize, tuple) and len(resize) == 2:
                    img = img.resize(resize, Image.Resampling.LANCZOS)
                
                # 复制图片对象（避免with块结束后图片被关闭）
                frame = img.copy()
                frame_list.append(frame)

        # 检查是否成功加载至少一张图片
        if len(frame_list) == 0:
            raise ValueError("未加载到任何有效图片，无法生成GIF")

        # 保存为GIF动图（第一张图片作为基础，后续图片作为帧追加）
        frame_list[0].save(
            fp=output_gif_path,
            format='GIF',
            append_images=frame_list[1:],  # 追加剩余帧
            save_all=True,                 # 保存所有帧
            duration=duration,             # 帧间隔时长
            loop=loop                      # 循环次数
        )

        print(f"GIF动图生成成功！保存路径：{os.path.abspath(output_gif_path)}")
        print(f"GIF信息：共{len(frame_list)}帧，每帧显示{duration}ms，循环次数：{loop if loop != 0 else '无限'}")

    except Exception as e:
        print(f"生成GIF失败：{str(e)}")
        # 异常时清理已加载的图片资源
        for frame in frame_list:
            frame.close()

# ------------------- 快速使用示例 -------------------
if __name__ == "__main__":
    # 1. 配置你的图片路径（按播放顺序排列，支持绝对路径/相对路径）
    # 示例：当前目录下的 1.jpg、2.png、3.bmp
    your_image_paths = [
        "images/mask_rgb_4/mask_rgb_4_0/0.png",
        "images/mask_rgb_4/mask_rgb_4_0/49.png",
        "images/mask_rgb_4/mask_rgb_4_0/49.png",
        "images/mask_rgb_4/mask_rgb_4_0/99.png",
        "images/mask_rgb_4/mask_rgb_4_0/149.png",
        "images/mask_rgb_4/mask_rgb_4_0/199.png",
        "images/mask_rgb_4/mask_rgb_4_0/499.png",
        "images/mask_rgb_4/mask_rgb_4_0/rgb8.png",
        "images/mask_rgb_4/mask_rgb_4_0/ref.png",
        "images/mask_rgb_4/mask_rgb_4_0/anomaly_map.jpg"
    ]

    # 2. 调用函数生成GIF
    images_to_gif(
        image_paths=your_image_paths,
        output_gif_path="my_animation.gif",  # 自定义输出文件名
        duration=300,                       # 每帧显示300毫秒
        loop=0,                             # 无限循环
        resize=(800, 800)                   # 统一调整为600x400尺寸（可选）
    )