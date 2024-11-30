import os
import torch
import numpy as np
from PIL import Image
from uuid import uuid4
import cv2
from matplotlib import pyplot as plt

def visualize_attention(image, attn_weights, text_description, save_dir="attention_maps", is_show=False):
    """
    可视化注意力图：将注意力权重与图像结合展示，并将结果保存为图像文件。
    - image: 输入的原始图像 (torch.Tensor, PIL.Image 或 numpy.ndarray)
    - attn_weights: 模型输出的注意力权重 (torch.Tensor)
    - text_description: 输入的文本描述，用于生成文件名
    - save_dir: 保存可视化图像的目录
    - is_show: 是否显示图像
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C)
    elif isinstance(image, Image.Image):
        image = np.array(image)

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # 处理注意力权重
    attn_weights = attn_weights.squeeze().cpu().numpy()  # 转换为 numpy

    # 如果 attn_weights 的维度不是 2D，取平均值以减少维度
    if attn_weights.ndim > 2:
        attn_weights = np.mean(attn_weights, axis=0)
    
    print(f"attn_weights shape before resize: {attn_weights.shape}")  # 确保是 2D

    # 将注意力权重调整到与图像相同的大小
    attn_weights = cv2.resize(attn_weights, (image.shape[1], image.shape[0]))

    # 对注意力权重进行归一化处理到[0, 1]区间
    attn_weights = cv2.normalize(attn_weights, None, 0, 1, cv2.NORM_MINMAX)

    # 将注意力权重转换为颜色映射
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_weights), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0

    # 将热图与原始图像叠加
    overlay = np.float32(image) / 255.0
    overlay = cv2.addWeighted(overlay, 0.7, heatmap, 0.3, 0)

    # 生成文件名（根据文本描述）
    print(text_description)
    file_name = f"{str(uuid4())}.png"
    file_path = os.path.join(save_dir, file_name)

    if is_show:
        # 显示图像和热图
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title(f"Attention map for: {text_description}")
        plt.axis('off')
        plt.show()

    # 保存可视化结果
    plt.imsave(file_path, overlay)
    plt.close()  # 关闭图像，释放内存
    print(f"Attention map saved as {file_path}")

if __name__ == '__main__':
    import pickle

    # 加载保存的数据
    with open('saved_data0.pkl', 'rb') as f:
        saved_data = pickle.load(f)

    image = saved_data['image']
    attn_weights = saved_data['attn_weights']
    instruction_text = saved_data['instruction_text']

    # 调用 visualize_attention 函数
    visualize_attention(image, attn_weights, instruction_text, save_dir="attention_maps", is_show=True)

# import os
# from uuid import uuid4

# import cv2
# from matplotlib import pyplot as plt
# import numpy as np

# def visualize_attention(image, attn_weights, text_description, save_dir="attention_maps", is_show=False):
#     """
#     可视化注意力图：将注意力权重与图像结合展示，并将结果保存为图像文件。
#     - image: 输入的原始图像
#     - attn_weights: 模型输出的注意力权重
#     - text_description: 输入的文本描述，用于生成文件名
#     - save_dir: 保存可视化图像的目录
#     - is_show: 是否显示图像
#     """
#     # 创建保存目录（如果不存在）
#     os.makedirs(save_dir, exist_ok=True)

#     attn_weights = attn_weights.squeeze().cpu().numpy()  # (batch, img_len, text_len)
#     attn_weights = np.mean(attn_weights, axis=-1)  # 对文本维度取均值，得到每个图像区域的平均注意力

#     # 对注意力权重进行归一化处理到[0, 1]区间
#     attn_weights = cv2.normalize(attn_weights, None, 0, 1, cv2.NORM_MINMAX)

#     # 将图像和注意力权重合并
#     heatmap = cv2.applyColorMap(np.uint8(255 * attn_weights), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255.0

#     # 将热图与原始图像叠加
#     overlay = np.float32(image) / 255.0
#     overlay = cv2.addWeighted(overlay, 0.7, heatmap, 0.3, 0)

#     # 生成文件名（根据文本描述）
#     file_name = f"{text_description.replace(' ', '_')}_{str(uuid4())}.png"
#     file_path = os.path.join(save_dir, file_name)

#     if is_show:
#         # 显示图像和热图
#         plt.figure(figsize=(10, 10))
#         plt.imshow(overlay)
#         plt.title(f"Attention map for: {text_description}")
#         plt.axis('off')
#         plt.show()

#     # 保存可视化结果
#     plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
#     plt.close()  # 关闭图像，释放内存
#     print(f"Attention map saved as {file_path}")


# # # 可视化函数（与之前相同）
# # def visualize_attention(image, attn_weights, text_description):
# #     """
# #     可视化注意力图：将注意力权重与图像结合展示
# #     - image: 输入的原始图像
# #     - attn_weights: 模型输出的注意力权重
# #     - text_description: 输入的文本描述，用于标题
# #     """
# #     attn_weights = attn_weights.squeeze().cpu().numpy()  # (batch, img_len, text_len)
# #     attn_weights = np.mean(attn_weights, axis=-1)  # 对文本维度取均值，得到每个图像区域的平均注意力

# #     # 对注意力权重进行归一化处理到[0, 1]区间
# #     attn_weights = cv2.normalize(attn_weights, None, 0, 1, cv2.NORM_MINMAX)

# #     # 将图像和注意力权重合并
# #     heatmap = cv2.applyColorMap(np.uint8(255 * attn_weights), cv2.COLORMAP_JET)
# #     heatmap = np.float32(heatmap) / 255.0

# #     # 将热图与原始图像叠加
# #     overlay = np.float32(image) / 255.0
# #     overlay = cv2.addWeighted(overlay, 0.7, heatmap, 0.3, 0)

# #     # 显示图像和热图
# #     plt.figure(figsize=(10, 10))
# #     plt.imshow(overlay)
# #     plt.title(f"Attention map for: {text_description}")
# #     plt.axis('off')
# #     plt.show()