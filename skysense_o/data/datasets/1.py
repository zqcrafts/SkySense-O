# from skysense_o import datasets
# import register_skysa_graph
# import os
# from PIL import Image
# import numpy as np

# def count_unique_values_in_images(folder_path):
#     # 遍历文件夹中的所有文件
#     for filename in os.listdir(folder_path):
#         if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 检查文件是否为图片
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 # 打开图片
#                 with Image.open(file_path) as img:
#                     # 将图片转换为numpy数组
#                     img_array = np.array(img)
#                     # 计算唯一值
#                     unique_values = np.unique(img_array)
#                     print(f"Image: {filename}, Unique Values: {unique_values}")
#             except Exception as e:
#                 print(f"Error processing {filename}: {e}")

# # 使用示例
# folder_path = '/gruntdata/rs_nas/workspace/xingsu.zq/SkySA_DataEngine/dataset/sub_dataset/rs5m_v2_5k/ann_dir'
# count_unique_values_in_images(folder_path)