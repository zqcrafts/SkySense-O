import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import matplotlib
import os 
import json


def visual_ann(ann_path, img_path, output_path, class_text):

    class_dict = {}
    for index, category_text in  enumerate(class_text):
        class_dict[index] = category_text

    unique_values = np.unique(ann_path)   
    colormap = matplotlib.colormaps['Set1'] # Set1 tab10
    color_map = {
        value: tuple((np.array(colormap(i / (len(unique_values) ) )[:3]) * 255).astype(int).tolist())
        for i, value in enumerate(unique_values)
    }
    # 创建三通道彩色图像
    color_image = np.zeros((ann_path.shape[0], ann_path.shape[1], 3), dtype=np.uint8)
    for value, color in color_map.items():
        # if value != (num_class-1):  # 筛选出一些类别不显示
        # if value != 15 and value != 8:
        # if value != 0:
        mask = (ann_path == value)
        color_image[mask] = color

    # 可视化和彩色图像标注灰度值数字标签
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

    # 标注颜色区域的原灰度值数字标签和白色底框
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9 # 0.3
    thickness = 2 # 1
    
    only_run_once = True
    
    # 在这里过滤不想要的标签
    for value in unique_values[unique_values!=255]:
        # if value != 15 and value != 7:
        # print(value)
        if value != 0:
            mask = np.where(ann_path == value)
            if mask[0].size > 0:
                y, x = np.median(mask, axis=1).astype(int)  # 寻找每个灰度值的中位数位置

                value_tag = class_dict[value]
                origin_img = cv2.imread(img_path)
                origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

                if only_run_once:  # only run once
                    alpha = 0.4
                    color_image = color_image.astype(np.float32)
                    overlay_image = cv2.addWeighted(color_image, 1 - alpha, origin_img.astype(np.float32), alpha, 0)
                    color_image = overlay_image.astype(np.uint8)
                    only_run_once = False
                # 绘制白色底框
                (text_width, text_height), baseline = cv2.getTextSize(str(value_tag), font, font_scale, thickness)
                cv2.rectangle(
                    color_image, 
                    (x - text_width // 2 - 2 -50, y - text_height // 2 - 2), 
                    (x + text_width // 2 + 2 -50, y + text_height // 2 + 2), 
                    (255, 255, 255), -1
                )
                # 绘制标签，用原来的灰度值
                cv2.putText(
                    color_image, str(value_tag), 
                    (x - text_width // 2 -50, y + text_height // 2), 
                    font, font_scale, color_map[value], thickness, cv2.LINE_AA
                )         
    
    matplotlib.pyplot.close()
    cv2.imwrite(output_path, color_image)


if __name__ == "__main__":

    ann_path = './2.png'
    img_path = '/gruntdata/rs_nas/workspace/wenshuo.ljw/project/CATSEG/datasets/tent_seg/ref_image_crop.jpg'
    save_path = './5.png'
    class_text = '/gruntdata/rs_nas/workspace/wenshuo.ljw/project/CATSEG/datasets/sky5k/sky5k.json'
    visual_ann(ann_path, img_path, save_path, class_text)
