import cv2
import os
import numpy as np

# # 指定图片目录
directory1 = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/loveda/ann_dir/val_process"

# directory2 = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/samrs/samrs_sota/labels/"
# directory3 = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/samrs/samrs_sior/val_labels/"
# directory4 = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/samrs/samrs_sior/labels/"
# directory5 = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/samrs/samrs_sior/val_labels/"
# directory6 = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/samrs/samrs_fast/labels/val"
# directory7 = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/samrs/samrs_fast/labels/train"
# directory8 = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/potsdam/ann_dir/val"
# directory9 = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/potsdam/ann_dir/train"

# 遍历目录中的所有图片
# for directory in directory5: #[directory1, directory2, directory3, directory4, directory5, directory6, directory7, directory8, directory9]:
directory = directory1
for filename in os.listdir(directory):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        # 构建完整路径
        file_path = os.path.join(directory, filename)
        
        # 读取图像
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        # 检查读取是否成功
        if 255 in np.unique(np.array(image)):
            print(1111111111111111111)
        # print(np.unique(np.array(image)))
        # exit()
        updated_image = image - 1

        # save_file_path = os.path.join(directory,  filename)
        # os.mkdir(save_file_path, exist_OK = True)
        # 保存更新后的图像（覆盖原始图像或存储到新的位置）

        

        save_path = "/gruntdata/rs_nas/workspace/xingsu.zq/CATSEG_skysense_refactored/datasets/loveda/ann_dir/val"
        save_path = os.path.join(save_path, filename)
        # cv2.imwrite(save_path, updated_image)

print(f'{directory} finish')
