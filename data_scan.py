# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:33:19 2024

@author: ypwang
"""

import os
import glob
import shutil
import random
from PIL import Image
def get_BD_paths(base_path):
    #取得所有BD路徑

    
    BD_paths = []
    
    batch_path = os.listdir(base_path)
    for batch_path_ in batch_path:
        if not('ai'in batch_path_):continue #確認是人工或AI快評
        full_batch_path_ = os.path.join(base_path, batch_path_)
        if not(os.path.isdir(full_batch_path_)): continue
        user_path = os.listdir(full_batch_path_)
        
        for user_path_ in user_path:
            full_user_path_ = os.path.join(base_path, batch_path_, user_path_)
            if not(os.path.isdir(full_user_path_)): continue
            BD_path = os.listdir(full_user_path_)
            
            for BD_path_ in BD_path:
                
                full_BD_path_ = os.path.join(base_path, batch_path_, user_path_, BD_path_)
                if not(os.path.isdir(full_BD_path_)): continue
                if 'wall_classification' in os.listdir(full_BD_path_):
                    full_BD_path_ = os.path.join(full_BD_path_, 'wall_classification')
                    BD_paths.append(full_BD_path_)
    return BD_paths

def get_image_paths_sided(BD_paths):
    # 定義各個子資料夾的路徑
    categories = {
        'non': ['x_none', 'y_none'],
        'W3': ['x_open_Brick', 'x_open_RC', 'y_open_Brick', 'y_open_RC'],
        'W4': ['x_cont_Brick', 'x_cont_RC', 'y_cont_Brick', 'y_cont_RC']
    }

    # 儲存各個分類資料夾內影像路徑的字典
    image_paths = {
        'non': [],
        'W3': [],
        'W4': []
    }

    # 遍歷每個分類資料夾並收集影像路徑
    for BD_path in BD_paths:
        for category, subfolders in categories.items():
            for subfolder in subfolders:
                folder_path = os.path.join(BD_path, subfolder)
                image_paths[category].extend(glob.glob(os.path.join(folder_path, '*.jpg')))

    return image_paths

def get_image_paths_meta(BD_paths):
    # 定義各個子資料夾的路徑
    categories = {
        'Brick': ['x_open_Brick', 'x_cont_Brick', 'y_open_Brick', 'y_cont_Brick'],
        'RC': ['x_open_RC', 'x_cont_RC', 'y_open_RC', 'y_cont_RC']
    }

    # 儲存各個分類資料夾內影像路徑的字典
    image_paths = {
        'Brick': [],
        'RC': []
    }

    # 遍歷每個分類資料夾並收集影像路徑
    for BD_path in BD_paths:
        for category, subfolders in categories.items():
            for subfolder in subfolders:
                folder_path = os.path.join(BD_path, subfolder)
                image_paths[category].extend(glob.glob(os.path.join(folder_path, '*.jpg')))

    return image_paths

base_path = 'D:\作業繳交'
BD_paths = get_BD_paths(base_path)

random.seed(2306008)
random.shuffle(BD_paths)
split_num = int(len(BD_paths)*0.85)

train_BD_paths = BD_paths[:split_num]
test_BD_paths = BD_paths[split_num:]

#需注意切割後的各個分類數量是否合理
train_image_paths = get_image_paths_sided(train_BD_paths)
test_image_paths = get_image_paths_sided(test_BD_paths)
    
s = \
'''Train
建築物數量: {BD_num}
四面圍束牆: {W4}
三面圍束牆: {W3}
無有效牆: {non}
'''.format(BD_num=len(train_BD_paths), W4=len(train_image_paths['W4']), W3=len(train_image_paths['W3']), non=len(train_image_paths['non']))
print(s)


s = \
'''Test
建築物數量: {BD_num}
四面圍束牆: {W4}
三面圍束牆: {W3}
無有效牆: {non}
'''.format(BD_num=len(test_BD_paths), W4=len(test_image_paths['W4']), W3=len(test_image_paths['W3']), non=len(test_image_paths['non']))
print(s)


def copy2data(cls_, src, dst, file_name):
    # file_name = os.path.basename(src)
    dst_path = os.path.join(dst, cls_, file_name)
    print(src,dst_path)
    # Open the image file
    with Image.open(src) as img:
        # If the height is greater than the width, rotate the image 90 degrees
        if img.height > img.width:
            img = img.rotate(90, expand=True)
        img.save(dst_path)
    # shutil.copyfile(src, dst_path)
    



for cls_, files_path in train_image_paths.items():
    for ind, file_path in enumerate(files_path): 
        file_name = cls_ + "_" + str(ind) + '.jpg'
        copy2data(cls_, file_path, 'D:/project/Seismic Rapid Evaluation/Wall Model Dev/Data/Type/train', file_name)

for cls_, files_path in test_image_paths.items():
    for ind, file_path in enumerate(files_path): 
        file_name = cls_ + "_" + str(ind) + '.jpg'
        copy2data(cls_, file_path, 'D:/project/Seismic Rapid Evaluation/Wall Model Dev/Data/Type/test', file_name)


# #需注意切割後的各個分類數量是否合理
# train_image_paths = get_image_paths_meta(train_BD_paths)
# test_image_paths = get_image_paths_meta(test_BD_paths)
    
# s = \
# '''Train
# 建築物數量: {BD_num}
# 磚牆: {Brick}
# RC牆: {RC}
# '''.format(BD_num=len(train_BD_paths), Brick=len(train_image_paths['Brick']), RC=len(train_image_paths['RC']))
# print(s)


# s = \
# '''Test
# 建築物數量: {BD_num}
# 磚牆: {Brick}
# RC牆: {RC}
# '''.format(BD_num=len(test_BD_paths), Brick=len(test_image_paths['Brick']), RC=len(test_image_paths['RC']))
# print(s)
