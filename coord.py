import pandas as pd
import numpy as np
from utils import *
# 示例 coor_df 数据
# 假设数据如下
section_id = "151676"
adata, image_path = get_sectionData(section_id)


coor_df = pd.DataFrame(adata.obsm["coord"])
r = 2


def polar_to_cartesian(r, theta, origin):
    """将极坐标转换为笛卡尔坐标"""
    x = origin[1] + r * np.cos(theta)
    y = origin[0] + r * np.sin(theta)
    return (y, x)


# 平移方向和角度
translations = {
    'Spatial Transcriptomics': [
        (r / 2, 0),
        (np.sqrt(2) * r / 2, np.pi / 4),
        (r / 2, np.pi / 2)
    ],
    '10x Visium': [
        (np.sqrt(3) * r / 3, np.pi / 6),
        (np.sqrt(3) * r / 3, np.pi / 2),
        (np.sqrt(3) * r / 3, 5 * np.pi / 6)
    ]
}

# 原始坐标
original_coords = coor_df.values

# 存储新的未测量点
new_coords = []

# 选择数据类型：'Spatial Transcriptomics' 或 '10x Visium'
data_type = 'Spatial Transcriptomics'  # 这里可以根据实际需求进行更改

for origin in original_coords:
    for translation in translations[data_type]:
        new_coords.append(polar_to_cartesian(translation[0], translation[1], origin))

# 去重并转换为 DataFrame
new_coords = list(set(new_coords))
new_coords_df = pd.DataFrame(new_coords, columns=['y', 'x'])

# 输出新点的 DataFrame
print(new_coords_df)
