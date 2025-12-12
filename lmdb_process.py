#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-12-27 01:08:30
# @Author  : Dengpan Fu (fdpan@mail.ustc.edu.cn)

import os
import numpy as np
import cv2
import lmdb
import pickle
import json
output_dir = '/hy-tmp/LUPerson/lup_lmdb'
os.makedirs(output_dir, exist_ok=True)
json_file = "/root/SOLIDER-REID-PRO/data/luperson/merged_data_simple_normalizer.json"
lmdb_dir = "/hy-tmp/LUPerson/LUPerson/LUPerson/lmdb"

base_dir = 'luperson'
# lmdb_dir = os.path.join('.', 'LUPerson', 'lmdb')
def load_json_keys(json_file):
    """从JSON文件中加载所有key"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cleaned_keys = [key.replace('.jpg', '') for key in data.keys()]
        print(f"JSON文件 {json_file} 包含 {len(data)} 个key")
        return set(cleaned_keys)
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return set()
    
keys_file = '/hy-tmp/LUPerson/LUPerson/LUPerson/keys.pkl'
env = lmdb.open(lmdb_dir, readonly=True, lock=False)
# keys = pickle.load(open(keys_file, "rb"))

# 创建输出目录


print("=== 开始对比分析 ===")

# 1. 加载JSON文件的key
json_keys = load_json_keys(json_file)
for key in json_keys:
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    
    if buf is None:  # 添加安全检查
        print(f"Warning: Key '{key}' 不存在")
        continue
        
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    im = cv2.imdecode(img_flat, 1)
    
    if im is not None:
        # 使用key作为文件名，替换不合法的字符
        safe_key = key.replace('/', '_').replace('\\', '_')
        output_path = os.path.join(output_dir, f'{safe_key}.jpg')
        cv2.imwrite(output_path, im)
        # print(f"保存: {output_path}")
    else:
        print(f"图片解码失败: {key}")

