# -*- coding: utf-8 -*-
# time: 2022/10/12 21:45
# file: EuclidClass_run.py
# author: Euclid_Jie
from EuclidClass import EuclidClass

# 文件存储,
subPathList1 = ['data/1. 主数据集_25k/df_train_b1.csv',
                'data/1. 主数据集_25k/df_test_b1.csv']
subPathList2 = ['data/2. 细分数据集_25k/df_basic_b1.csv',
                'data/2. 细分数据集_25k/df_corp_b1.csv',
                'data/2. 细分数据集_25k/df_judicial_b1.csv',
                'data/2. 细分数据集_25k/df_loan1_b1.csv',
                'data/2. 细分数据集_25k/df_loan2_b1.csv',
                'data/2. 细分数据集_25k/df_query_b1.csv']
subPathList3 = ['data/3. 其他产品数据_10k/df_train_b2.csv']
# 通过文件路径List的拼接可实现导入指定数据，默认全部导入
PathList = subPathList1+subPathList2+subPathList3

# 初始化对象
test = EuclidClass(PathList)
# 调用方法
test.main_run()
