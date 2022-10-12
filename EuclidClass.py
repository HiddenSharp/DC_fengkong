# -*- coding: utf-8 -*-
# time: 2022/10/12 21:45
# file: EuclidClass.py
# author: Euclid_Jie

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class EuclidClass(object):
    def __init__(self, dataPathList):
        self.dataPathLish = dataPathList

    def get_data(self):
        """
        读取数据的函数
        :return:
        """
        # 导入主数据集
        self.mainDataTrian = pd.read_csv(self.dataPathLish[0])
        self.mainDataTest = pd.read_csv(self.dataPathLish[1])
        print("已导入数据，主训练集数据维度为{}，主测试集维度为{}".format(self.mainDataTrian.shape, self.mainDataTest.shape))
        try:
            # 导入细分数据集
            self.basci = pd.read_csv(self.dataPathLish[2])
            self.crop = pd.read_csv(self.dataPathLish[3])
            self.judicial = pd.read_csv(self.dataPathLish[4])
            self.loan1 = pd.read_csv(self.dataPathLish[5])
            self.loan2 = pd.read_csv(self.dataPathLish[6])
            self.query = pd.read_csv(self.dataPathLish[7])
            print("已导入细分数据集")
        except:
            print("本次不导入细分数据集")
        try:
            # 导入其他产品数据集
            self.train2 = pd.read_csv(self.dataPathLish[8])
            print("已导入其他产品数据集")
        except:
            print("本次不导入其他产品数据集")

    def _merge_tarin_test(self, dataDf, on=None, how=None):
        # 用于同时merge训练集和测试集

        self.mainDataTrian = self.mainDataTrian.merge(dataDf, on=on, how=how)
        self.mainDataTest = self.mainDataTest.merge(dataDf, on=on, how=how)
        print("合并数据完成，主训练集数据维度为{}，主测试集维度为{}".format(self.mainDataTrian.shape, self.mainDataTest.shape))

    def pre_process(self):
        """
        数据预处理
        :return:
        """
        # 尝试将细分数据拼接到主数据上，显然存在列名冲突
        ## 此部分Id无重复
        for dataDf in [self.basci, self.crop, self.judicial]:
            self._merge_tarin_test(dataDf, on='cust_id', how='left')
        ## 此部分Id重复
        # TODO 可考虑这部分数据如何处理，目前先不merge
        for dataDf in [self.loan1, self.loan2, self.query]:
            pass

    def modle_build(self):
        """
        创建模型
        :return:
        """
        pass

    def saveData(self, dataDf=None, filename=None):
        """
        输出数据，默认输出主数据集
        :return:
        """
        self.mainDataTrian.to_csv('Euclid输出/mainDataTrian.csv', index=False)
        self.mainDataTest.to_csv('Euclid输出/mainDataTest.csv', index=False)
        # 如果指定了某个数据，会输出该数据为csv文件
        try:
            dataDf.to_csv('Euclid输出/' + filename + ',csv', index=False)
        except:
            pass

    def main_run(self):
        """
        主函数
        :return:
        """
        self.get_data()
        self.pre_process()
        self.modle_build()
        self.saveData()
        print("all process done!")

        pass
