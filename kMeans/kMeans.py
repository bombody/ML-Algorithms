#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/7/31 下午 01:42
# @Author  : YuXin Chen

import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

class kMeans(object):
    def __init__(self, n_clusters=10, initCent='random', max_iter=300):
        if hasattr(initCent, '__array__'):
            n_clusters = initCent.shape[0]
            self.centroids = np.asarray(initCent, dtype=np.float)
        else:
            self.centroids = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.initCent = initCent
        self.clusterAssment = None
        self.labels = None
        self.sse = None
    # 计算两个向量的欧式距离
    def distEclud(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)

    # 计算两点的曼哈顿距离
    def distManh(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB, ord=1)

    # 为给点的数据集构建一个包含k个随机质心的集合
    def randCent(self, X, k):
        n = X.shape[1]  # 特征维数，也就是数据集有多少列
        centroids = np.empty((k, n))  # k*n的矩阵，用于存储每簇的质心
        for j in range(n):  # 产生质心，一维一维地随机初始化
            minJ = min(X[:, j])
            rangeJ = float(max(X[:, j]) - minJ)
            centroids[:, j] = (minJ + rangeJ * np.random.rand(k, 1)).flatten()
        return centroids

    def fit(self, X):
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
         