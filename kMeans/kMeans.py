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
            except:
                raise TypeError("numpy.ndarray required for X")
        m = X.shape[0]  # 样本数量
        self.clusterAssment = np.empty((m, 2))  # m*2的矩阵，第一列表示样本属于哪一簇，第二列存储该样本与质心的平方误差(Squared Error,SE)
        if self.initCent == 'random':   # 可以指定质心或者随机产生质心
            self.centroids = self.randCent(X, self.n_clusters)
        clusterChanged = True
        for _ in range(self.max_iter):# 指定最大迭代次数
            clusterChanged = False
            for i in range(m):  # 将每个样本分配到离它最近的质心所属的簇
                minDist = np.inf
                minIndex = -1
                for j in range(self.n_clusters):    #遍历所有数据点找到距离每个点最近的质心
                    distJI = self.distEclud(self.centroids[j, :], X[i, :])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                if self.clusterAssment[i, 0] != minIndex:
                    clusterChanged = True
                    self.clusterAssment[i, :] = minIndex, minDist ** 2
            if not clusterChanged:  # 若所有样本点所属的簇都不改变,则已收敛，提前结束迭代
                break
            for i in range(self.n_clusters):  # 将每个簇中的点的均值作为质心
                ptsInClust = X[np.nonzero(self.clusterAssment[:, 0] == i)[0]]  # 取出属于第i个族的所有点
                if(len(ptsInClust) != 0):
                    self.centroids[i, :] = np.mean(ptsInClust, axis=0)

        self.labels = self.clusterAssment[:, 0]
        self.sse = sum(self.clusterAssment[:, 1])   # Sum of Squared Error,SSE

class biKMeans(object):
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.centroids = None
        self.clusterAssment = None
        self.labels = None
        self.sse = None
    # 计算两点的欧式距离
    def distEclud(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)

    # 计算两点的曼哈顿距离
    def distManh(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB,ord = 1)
    def fit(self, X):
        m = X.shape[0]
        self.clusterAssment = np.zeros((m, 2))
        if(len(X) != 0):
            centroid0 = np.mean(X, axis=0).tolist()
        centList = [centroid0]
        for j in range(m):  # 计算每个样本点与质心之间初始的SE
            self.clusterAssment[j, 1] = self.distEclud(np.asarray(centroid0), X[j, :]) ** 2

        while (len(centList) < self.n_clusters):
            lowestSSE = np.inf
            for i in range(len(centList)):  # 尝试划分每一族,选取使得误差最小的那个族进行划分
                ptsInCurrCluster = X[np.nonzero(self.clusterAssment[:, 0] == i)[0], :]
                clf = kMeans(n_clusters=2)
                clf.fit(ptsInCurrCluster)
                centroidMat, splitClustAss = clf.centroids, clf.clusterAssment  # 划分该族后，所得到的质心、分配结果及误差矩阵
                sseSplit = sum(splitClustAss[:, 1])
                sseNotSplit = sum(self.clusterAssment[np.nonzero(self.clusterAssment[:, 0] != i)[0], 1])
                if (sseSplit + sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitClustAss.copy()
                    lowestSSE = sseSplit + sseNotSplit
            # 该族被划分成两个子族后,其中一个子族的索引变为原族的索引，另一个子族的索引变为len(centList),然后存入centList
            bestClustAss[np.nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centList)
            bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentToSplit
            centList[bestCentToSplit] = bestNewCents[0, :].tolist()
            centList.append(bestNewCents[1, :].tolist())
            self.clusterAssment[np.nonzero(self.clusterAssment[:, 0] == bestCentToSplit)[0], :] = bestClustAss
        self.labels = self.clusterAssment[:, 0]
        self.sse = sum(self.clusterAs