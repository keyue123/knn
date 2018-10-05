#!/usr/bin/python                                                                                                                                                                                                                    
#coding=utf-8

#File Name: test.py
#Author   : john
#Mail     : john.y.ke@mail.foxconn.com 
#Created Time: Sat 01 Sep 2018 05:38:56 PM CST

from numpy import *

def classify(input, dataSet, label, k):
    dataSize = dataSet.shape[0]
    diff = tile(input, (dataSize, 1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff, axis=1)  #行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5

    sortedDistIndex = argsort(dist)  ##argsort()根据元素的值从大到小对元素进行排序，返回下标

    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1#对选取的K个样本所属的类别个数进行统计

    #选取出现的类别次数最多的类别
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))#创建矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0: 3]
        classLabelVector.append(int(listFromLine[-1]))#获取结果
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)#返回矩阵每一列的最小值
    maxVals = dataSet.max(0)#返回矩阵每一列的最大值
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

if __name__ == '__main__':
    resultList = [u'不喜欢', u'魅力一般', u'极具魅力']
    percentTats = float(input(u"玩视频游戏所耗时间百分比："))
    ffMiles = float(input(u"每年获得的飞行常客里程数："))
    iceCream = float(input(u"每周消费的冰淇淋公升数："))
    datingDataMat, datingLabels = file2matrix('C:\\Users\\John\\Desktop\\DataBooks\\MachineLearning\\Ch02\\datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("对此人感觉: ", resultList[classifierResult - 1])