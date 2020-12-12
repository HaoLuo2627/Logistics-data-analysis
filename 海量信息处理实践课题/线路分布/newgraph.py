# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:37:57 2020

@author: John
"""
import pandas as pd
import os
import numpy as np
from geopy import distance

__TEST__ = False

class myPoint:
    def __init__(self, x0, y0):
        self.x = x0
        self.y = y0

# check if two cities border on each other approximately
def checkIfBorderOn(p1, p2, p3, p4):
    if p1.x > p2.x:
        p1.x, p2.x = p2.x, p1.x
    if p1.y < p2.y:
        p1.y, p2.y = p2.y, p1.y
    if p3.x > p4.x:
        p3.x, p4.x = p4.x, p3.x
    if p3.y < p4.y:
        p3.y, p4.y = p4.y, p3.y
    if p3.x > p2.x or p4.x < p1.x or p4.y > p1.y or p3.y < p2.y:
        return False
    else:
        return True


if os.path.exists(r"./citiesGPS.csv"):
    df = pd.read_csv(r"./citiesGPS.csv",\
                                header=0,names=['city', 'GPS', 'boundingbox'],encoding='utf-8',quotechar='"')

d = distance.distance
numCities = df.shape[0]
# ======================================================================================
# Use Floyd Algorithm to calculate shortest path
# if not exists, calculate
if not os.path.exists('newDistance_shortest.npz'):
    # if exists initialization, read from file
    if os.path.exists('newDistance.npz'):
        c = np.load('newDistance.npz')
        W = c['W']
        R = c['R']
    else:
        # initialize matrix
        W = np.full((numCities, numCities), np.inf)
        R = np.full((numCities, numCities), -1)
        np.fill_diagonal(W, 0)

        
        for ptr1 in range(numCities):
            for ptr2 in range(ptr1 + 1,numCities):
                city1 = eval(df.iloc[ptr1, 2])
                city2 = eval(df.iloc[ptr2, 2])
                p1 = myPoint(city1[2], city1[1])
                p2 = myPoint(city1[3], city1[0])
                p3 = myPoint(city2[2], city2[1])
                p4 = myPoint(city2[3], city2[0])
                if checkIfBorderOn(p1, p2, p3, p4):
                    city1GPS = eval(df.iloc[ptr1, 1])
                    city2GPS = eval(df.iloc[ptr2, 1])
                    W[ptr1, ptr2] = d(city1GPS, city2GPS).km
                    W[ptr2, ptr1] = W[ptr1, ptr2]
        
        for j in range(numCities):
            _tmp = W[:, j]
            R[np.isfinite(_tmp), j] = j
        np.fill_diagonal(R, -1)  
        
        np.savez('newDistance.npz', W = W, R = R)
        
    #Floyd Algorithm
    for k in range(numCities):
        for i in range(numCities):
            for j in range(numCities):
                if W[i,k]+W[k,j] < W[i,j]:
                    W[i,j] = W[i,k] + W[k,j]
                    R[i,j] = R[i,k]
    np.savez('newDistance_shortest.npz',W = W, R = R)
else:
    # if exists, read from file
    c = np.load('newDistance_shortest.npz')
    W = c['W']
    R = c['R']
    
# =======================================================================================

# ===========================================================================================
# calculate final result
def findShortestPath(s,t,*args):
    try:
        sIndex = df[df['city'].map(lambda x: x.startswith(s))].index.tolist()[0]
        tIndex = df[df['city'].map(lambda x: x.startswith(t))].index.tolist()[0]
        
        # 查看最短路径的路由  
        if R[sIndex, tIndex] == -1:
            print('city {} to {} unreachable.'.format(s, t))
            return np.inf
        index = sIndex           
        shortest_path = [index]
        while R[index, tIndex] != tIndex:
            index = R[index, tIndex]
            shortest_path.append(index)
        shortest_path.append(tIndex)
        route = '->'.join([df.iloc[x, 0] for x in shortest_path])
        return W[sIndex, tIndex], route
    except:
        print('city {} or {} not found.'.format(s, t))
        return None, ''

# ================================================================================

# test
if __TEST__:
    citys = '广东,广州'
    cityt = '河南,郑州'
    
    dis, route = findShortestPath(citys, cityt, True)
    print('最短路线长度：{} km'.format(dis))
    print('路由：' + route)