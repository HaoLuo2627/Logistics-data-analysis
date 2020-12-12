# -*- coding: utf-8 -*-
import pymysql
import pandas as pd
import numpy as np
import os
from graph import findShortestPath
import math
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rc('font', family='SimHei', weight='bold')
plt.rcParams['axes.unicode_minus'] = False

def initClientEncode(conn):
    '''mysql client encoding=utf8'''
    curs = conn.cursor()
    curs.execute("SET NAMES utf8")
    conn.commit()
    return curs


def databaseQuery(customer):
    strHost = 'localhost'
    strDB = 'logistics'
    strUser = 'root'
    strPasswd = 'luoh990618'
    db = pymysql.connect(host=strHost, db=strDB, user=strUser, passwd=strPasswd, charset="utf8")
    curs = initClientEncode(db)
    selectSequence = 'SELECT type, province, city FROM address_{} WHERE yundanID = {};'
    selectYunDanID = 'SELECT DISTINCT ID FROM yundan_{} ORDER BY ID ASC;'.format(customer)
    try:
        curs.execute(selectYunDanID)
        yundanIDs = curs.fetchall()
        track = [ [yundanID[0], tuple() ] for yundanID in yundanIDs]
        for i in range(len(yundanIDs)):
            curs.execute(selectSequence.format(customer, yundanIDs[i][0]))
            qqq = [(pair[0],','.join(pair[-2:])) for pair in curs.fetchall()]
            qq = list(set(qqq))
            qq.sort(key=qqq.index)
            track[i][1] = tuple(qq)          
    except Exception as e:
        print(e.args)
    finally:
        db.close()
    return track


def parse_cities(cities):
    if type(cities) is not dict:
        raise TypeError('cities should be a dict.')
    c = [('提货', x) for x in cities['提货']]
    c += [('卸货', x) for x in cities['卸货']]
    return tuple(c)


def trackDistributionStatistics(customer, cities):
    if os.path.exists('track_{}.csv'.format(customer)):
        # Load from local file, save time for db query, can save one minute or so
        dataframe = pd.read_csv('track_{}.csv'.format(customer),\
                                header=None,names=['yundanID','track'],encoding='utf-8',quotechar='"')
        dataframe['track'] = pd.Series(data = [eval(x) for x in dataframe['track']], name='track')
    else:
        yundanTrack = databaseQuery(customer)
        dataframe = pd.DataFrame(data=yundanTrack, columns=['yundanID','track'])
        dataframe.to_csv('track_{}.csv'.format(customer),header=False,index=False,encoding='utf-8',quotechar='"')
    
    # group by track, rank count descending
    trackCount = dataframe['track'].value_counts()
    d = {'track':trackCount.index, 'freq':trackCount.to_numpy()}
    trackCount = pd.DataFrame(d, columns=['track','freq'])
    
    # search history track, get track that passes through expected points
    boolIndex = [False for i in range(trackCount.shape[0])]
    cities_tuple = parse_cities(cities)
    cities_set = set(cities_tuple)
    for i in range(trackCount.shape[0]):
        elem = trackCount.iloc[i, 0]
        if cities_set == (set(elem)):
            boolIndex[i] = True
    trackDistribution = trackCount.loc[boolIndex,:].copy()
    trackDistribution.index = range(trackDistribution.shape[0])
    trackDistribution['percentage'] = trackDistribution['freq'] / trackDistribution['freq'].sum()
    return trackDistribution,trackCount

                        
def FullPermutation(a, begin, end, result):
    if begin >= end:
        result.append(a.copy())
    else:
        for i in range(begin, end):
            if begin != i:
                a[begin], a[i] = a[i], a[begin]
            FullPermutation(a, begin+1, end, result)
            if begin != i:
                a[begin], a[i] = a[i], a[begin]


def suggestion(cities):
    sNodes = cities['提货']
    tNodes = cities['卸货']
    _SourceProvinces = [x.split(',')[0] for x in sNodes]
    _DestinationProvinces = [x.split(',')[0] for x in tNodes]
    SourceProvinces = list(set(_SourceProvinces))
    DestinationProvinces = list(set(_DestinationProvinces))
    numOfSource = len(SourceProvinces)
    numOfDest = len(DestinationProvinces)
    suggestPath = ''
    distanceInTotal = 0
    if numOfSource == 1 and numOfDest == 1:
        distanceInTotal = findShortestPath(SourceProvinces[0], DestinationProvinces[0])
        commonNodes = list(set(sNodes).intersection(set(tNodes)))
        if len(commonNodes) > 0:
            for item in commonNodes:
                sNodes.remove(item)
                tNodes.remove(item)
            sNodes += commonNodes
            tNodes = commonNodes[::-1] + tNodes
        suggestPath = '->'.join([x+'(提货)' for x in sNodes] + [x+'(卸货)' for x in tNodes])
    else:
        # 根据数据的实际情况，提货点都在一个城市，自然在同一省份。卸货点可能为多个不同省的城市
        distanceDict = dict()
        for i in range(numOfDest):
            for j in range(i+1,numOfDest,1):
                key = '-'.join([DestinationProvinces[i],DestinationProvinces[j]])
                if key in distanceDict.keys() or '-'.join([DestinationProvinces[j],DestinationProvinces[i]]) in distanceDict.keys():
                    continue
                distanceDict[key] = \
                    findShortestPath(DestinationProvinces[i],DestinationProvinces[j])
        for i in range(numOfSource):
            for j in range(i+1,numOfSource):
                key = '-'.join([SourceProvinces[i],SourceProvinces[j]])
                if key in distanceDict.keys() or '-'.join([SourceProvinces[j],SourceProvinces[i]]) in distanceDict.keys():
                    continue
                distanceDict[key] = \
                    findShortestPath(SourceProvinces[i],SourceProvinces[j])
        for i in range(numOfSource):
            for j in range(numOfDest):
                key = '-'.join([SourceProvinces[i],DestinationProvinces[j]])
                if key in distanceDict.keys() or '-'.join([DestinationProvinces[j],SourceProvinces[i]]) in distanceDict.keys():
                    continue
                distanceDict[key] = \
                    findShortestPath(SourceProvinces[i],DestinationProvinces[j])
        # 对宿按序号全排列
        sourceIndexList = []
        FullPermutation([i for i in range(numOfSource)], 0, numOfSource, sourceIndexList)
        destIndexList = []
        FullPermutation([i for i in range(numOfDest)], 0, numOfDest, destIndexList)
        # 获得全排列后，把源放在序列首位，两两之间求距离
        allTrackDistance = np.zeros((math.factorial(numOfSource)*math.factorial(numOfDest), 1))
        for k in range(math.factorial(numOfSource)):
            for l in range(math.factorial(numOfDest)):
                trackList = []
                trackList += [SourceProvinces[i_] for i_ in sourceIndexList[k]]
                trackList += [DestinationProvinces[i_] for i_ in destIndexList[l]]
                for j_ in range(len(trackList)-1):
                    key = '-'.join(trackList[j_:j_+2])
                    key = key if key in distanceDict.keys() else '-'.join([trackList[j_+1],trackList[j_]])
                    allTrackDistance[k * math.factorial(numOfDest) + l, :] += distanceDict[key]
        # 找出距离最小的路线
        shortestPathIndex = allTrackDistance.argmin()
        k = shortestPathIndex // math.factorial(numOfDest)
        l = shortestPathIndex % math.factorial(numOfDest)
        # 最小的路线是省的序列，结果字符串中将同一个省的城市按顺序连续输出，同时给出线路的近似长度
        # 放在distanceInTotal和suggestPath中
        distanceInTotal = round(allTrackDistance[shortestPathIndex][0],2)
        sourceList = [SourceProvinces[i_] for i_ in sourceIndexList[k]]
        destList = [DestinationProvinces[i_] for i_ in destIndexList[l]]
        pathList = []
        if sourceList[-1] == destList[0]:
            prov = sourceList[-1]
            intersect = set(sNodes).intersection(set(tNodes))
            intersect = [x for x in intersect if x.split(',')[0] == prov]
        else:
            prov = ''
        for sourceProv in sourceList:
            if sourceProv != prov:
                for m in range(len(_SourceProvinces)):
                    if sourceProv == _SourceProvinces[m]:
                        pathList.append(sNodes[m]+'(提货)')
            else:
                for m in range(len(_SourceProvinces)):
                    if sourceProv == _SourceProvinces[m] and sNodes[m] not in intersect:
                        pathList.append(sNodes[m]+'(提货)')
                pathList.extend([x+'(提货)' for x in intersect])
                        
        for destProv in destList:
            if destProv != prov:
                for n in range(len(_DestinationProvinces)):
                    if destProv == _DestinationProvinces[n]:
                        pathList.append(tNodes[n]+'(卸货)')
            else:
                pathList.extend([x+'(卸货)' for x in intersect[::-1]])
                for n in range(len(_DestinationProvinces)):
                    if destProv == _DestinationProvinces[n] and tNodes[n] not in intersect:
                        pathList.append(tNodes[n]+'(卸货)')

        suggestPath = '->'.join(pathList)
    
    return distanceInTotal, suggestPath
        

def plotBar(customer, trackCount, **kwargs):
    trackSummary = trackCount.loc[:, ['track', 'freq']]
    if 'threshold' in kwargs.keys():
        trackSummary = trackSummary.loc[trackSummary['freq'] > kwargs['threshold']]
    if 'limit' in kwargs.keys():
        trackSummary = trackSummary.iloc[0:kwargs['limit'], :]
    if trackSummary.shape[0] == 0:
        print('No record that satisfies required conditions.')
        return
    # 尝试画历史线路频数直方图
    labels = trackSummary['track']
    data = trackSummary['freq']
    fig, ax = plt.subplots()
    y = np.arange(len(labels))
    rects = ax.barh(y, data)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xticks(())
    ax.set_ylabel('路线')
    ax.set_xlabel('频数')
    ax.set_title('客户{}线路分布直方图'.format(customer))
    
    for rect in rects:
        width = rect.get_width()
        ax.annotate('{}'.format(width), 
                    xy = (width, rect.get_y() + rect.get_height()/2),
                    xytext = (3, 0),
                    textcoords = "offset points",
                    ha = 'left', va = 'center')
    fig.tight_layout()
    plt.show()

def suggestRank(score):
    if score >= 0.8:
        return 5
    elif score>= 0.6:
        return 4
    elif score >= 0.4:
        return 3
    elif score >= 0.2:
        return 2
    else:
        return 1


if __name__ == '__main__':
    #==========================================================================
    #客户需求
    customer = 'A'
    cities = {'提货':['陕西,咸阳'], '卸货':['陕西,西安','甘肃,兰州','宁夏,银川']}
    #==========================================================================
    # 模型参数
    alpha = 0.2
    beta = 0.3
    gamma = 1-alpha-beta
    # 查数据库，得到线路历史分布
    trackDistribution,trackCount = trackDistributionStatistics(customer,cities)
    # 基于图论优化，找出经过这些点的最短路
    distanceInTotal, suggestPath = suggestion(cities)
    print('图论算法推荐路线:\n'+suggestPath)
    #=========================================================================
    # 专家评议法
    if trackDistribution.shape[0] > 0:
        for idx in range(trackDistribution.shape[0]):
            tmp = trackDistribution.iloc[idx, 0]
            trackStr = '->'.join([x[1]+'('+x[0]+')' for x in tmp])
            trackDistribution.iloc[idx, 0] = trackStr
        print('历史上符合查询要求的运单的路线和选择这条路线的人数')
        print(trackDistribution)
        trackDistribution['score'] = trackDistribution['percentage'] * beta
        trackDistribution.loc[trackDistribution['freq'] >= 10, 'score'] += alpha
        shortestTrack = trackDistribution['track'] == suggestPath
        if shortestTrack.max() == True:
            trackDistribution.loc[shortestTrack, 'score'] += gamma
        else:
            trackDistribution = trackDistribution.append(\
                {'track':suggestPath,'freq':0,'percentage':0,'score':gamma}, ignore_index=True)
        print('系统推荐路线：{}'.format(trackDistribution.iloc[trackDistribution['score'].argmax(), 0]))
        fen = trackDistribution.iloc[trackDistribution['score'].argmax(),3]
        print('推荐星级：{}'.format(suggestRank(fen)))
    else:
        print('历史上没有符合要求的运单')
        print("系统推荐路线：{}".format(suggestPath))
        print('推荐星级：{}'.format(suggestRank(gamma)))
    print('近似总长度:{} km'.format(distanceInTotal))    
    #==========================================================================
    #==========================================================================
    # 将客户线路分布统计结果存储为本地文件
    trackCount1 = trackCount.copy()
    for i in range(trackCount.shape[0]):
        tmp = trackCount.iloc[i, 0]
        trackStr = []
        for item in tmp:
            trackStr.append( item[1] + '(' + item[0] +')' )
        trackCount1.iloc[i, 0] = '->'.join(trackStr)
    trackCount1['percentage'] = trackCount1['freq'] / trackCount1['freq'].sum()
    if not os.path.exists('trackCount-{}.csv'.format(customer)):
        trackCount1.to_csv('trackCount-{}.csv'.format(customer),\
                           header=['Track','Count','Percentage'],index=False,encoding='UTF-8',quotechar='"')
    #==========================================================================
    # 画线路分布直方图
    plotBar(customer, trackCount1, threshold = 100, limit=20)
   

