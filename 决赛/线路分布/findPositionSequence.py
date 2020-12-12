# -*- coding: utf-8 -*-
import pymysql
import pandas as pd
import numpy as np
import os
from newgraph import findShortestPath
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

def parseRoute(s,someRoute):
    tmpList = []
    for i in range(len(someRoute)):
        tmp = someRoute[i].split('->')
        if i != 0:
            if tmpList[-1] != tmp[0]:
                tmp = tmp[::-1]
            tmpList += tmp[1:]
        else:
            if tmp[0].startswith(s) == False:
                tmp = tmp[::-1]
            tmpList += tmp
    return '->'.join(tmpList)

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def trackDistributionStatistics(customer, cities, shaoDan=False):
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
    # calculate track hot
    ttt = np.log(trackCount['freq'] + 1)
    ma = ttt.max()
    ttt = (ttt - ma/2)
    trackCount['hot'] = sigmoid(ttt)
    # search history track, get track that passes through expected points
    boolIndex = [False for i in range(trackCount.shape[0])]
    cities_tuple = parse_cities(cities)
    cities_set = set(cities_tuple)
    for i in range(trackCount.shape[0]):
        elem = trackCount.iloc[i, 0]
        #有考虑捎单问题要修改
        # 这么处理对捎单的考虑其实不充分
        if cities_set == (set(elem)) and True != shaoDan:
            boolIndex[i] = True
        elif cities_set.issubset(set(elem)) and True == shaoDan:
            boolIndex[i] = True
    trackDistribution = trackCount.loc[boolIndex,:].copy()
    trackDistribution.index = range(trackDistribution.shape[0])
    trackDistribution['percentage'] = trackDistribution['freq'] / trackDistribution['freq'].sum()
    
    for i in range(trackCount.shape[0]):
        tmp = trackCount.iloc[i, 0]
        trackStr = []
        for item in tmp:
            trackStr.append( item[1] + '(' + item[0] +')' )
        trackCount.iloc[i, 0] = '->'.join(trackStr)

    return trackDistribution, trackCount

                        
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
    sNodes = list(set(sNodes))
    tNodes = list(set(tNodes))
    numOfSource = len(sNodes)
    numOfDest = len(tNodes)
    suggestPath = ''
    distanceInTotal = 0
    # 源城市两两之间距离，宿城市两两之间距离，源到宿两两之间距离
    distanceDict = dict()
    routeDict = dict()
    for i in range(numOfDest):
        for j in range(i+1,numOfDest):
            key = '-'.join([tNodes[i],tNodes[j]])
            if key in distanceDict.keys() or '-'.join([tNodes[j],tNodes[i]]) in distanceDict.keys():
                continue
            distanceDict[key], routeDict[key] = \
                findShortestPath(tNodes[i],tNodes[j],True)
    for i in range(numOfSource):
        for j in range(i+1,numOfSource):
            key = '-'.join([sNodes[i],sNodes[j]])
            if key in distanceDict.keys() or '-'.join([sNodes[j],sNodes[i]]) in distanceDict.keys():
                continue
            distanceDict[key], routeDict[key] = \
                findShortestPath(sNodes[i],sNodes[j],True)
    for i in range(numOfSource):
        for j in range(numOfDest):
            key = '-'.join([sNodes[i],tNodes[j]])
            if key in distanceDict.keys() or '-'.join([tNodes[j],sNodes[i]]) in distanceDict.keys():
                continue
            distanceDict[key], routeDict[key] = \
                findShortestPath(sNodes[i],tNodes[j],True)
    # 对源和宿按序号全排列
    sourceIndexList = []
    FullPermutation([i for i in range(numOfSource)], 0, numOfSource, sourceIndexList)
    destIndexList = []
    FullPermutation([i for i in range(numOfDest)], 0, numOfDest, destIndexList)
    # 获得全排列后，把源放在序列首位，两两之间求距离
    mFactorial = math.factorial(numOfSource)
    nFactorial = math.factorial(numOfDest)
    allTrackDistance = np.zeros((mFactorial * nFactorial, 1))
    routeList = []
    for k in range(mFactorial):
        for l in range(nFactorial):
            route = []
            trackList = []
            trackList += [sNodes[i_] for i_ in sourceIndexList[k]]
            trackList += [tNodes[i_] for i_ in destIndexList[l]]
            for j_ in range(len(trackList)-1):
                key = '-'.join(trackList[j_:j_+2])
                key = key if key in distanceDict.keys() else '-'.join([trackList[j_+1],trackList[j_]])
                allTrackDistance[k * nFactorial + l, :] += distanceDict[key]
                route.append(routeDict[key])
            route = parseRoute(trackList[0], route)
            routeList.append(route)
    # 找出距离最小的路线
    ind = np.argsort(allTrackDistance, axis=0)
    k = ind[:3,0] // nFactorial
    l = ind[:3,0] % nFactorial
    # 最小的路线是省的序列，结果字符串中将同一个省的城市按顺序连续输出，同时给出线路的近似长度
    # 放在distanceInTotal和suggestPath中
    distanceInTotal = allTrackDistance[ind,0][:3]
    suggestRoute = [routeList[iii_] for iii_ in ind[:3,0].tolist()]
    suggestPath = []
    
    for idx_ in range(len(k)):
        sourceList = [sNodes[i_] for i_ in sourceIndexList[k[idx_]]]
        destList = [tNodes[i_] for i_ in destIndexList[l[idx_]]]
        pathList = []
        for x in sourceList:
            pathList.append(x+'(提货)')
        for y in destList:
            pathList.append(y+'(卸货)')
    
        suggestPath.append('->'.join(pathList))

    
    return distanceInTotal, suggestPath, suggestRoute
        

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
    ax.set_xlabel('运单数')
    ax.set_title('客户{}线路分布直方图'.format(customer))
    
    for rect in rects:
        width = rect.get_width()
        ax.annotate('{}'.format(width), 
                    xy = (width, rect.get_y() + rect.get_height()/2),
                    xytext = (3, 0),
                    textcoords = "offset points",
                    ha = 'left', va = 'center')
    fig.tight_layout()
    plt.savefig('客户{}线路分布直方图.png'.format(customer))
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

# =============================================================================
# 求每一条路线长度
# =============================================================================



if __name__ == '__main__':
    #==========================================================================
    #客户需求
    customer = 'A'
    cities = {'提货':['陕西,咸阳'], '卸货':['陕西,西安','青海,西宁',\
                                    '陕西,安康']}
    #==========================================================================
    # 模型参数
    alpha = 0.2
    beta = 0.3
    gamma = 1-alpha-beta
    # 查数据库，得到线路历史分布
    trackDistribution,trackCount = trackDistributionStatistics(customer,cities)
    # 基于图论优化，找出经过这些点的最短路
    distanceInTotal, suggestPath, suggestRoute = suggestion(cities)
    print('算法推荐的最优地点访问顺序:\n'+suggestPath[distanceInTotal.argmin()])
    #=========================================================================
    # 专家评议法
    if trackDistribution.shape[0] > 0:
        for idx in range(trackDistribution.shape[0]):
            tmp = trackDistribution.loc[idx, 'track']
            trackStr = '->'.join([x[1]+'('+x[0]+')' for x in tmp])
            trackDistribution.loc[idx, 'track'] = trackStr
            dis = 0
            route = []
            for ddd in range(len(tmp) - 1):
                disTmp, routeTmp = findShortestPath(tmp[ddd][1], tmp[ddd+1][1], True)
                dis = dis + disTmp
                route.append(routeTmp)
            route = parseRoute(tmp[0][1], route)
            trackDistribution.loc[idx, 'distance'] = dis
            trackDistribution.loc[idx, 'route'] = route
        print('历史上符合查询要求的运单的路线和选择这条路线的人数')
        print(trackDistribution[['track','freq','distance']])
    else:
        print('历史上没有符合要求的运单')
        
    for idx in range(len(suggestPath)):
        ifexists = trackDistribution['track'] == suggestPath[idx]
        # if path not exists in query path
        if ifexists.max() != True:
            trackDistribution = trackDistribution.append(\
                {'track':suggestPath[idx],\
                 'route':suggestRoute[idx],\
                     'freq':0,\
                         'percentage':0,\
                             'hot':sigmoid(-0.5 * np.log(1 + trackCount['freq'].max())),\
                                 'distance':distanceInTotal[idx,0]},\
                    ignore_index=True)
    distance = trackDistribution['distance']
    if trackDistribution.shape[0] > 1:
        trackDistribution['short'] = sigmoid((distance.mean() - distance) / distance.std())
    else:
        trackDistribution['short'] = 1
    # calculate score
    trackDistribution['score'] = alpha * trackDistribution['hot'] + \
        beta * trackDistribution['percentage'] + \
            gamma * trackDistribution['short']

    highScoreIndex = np.argsort(trackDistribution['score'])[-3:]
    print('系统推荐路线：')
    count = 0
    for idx in highScoreIndex[::-1]:
        print('路线 %d：' % (count+1))
        print('节点序列：'+trackDistribution.loc[idx, 'track'])
        print('具体路线：'+trackDistribution.loc[idx, 'route'])
        fen = trackDistribution.loc[idx, 'score']
        print('推荐星级：{}'.format(suggestRank(fen)))
        routeDistance = trackDistribution.loc[idx, 'distance']
        print('近似总长度:{} km'.format(round(routeDistance, 3)))
        count = count + 1
    #==========================================================================
    #==========================================================================
    # 将客户线路分布统计结果存储为本地文件

    
    if not os.path.exists('trackCount-{}.csv'.format(customer)):
        trackCount.to_csv('trackCount-{}.csv'.format(customer),\
                            header=['Track','Freq','Hot'],index=False,encoding='UTF-8',quotechar='"')
    #==========================================================================
    # 画线路分布直方图
    plotBar(customer, trackCount, threshold = 100, limit=20)
   

