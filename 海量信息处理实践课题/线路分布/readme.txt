本文件夹下各文件说明
citiesGPS.csv存储全国720个城市的GPS坐标信息。第一列是城市名称，第二列是城市的GPS坐标点，第三列是城市边界的GPS坐标。本文件夹中所有csv文件使用UTF-8编码。

findPositionSequence.py是模型主文件，完成论文4.2.1和4.2.3节的功能，并通过调用newgraph.py完成4.2.2节叙述的利用Floyd算法求最短路线的功能。
在代码267和268两行可以修改，功能是指定客户A还是客户B，提货、卸货地点需求。之后代码会运行出算法推荐的最优地点访问顺序、历史上满足需求的路线和运单数，系统推荐路线的地点访问顺序、具体路由、推荐星级和近似总长度。模型按照得分从高到低至多推荐三条路线。

hbasecommands.txt是在HBase中创建表格、存储数据、查询数据的全部代码。

hivecommands.txt是在Hive中创建数据库、创建表格、导入数据、查询数据的代码。

newDistance.npz存储Floyd算法优化需要的初始权值矩阵和路由矩阵

newDistance_shortest.npz存储Floyd算法运行完毕之后的权值矩阵和路由矩阵，是最短路径

newgraph.py实现Floyd算法求图中任意两节点之间的最短路径，给出最短路径的长度和路由。一般直接加载newDistance_shortest.npz文件存储的权值矩阵和路由矩阵，返回结果。

track_A.csv存储客户A每个运单对应的路线，两列分别是：运单ID、路线途经城市的行为和顺序。

track_B.csv存储客户B每个运单对应的路线，两列分别是：运单ID、路线途经城市的行为和顺序。

trackCount-A.csv存储客户A的所有线路和对应运单数，第一列Track代表线路，第二列Count代表走这条线路的运单数，第三列Hot代表路线的热门度。

trackCount-B.csv存储客户B的所有线路和对应运单数，第一列Track代表线路，第二列Count代表走这条线路的运单数，第三列Hot代表路线的热门度。