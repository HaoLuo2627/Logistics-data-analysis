本文件夹下各文件说明
distance.npz存储Floyd算法优化需要的初始权值矩阵和路由矩阵

distance_shortest.npz存储Floyd算法运行完毕之后的权值矩阵和路由矩阵，是最短路径

findPositionSequence.py是模型主文件，完成论文4.2.1和4.2.3节的功能，并通过调用graph.py完成4.2.2节叙述的利用Floyd算法求最短路线的功能。
在代码252和253两行可以修改，功能是指定客户A还是客户B，提货、卸货地点需求。之后代码会运行出图论算法推荐路线、历史上满足需求的路线和运单数，系统最终推荐路线，推荐星级和路线近似总长度。

graph.Py实现Floyd算法求图中任意两节点之间的最短路径，给出路由和最短路径的长度。一般直接加载distance_shortest.npz文件存储的权值矩阵和路由矩阵，返回结果。

track_A.csv存储客户A每个运单对应的路线，两列分别是：运单ID、路线途径城市的行为和顺序

track_B.csv存储客户B每个运单对应的路线。

trackCount-A.csv存储客户A的所有线路和对应运单数，第一列Track代表线路，第二列Count代表走这条线路的运单数，第三列Percentage代表走这条线路的运单数占客户A的运单总数的比例。

trackCount-B.csv存储客户B的所有线路和对应运单数。