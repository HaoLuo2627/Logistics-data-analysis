# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:41:07 2020

@author: lenovo
"""
# -*- coding: utf-8 -*-

import pandas as pd
import xgboost as xgb
import numpy as np
import warnings
import math
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

f = open("predict1.txt")               # 返回一个文件对象   
lines = f.readlines()               # 调用文件的 readline()方法   
fname='predict1.txt'
newstr=""
line0=[]
with open(fname,'r+',encoding='utf-8') as f:
    for line in f.readlines():
        txt=lines[0].split(';')
        line0.append(txt)
        
data_path='./predict3.txt'
data=pd.read_csv(data_path,header=None,sep='\s+',converters={7:lambda x:int(x)-1})
data.rename(columns={2:'lable'},inplace=True)
print(data)

# # # 生产一个随机数并选择小于0.8的数据
# mask=np.random.rand(len(data))<0.8
# train=data[mask]
# test=data[~mask]
#
# # 生产DMatrix
# xgb_train=xgb.DMatrix(train.iloc[:,:6],label=train.lable)
# xgb_test=xgb.DMatrix(test.iloc[:,:6],label=test.lable)
fangshu = input('请输入运单方数：')
dunshu = input('请输入运单吨数：')
#data.append([fangshu,dunshu,0])


X=data.iloc[:,:2]
Y=data.iloc[:,2]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=100)
'''
X_train=X
y_train=Y
y_test =0
X_test=(fangshu,dunshu)
'''
#X_test=X_test.append([fangshu,dunshu,0])
xgb_train=xgb.DMatrix(X_train,label=y_train)
xgb_test=xgb.DMatrix(X_test,label=y_test)



# 设置模型参数

params={
    'objective':'multi:softprob',
    'eta':0.1,
    'max_depth':5,
    'num_class':7
}

watchlist=[(xgb_train,'train'),(xgb_test,'test')]
# 设置训练轮次，这里设置60轮
num_round=60
bst=xgb.train(params,xgb_train,num_round,watchlist)

# 模型预测

pred=bst.predict(xgb_test)
pred1=pred
print(pred)

pred[0]=pred[0]*0.85+(-1/80)*0.15*math.exp(2)
pred[1]=pred[1]*0.85+(-1/80)*0.15*math.exp(2.5)
pred[2]=pred[2]*0.85+(-1/80)*0.15*math.exp(1.8)
pred[3]=pred[3]*0.85+(-1/60)*0.15*math.exp(3)
pred[4]=pred[4]*0.85+(-1/90)*0.15*math.exp(1)
pred[5]=pred[5]*0.85+(-1/80)*0.15*math.exp(2)
pred[6]=pred[6]*0.85+(-1/80)*0.15*math.exp(2.2)
#模型评估

'''
# error_rate=np.sum(pred!=test.lable)/test.lable.shape[0]
error_rate=np.sum(pred!=y_test)/y_test.shape[0]

print('测试集错误率(softmax):{}'.format(error_rate))

accuray=1-error_rate
print('测试集准确率：%.4f' %accuray)


# 模型保存
bst.save_model("./002.model")


# 模型加载
bst=xgb.Booster()
bst.load_model("./002.model")
pred=xgb.predict_proba(xgb_test)
print(pred)
'''