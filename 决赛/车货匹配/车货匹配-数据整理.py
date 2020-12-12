# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 19:23:37 2020

@author: lenovo
"""

import jieba
#f = open("foo.txt")               # 返回一个文件对象   
#line = f.readline()               # 调用文件的 readline()方法   
fname='b_tihuo.txt'
newstr=""
with open(fname,'r+',encoding='utf-8') as f:
    for line in f.readlines():
        txt=line[:-1].split(';')
        words  = jieba.lcut(txt[23])
        counts = {}
        for word in words:
            if len(word) == 1:
                continue
            else:
                counts[word] = counts.get(word,0) + 1
        items = list(counts.items())
        items.sort(key=lambda x:x[1], reverse=True)
        #print(items)
        strr=" "
        rs = line.rstrip('\n') #去除原来每行后面的换行符，但有可能是\r或\r\n
        newstr=newstr+rs
        for i in range(len(items)):
            if i<5:
                word, count = items[i] 
                newstr=newstr+' ;'+word+' ; '+str(count)
           # print(newstr)
        if len(items)<5:
            for i in range(5-len(items)):
                newstr=newstr+' ;'
        newstr=newstr+'\n'
        #newfile=open('1.txt','a')
        #newfile.write(newname+'\n')
        #newfile.close()
print(newstr)    
with open("b_tihuo_1.txt","w") as f:
        f.write(newstr) 
#        for i in len(items):
#            word, count = items[i]            
#            print ("{0:<10}{1:>5}".format(word, count))
#txt = open("new_car_time_onlyxiehuo.txt", "r", encoding='utf-8').readline()
#shengfen=("北京","天津","上海","重庆","河北","山西","辽宁","吉林","黑龙江省，江苏省，浙江省，安徽省，福建省，江西省，山东省，河南省，湖北省，湖南省，广东省，海南省，四川省，贵州省，云南省，陕西省，甘肃省，青海省，台湾省，内蒙古自治区，广西壮族自治区，西藏自治区，宁夏回族自治区，新疆维吾尔自治区，香港特别行政区，澳门特别行政区)
#while txt:
    #txt=txt[:-1].split(';')
        