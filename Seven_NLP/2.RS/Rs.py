# %%
# 基本协同算法推荐系统()
# 步骤:
    # 1.计算物品间的相似度
    # 2.引入评分,基于余弦的相似度计算
    # 3.基于用户的协同算法(人以群分)
    # 4.基于物品的协同过滤算法(物以类聚)

# ItermCF算法
# 此法是直接使用同时购买这两个物品的人数
import math 
def ItemSimilarity(train):
    c = dict()  # 同时被购买数
    N = dict()  # 购买用户数
    for u,items in train.items():   # 遍历字典取出键和键值对
        for i in items.keys():
            if i not in N.keys():
                N[i]  = 0
            N[i]  +=1
            for j in items.keys():
                if i == j:
                    continue
                if i not in c.keys():
                    c[i] = dict()
                if j not in c[i].keys():
                    c[i][j] = 0
                # 当用户同时购买了i和j,则加1
                c[i][j] +=1
    w = dict()  # 相似分
    for i,related_items in c.items():
        if i not in w.keys():
            w[i] = dict()
        for j ,cij in related_items.items():
            w[i][j] = cij/math.sqrt(N[i]*N[j])
    return w

Train_Data = {
'A':{'苹果':1,'⾹蕉':1,'⻄⽠':1},
'B':{'苹果':1,'⻄⽠':1},
'C':{'苹果':1,'⾹蕉':1,'菠萝':1},
'D':{'⾹蕉':1,'葡萄':1},
'E':{'葡萄':1,'菠萝':1},}

W = ItemSimilarity(Train_Data)
print(W)


# %%
# Item-CF余弦算法
# 存在用户同时购买了但是不喜欢的情况,所以数据集把用户评分引入到相似度中


import math 
def ItemSimilarity_cos(train):
    c = dict()  # 同时被购买数
    N = dict()  # 购买用户数
    for u,items in train.items():   # 遍历字典取出键和键值对
        for i in items.keys():
            if i not in N.keys():
                N[i]  = 0
            N[i] += items[i]* items[i]
            for j in items.keys():
                if i == j:
                    continue
                if i not in c.keys():
                    c[i] = dict()
                if j not in c[i].keys():
                    c[i][j] = 0
                # 当用户同时购买了i和j,则加评分乘积
                c[i][j] += items[i]*items[j]
    w = dict()  # 相似分
    for i,related_items in c.items():
        if i not in w.keys():
            w[i] = dict()
        for j ,cij in related_items.items():
            w[i][j] = cij / (math.sqrt(N[i])*math.sqrt(N[j]))
    return w

Train_Data = {
'A':{'苹果':2,'⾹蕉':2,'⻄⽠':2},
'B':{'苹果':2,'⻄⽠':2},
'C':{'苹果':2,'⾹蕉':2,'菠萝':2},
'D':{'⾹蕉':2,'葡萄':2},
'E':{'葡萄':2,'菠萝':2},
'F':{'⾹蕉':2,'⻄⽠':2}}
W= ItemSimilarity_cos(Train_Data)
print(W)

# %%
# 基于用户的协同算法(人以群分法)
# 先计算用户U和其他用户的相似度,然后取和U最相似的几个用户把他们购买过的物品推荐给用户U
def defItemIndex(DictUser):
    DictItem = defaultdict(defaultdict)
    # 遍历每个用户
    for key  in DictUser:
        # 遍历每个用户的购买记录
        for i in DictUser[key]:
            DictItem[i[0]][key]=i[1]
    return DictItem
# 根据相似度公式计算用户之间的相似度

def defUserSimilarity(DictItem):
    N = dict()   # 用户购买的数量
    C = defaultdict(defaultdict)
    W = defaultdict(defaultdict)
    # 遍历每个物品
    for key in DictItem:
        # 遍历用户K购买过的物品
        for i in DictItem[key]:
            # i[0]表示用户的id,如果未计算过,则初始化为0 
            if i[0] not in N.keys():
                N[i[0]] = 0     
            N[i[0]] += 1
            # (i,j)是物品k同时被购买的用户两两匹配对
            for j in DictItem[key]:
                if i(0) == j(0):
                    continue
                if j[0] not in C[i[0]].keys():
                    c[i[0]][j[0]] = 0 
                #  c[i[0]][j[0]]表示用户i和j购买同样物品的数量
                c[i[0]][j[0]] += 1
    for i ,related_user in C.items():
        for j ,cij in related_items.items():
            w[i][j] = cij/math.sqrt(N[i]*N[j])
    return W 

# %%
from collections import defaultdict
import math
def defItemIndex(DictUser):
    DictItem = defaultdict(defaultdict)
        ##遍历每个⽤户
    for key in DictUser:
    ##遍历⽤户k的购买记录
        for i in DictUser[key]:
            DictItem[i][key]=[key,DictUser[key][i]]
    return DictItem

def defUserSimilarity(DictItem):
    N=dict() #⽤户购买的数量
    C=defaultdict(defaultdict)
    W=defaultdict(defaultdict)
    ##遍历每个物品
    for key in DictItem:
    ##遍历⽤户k购买过的物品
    #print(key,":")
        for x in DictItem[key]:
            i = DictItem[key][x]
            #i[0]表示⽤户的id ，如果未计算过，则初始化为0
            if i[0] not in N.keys():
                N[i[0]]=0
                
            N[i[0]]+=1
            ## (i,j)是物品k同时被购买的⽤户两两匹配对
            for j in DictItem[key]:
                if i[0]==j[0]:
                    continue
                if j[0] not in C[i[0]].keys():
                    C[i[0]][j[0]]=0
            #C[i[0]][j[0]]表示⽤户i和j购买同样物品的数量
                C[i[0]][j[0]]+=1
    for i,related_user in C.items():
        for j,cij in related_user.items():
            W[i][j]=cij/math.sqrt(N[i]*N[j])
    return W

Train_Data = {
'A':{'苹果':1,'⾹蕉':1,'⻄⽠':1},
'B':{'苹果':1,'⻄⽠':1},
'C':{'苹果':1,'⾹蕉':1,'菠萝':1},
'D':{'⾹蕉':1,'葡萄':1},
'E':{'葡萄':1,'菠萝':1},
'F':{'⾹蕉':1,'⻄⽠':1}}
W= defItemIndex(Train_Data)

defUserSimilarity(W)

# %%
