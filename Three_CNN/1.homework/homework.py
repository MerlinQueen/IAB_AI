# %%
import numpy as np 
import math 
import pandas as pd 
import matplotlib.pyplot as plt 


# %%
"""
    线性变换f, Additivity：f(V+U)=f(v)+f(u); homogeneity f(aV)=af(V)
    证明变f（X）= AX +b 不是线性变换
    f(X+Y)= A(X+Y) + b  不等 f（X）+  f(Y)   
    不符合可加性，称为仿射变换
    通过把b加到A的最后一列，X向量增加一维，是常数1，新的变换是线性变换。
"""

# %%

"""
给定变换矩阵 A=[[1.5 , 0.5], [0.5, 1.5]]
通过作图，证明单位向量集合的A变换，是A的本征向量为轴的椭圆
给定A的本征变换为UDV，画出xy平面中由点（0，0), (0,1),(1,1),(1,0)形成的正方形内的点的集合，依次经过V、D、U变换后的区域
"""


# %%
A = np.array([[1.5,0.5],[0.5,1.5]])


# %%
