# %%
# 爬取历史天气数据
# 导入必须的包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib
import requests
from bs4 import BeautifulSoup
import csv
# %%
# 网页分析,爬取城市,时间选择

cities = ['chengdu','aba','bazhong','dazhou','deyang','ganzi','guangan',
          'guangyuan','leshan','luzhou','meishan','mianyang','neijiang','nanchong',
          'panzhihua','scsuining','yaan','yibin','ziyang','zigong','liangshan']
url_base = 'http://www.tianqihoubao.com/lishi/'
  

# %%
# 自定义天气查询
urls = []
def URLS(city,year):
    for month in range(1,13):
        if month<10:
            url = url_base+city+'/month/'+str(year)+'0'+str(month)+'.html'
        else:
            url = url_base+city+'/month/'+str(year)+str(month)+'.html'
        urls.append(url)
URLS(cities[0],2018)
# %%
def getHtml():
    # 获取天气数据并保存
    try:
        for url in urls:
            url = 'http://www.tianqihoubao.com/lishi/chengdu/month/201101.html'
            html = requests.get(url)
    except urllib.error.URLError as e:
        if hasattr(e, 'code'):
            print(e.code)
        if hasattr(e,'reason'):
            print(e.reason)
    return html

# %%
# 数据解析
html = getHtml()
soup = BeautifulSoup(html.text,"lxml")
table = soup.find_all('table',{'class':'b'})[0]
tb = pd.read_html(html.text)[0]
tb.to_csv(r'weather.csv', mode='a', encoding='utf_8_sig', header=1, index=0)

# df = pd.DataFrame(data,columns=['data','weather','temperature','wind'])
# df.head(5)


# %%
