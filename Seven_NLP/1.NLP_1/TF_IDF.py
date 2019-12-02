# %%
import pkuseg



# %%
# 分词
seg = pkuseg.pkuseg()
seg.cut("统计局修订去年GDP数据:比初步核算增1.897万亿元")

# %%
# 删除标点符号
# Unicdoe4E00~9FFF表示中文，所以如果一个字符的utf-8编码在这个区间内，就说明它是中文
# 判断是否是中文
def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

# 格式化不属于中文的字符
def format_str(content):
    content_str = ''   # 创建一个空支字符,存放筛选过的字
    for i in content:       # 遍历整句话,检测是否是中文
        if is_chinese(i):
            content_str  = content_str+i
    return content_str



# %%
# 用Sklearn 计算 tf-idf权重
# 导入特征提取包
from sklearn import feature_extraction

# 导入TFidf转换工具
from sklearn.feature_extraction.text import TfidfTransformer
# 导入计算词频矩阵包
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 


# %%



fileName = [r'D:\Python_project\Jupyter_project\Seven_week\1.NLP_1\a.txt',
            r'D:\Python_project\Jupyter_project\Seven_week\1.NLP_1\b.txt',
            r'D:\Python_project\Jupyter_project\Seven_week\1.NLP_1\c.txt']    

# %%
# 创建一个空全集列表,存放关键字
corpus = []
for name in fileName:
    with open(name,'r') as f:
        str = f.read()   # 读取文件内容
        str = format_str(str)  # 去除非中文字符
        str = seg.cut(str)  # 进行分词
        corpus.append('{}'.format(str)) # 把字符串,以字符元素加入空了列表

print(corpus)

# 该类,会将文本中的词语转换为词频矩阵,矩阵元素a[i][j]表示j次在i类文本中的词频
vectorizer = CountVectorizer()
# 该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()

# 在给corpus加入内容之后使用 CountVectorizer() 建立词频矩阵
# TfidfTransformer() 统计每个词语的tf-idf权值
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus)) 
# 获取词袋类型中所有的词语
word = vectorizer.get_feature_names()
weight = tfidf.toarray() 


for (name,w) in zip(fileName,weight):
    # 第一个fit_reansform是计算tf-idf,第二个transform是将文本转换成词频   
    print(name,":")
    loc = np.argsort(-w)  # 按权重重要性排序
    for i in range(5):
        print(i+1,word[loc[i]],w[loc[i]])

# 把tf-idf矩阵抽取出来,元素a[i][j]表示j词在I类文本中的tf-idf权重




# %%
