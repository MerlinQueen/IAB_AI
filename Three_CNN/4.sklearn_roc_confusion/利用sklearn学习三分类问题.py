# %%
print(__doc__)
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
from itertools import cycle 
from sklearn.metrics import confusion_matrix    
from sklearn.metrics import roc_curve,auc 

# %%
# 利用三分类问题理解混淆矩阵,Roc curve ,利用sep分隔符 参数选择
y_pred = pd.read_csv("D:\\Python_project\\Jupyter_project\\Three_week\\4.sklearn_roc_confusion\\y_predicted_label.csv",sep='\t')
y_test = pd.read_csv("D:\\Python_project\\Jupyter_project\\Three_week\\4.sklearn_roc_confusion\\y_true.csv",sep='\t')
y_score = pd.read_csv("D:\\Python_project\\Jupyter_project\\Three_week\\4.sklearn_roc_confusion\\y_predicted_score.csv",sep='\t')

# 文本清理
# %%
y_test.head(5)
#%% 
y_pred.head(5)
# %%
y_score.head(5)

# %%
###定义初始变量，请将fpr tpr rpc auc 定义成 dict型 ###
# 因三分类 n_classes = 3
n_classes = 3
# fpr  falase positive rate
# tpr  ture    positve rate
fpr = dict()
tpr = dict()
roc_auc = dict()

# %%
# 利用一个简单循环 和 roc_curve 函数roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
# 其中 y_true是 样本的真实标签
# y score 是 样本的预测得分
# 提示 利用pandas包 操作时 可以利用 pandas.iloc
# 因为roc_vurve返回三个参数,_表示接收的最后一个参数
for i in range(n_classes):
    fpr[i], tpr[i], threasholds = roc_curve(y_test.iloc[:, i], y_score.iloc[:, i]) #
    roc_auc[i] = auc(fpr[i], tpr[i])
# %%
# 利用循环打印出曲线
# zip 反应一组有序的元组对
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color,label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

     

lw = 2

plt.plot([0,1],[0,1],color = 'navy',lw=lw,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel("false positive rate")
plt.ylabel('ture positive rate')
plt.title('Reciver opetaing chareacteristic example')
plt.legend(loc='lower right')
plt.show()

# %%
# 把one-hot 标签转化为list形式的3值标签,pred_class已经给出,请对true_class作出
pred_class = []
for i in range(len(y_pred)):
     if y_pred.iloc[i,0] == 1:
          pred_class.append('label_1')
     if y_pred.iloc[i,1] == 1:
          pred_class.append('label_2')
     if y_pred.iloc[i,2] == 1:
          pred_class.append('label_3')
     if y_pred.iloc[i,0] ==y_pred.iloc[i,1] == y_pred.iloc[i,2] ==0:
          pred_class.append('no_class')

true_class = []
for i in range(len(y_test)):
    if y_test.iloc[i,0] == 1:
        true_class.append('label_1')
    if y_test.iloc[i,1] == 1:
        true_class.append('label_2')
    if y_test.iloc[i,2] == 1:
        true_class.append('label_3')     

# %%
# 利用confusion_matrix函数,生成混淆矩阵
cm = confusion_matrix(true_class,pred_class)
cm_df = pd.DataFrame(cm,index=['label_1','label_2','label_3','no_class'],columns=['label_1','label_2','label_3','no_class'])
plt.figure(figsize = (5.5,4))
sns.heatmap(cm_df,annot=True)
plt.title('example of confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# %%
