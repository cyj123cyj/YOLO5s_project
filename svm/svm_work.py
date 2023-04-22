import numpy
import pandas
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

all_data = pandas.read_excel('data.xlsx', header=0)
classifications = all_data.iloc[:, 1]  # 分类的值
features = all_data.iloc[:, 2:]  # 特征值
# 数据标准化
std = StandardScaler()
X_std = std.fit_transform(features)
# 拆分训练集
x_train, x_test, y_train, y_test = train_test_split(features, classifications, test_size=0.3)
# SVC建模
svc_classification = SVC(kernel='rbf')
svc_classification.fit(x_train, y_train)
# 模型效果
scores = svc_classification.score(x_test, y_test)
print('模型预测效果分数：', scores)
pre_result = svc_classification.predict(x_test)
print("预测结果如下所示：")
print(pre_result)
# print('测试数据原本结果：')
# print(numpy.array(y_test.values.tolist()))
# print('测试数据如下所示：')
# print(x_test)‘

# 接下来绘制ROC曲线

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as AUC

print('数据：')
print(svc_classification.decision_function(x_test))
# 假正率，召回率，阈值
FPR, recall, thresholds = roc_curve(y_test, svc_classification.decision_function(x_test), pos_label=2)
area = AUC(y_test, svc_classification.decision_function(x_test))
print('输出：')
print(recall)
print(FPR)
print(area)

fig = plt.figure()
plt.plot(FPR, recall, color="red", label='ROC curve (area=%0.2f)' % area)  # 绘制的曲线
plt.plot([0, 1], [0, 1], color='black', linestyle='--')  # 中间斜线
# plt.xlim([-0.05,1.05])
# plt.ylim([-0.05,1.05])#坐标轴范围
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')  # 坐标轴标签
plt.title('Receiver operating characteristic example')  # title
plt.legend(loc="lower right")  # ROC curve标志存放位置
plt.show()
fig.savefig('svc_roc.pdf')




# 接下来进行五折交叉验证
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=False)
score_list = []
for train_index, test_index in kf.split(features):  # 调用split方法切分数据
    print('train_index:%s , test_index: %s ' %(train_index,test_index))
    train_part_data = pandas.DataFrame(features,
                                       index=train_index)
    test_part_data = pandas.DataFrame(features,
                                      index=test_index)
    y_part_train = pandas.DataFrame(classifications,
                                    index=train_index)
    y_part_test = pandas.DataFrame(classifications,
                                   index=test_index)
    # 开始训练模型
    svc_classification = SVC(kernel='rbf')
    svc_classification.fit(train_part_data, y_part_train)
    # 模型效果
    score = svc_classification.score(test_part_data, y_part_test)
    score_list.append(score)
print('五折交叉验证得分(分类精度)：')
print(score_list)



