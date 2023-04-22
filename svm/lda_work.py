import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as AUC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict

lda_model = LinearDiscriminantAnalysis()  # 创建LDA模型
all_data = pandas.read_excel('data.xlsx', header=0)  # 读取数据
classifications = all_data.iloc[:, 1]
features = all_data.iloc[:, 2:]
train_classifications = all_data.iloc[0:48, 1]  # 分类的值
train_features = all_data.iloc[0:48, 2:]  # 特征值
test_classifications = all_data.iloc[48:, 1]
test_features = all_data.iloc[48:, 2:]

# 五折交叉验证
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
scores = cross_val_score(lda_model, train_features, train_classifications, scoring='accuracy', cv=cv, n_jobs=-1)
print('五折交叉验证标准差：')
print(numpy.std(scores))  # 分数的标准差
print('五折交叉验证均值：')
print(numpy.mean(scores))  # 分数均值
print('五折交叉验证所有分数：')
print(scores)  # 所有分数

# 接下来使用LDA训练好的模型进行预测
lda_model.fit(train_features, train_classifications)
predict_result = lda_model.predict(test_features)  # 预测结果数据
print('预测结果：')
print(predict_result)
print('真实结果：')
print(test_classifications.values.tolist())

# 接下来绘制不同阈值下的ROC曲线
y_scores = cross_val_predict(lda_model,train_features,train_classifications, cv=3,
                             method="decision_function")
fpr, tpr, thread = roc_curve(test_classifications.values.tolist(), y_scores,pos_label=2)
fig = plt.figure()
plt.plot(fpr, tpr, color="red", label='ROC curve')  # 绘制的曲线
plt.plot([0, 1], [0, 1], color='black', linestyle='--')  # 中间斜线
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')  # 坐标轴标签
plt.title('Receiver operating characteristic example')  # title
plt.legend(loc="lower right")  # ROC curve标志存放位置
plt.show()
fig.savefig('lda_roc.pdf')


