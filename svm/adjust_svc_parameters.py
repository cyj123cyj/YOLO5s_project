import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

all_data = pandas.read_excel('data.xlsx', header=0)
classifications = all_data.iloc[:, 1]  # 分类的值
features = all_data.iloc[:, 2:]  # 特征值
print(features)
print(classifications)
# 定义参数组合 gamma probability
params = {'C': [0.001,0.01,0.1,1,10,100,1000], 'kernel': ['rbf', 'poly', 'sigmoid'],#[0.01, 0.1, 1, 1.2, 10, 100]
          'degree': [1,3,4,5]}  # c越大越容易过拟合，c越小容易出现欠拟合 ,'gamma': ['scale', 'auto']
svc_classification = SVC()
model = GridSearchCV(svc_classification, param_grid=params, cv=5)
model.fit(features, classifications)
print('最好的参数组合：')
print(model.best_params_)
print('最好的分数：')
print(model.best_score_)











