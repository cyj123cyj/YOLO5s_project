import pandas
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 定义数据集
all_data = pandas.read_excel('data.xlsx', header=0)  # 读取数据
y = all_data.iloc[:, 1]
X = all_data.iloc[:, 2:]
# 定义模型
model = LinearDiscriminantAnalysis()
# 定义模型评价方法
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# 定义网格
grid = dict()
grid['solver'] = ['svd', 'lsqr', 'eigen']
# 定义搜索
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# 执行搜索
results = search.fit(X, y)
# 总结
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)