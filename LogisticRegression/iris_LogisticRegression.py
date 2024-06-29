import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_iris()
iris_target = data.target
# 得到数据对应的标签
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)
iris_features.info()
# 合并标签和特征信息
iris_all = iris_features.copy()
iris_all['target'] = iris_target
# 可视化
sns.pairplot(data=iris_all, diag_kind='hist', hue='target')
plt.show()

iris_features_part = iris_features.iloc[:100]
iris_target_part = iris_target[:100]
x_train, x_test, y_train, y_test = train_test_split(iris_features_part, iris_target_part, test_size = 0.2, random_state = 2020)
clf = LogisticRegression(random_state=0, solver='lbfgs')
# 训练逻辑回归模型
clf.fit(x_train, y_train)
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
