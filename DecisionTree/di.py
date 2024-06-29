import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# 加载鸢尾花数据集
data = load_iris()
X = data.data  # 特征
y = data.target  # 目标标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 特征缩放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 训练决策树分类器
dt = DecisionTreeClassifier(random_state=0, max_depth=3)
dt.fit(X_train, y_train)

# 预测测试集
y_pred = dt.predict(X_test)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print('混淆矩阵：')
print(cm)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'准确率：{accuracy:.2f}')

# 分类报告
report = classification_report(y_test, y_pred, target_names=data.target_names)
print('分类报告：')
print(report)

# 可视化决策树
plt.figure(figsize=(12, 8))
tree.plot_tree(dt, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()
