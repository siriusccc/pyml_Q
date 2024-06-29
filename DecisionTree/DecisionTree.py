from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号

# 生成武侠风格的数据，确保所有特征值为正数
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, random_state=42)
X += np.abs(X.min())  # 平移数据确保为正

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建决策树模型，并设置最大深度为3
dt = DecisionTreeClassifier(max_depth=3)

# 训练模型
dt.fit(X_train, y_train)


# 绘制数据点和决策边界
def plot_decision_boundary(model, X, y):
    # 设置最小和最大值，以及增量
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # 预测整个网格的值
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.4)
    # 绘制不同类别的样本点
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', marker='x', label='普通武者')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', marker='o', label='武林高手')
    plt.xlabel('功力值')
    plt.ylabel('内功心法')
    plt.title('武侠世界中的武者分类图')
    plt.legend()


# 绘制决策边界和数据点
plot_decision_boundary(dt, X, y)
plt.show()
