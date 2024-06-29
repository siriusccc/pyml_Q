from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# dataset = pd.read_csv('Social_Network_Ads.csv')
# X = dataset.iloc[:, [2, 3]].values
# y = dataset.iloc[:, 4].values

# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 特征缩放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 训练KNN分类器
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)
# 预测测试集
y_pred = classifier.predict(X_test)
# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

print('混淆矩阵：')
print(cm)
