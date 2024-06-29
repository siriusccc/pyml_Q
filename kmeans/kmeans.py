import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 加载鸢尾花数据集
data = load_iris()
X = data.data  # 特征

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用K-means聚类
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_scaled)

# 聚类结果
clusters = kmeans.labels_

# 将聚类结果添加到数据框中
iris_df = pd.DataFrame(X, columns=data.feature_names)
iris_df['cluster'] = clusters

# 可视化聚类结果
sns.pairplot(iris_df, diag_kind='kde', hue='cluster', palette='viridis')
plt.suptitle('K-means Clustering on Iris Dataset', y=1.02)
plt.show()
