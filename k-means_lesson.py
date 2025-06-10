
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

iris = datasets.load_iris()

#Dataframe型に変換
# iris.data：(150,4) 　iris.feature_names：説明変数の名称
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)


# train_test_split関数で「訓練用データ」と「テスト用データ」に分割
# iris.data：(150,4)    iris.target：3種類の正解ラベルが150個    test_size：全データの2割がテスト用
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, 
                                                                    random_state=0)



Y_box = np.unique(Y_test)

print(Y_box)

k = len(Y_box)

print(f" k = {k}")

kmeans = KMeans(n_clusters=len(Y_box), init='k-means++')
kmeans.fit(X_train)


y_kmeans = kmeans.predict(X_test)

print(y_kmeans)
print(Y_test)