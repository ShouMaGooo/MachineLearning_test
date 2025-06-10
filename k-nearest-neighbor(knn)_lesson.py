import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

# K近傍法
from sklearn.neighbors import KNeighborsClassifier


'''
アヤメのデータをライブラリーから取り込み
『学習用データ』と『テスト用データ』に分割
'''
#データセットのライブラリーからイリスのデータセットを取り込む
#dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
# iris.data,  iris.target,  iris.target_names,     ..... 
iris = datasets.load_iris()

#Dataframe型に変換
# iris.data：(150,4) 　iris.feature_names：説明変数の名称
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)


# train_test_split関数で「訓練用データ」と「テスト用データ」に分割
# iris.data：(150,4)    iris.target：3種類の正解ラベルが150個    test_size：全データの2割がテスト用
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, 
                                                                    random_state=0)



# k=7(最も距離が近い順に7個とり、多数決で判別)
knn = KNeighborsClassifier(n_neighbors = 7)

# 学習
knn.fit(X_train,Y_train)

# テストデータを予測
Y_pred = knn.predict(X_test)

# 精度 check
print(knn.score(X_test,Y_test))