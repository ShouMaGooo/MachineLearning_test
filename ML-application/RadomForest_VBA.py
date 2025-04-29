'''
1.Pandas   データを表として扱います。データの変換などがやりやすくなります。
3.Matplotlib          データや結果を図にして可視化するために使います。
5.sklearn.tree_plot_tree        今回使う【ニューラルネットワーク】
6.Sklearn.model_selection.train_test_split    【訓練用】と【テスト用】に分けることができるライブラリ。
'''

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

'''
アヤメのデータをライブラリーから取り込み
『学習用データ』と『テスト用データ』に分割
'''
# CSVデータの読み込み
input_csv = "input/train/train_data.csv"  # 入力CSVファイル名を指定
data = pd.read_csv(input_csv)

# .ilocは"Integer location"の略
# iloc[行,列]のように指定する
# [:, 0] で『全ての行＋0列目』    [:, 1:] で『全ての行＋1列目から最後の列まで』
y = data.iloc[:, 0]
x = data.iloc[:, 1:]

print(f"x={x}")
print(f"x_columns={x.columns.values}")

print(f"y={y}")
print(f"y_columns={y.name}")

'''
決定木の作成(深さ５)
n_estimators：木の数(指定がないときは１００),  max_depth：深さ,
random_state：疑似乱数を出力する方式（ランダムシードという）
'''
RFC_clf = RandomForestClassifier(max_depth=5, random_state=42)


'''
fit関数で、『訓練用データ』を使用してモデルの学習を実行
fit(X,Y)
'''
RFC_clf.fit(x, y)


'''
1. 訓練用データで行った学習の正解率を確認
2. 作成したモデルで『テスト用データ』から予測してみる
'''


# 1. 精度(正解率)を計算 → 表示
Score=RFC_clf.score(x, y)
print(f"正解率は :{Score}")

# 2. 学習済みモデルと新たに読み取った『predict_csv』の値から予測 → その結果を表示
predict_csv = "input/predict/prediction_data.csv"  # 予測用CSVファイル名を指定
predict_data = pd.read_csv(predict_csv)

prediction = RFC_clf.predict(predict_data)
predict_rate = RFC_clf.predict_proba(predict_data)


# ランダムフォレスト内の決定木の数を表示
# len()：()内の該当する個数を表示   estimators_：
print(f"ランダムフォレスト内の決定木の数: {len(RFC_clf.estimators_)}", end="\n\n\n")

# 表示サイズを幅100インチ、高さ100インチに指定
#fig = plt.figure(figsize=(70, 70))

# プロットする木を指定
tree_1 = RFC_clf.estimators_[0]
tree_2 = RFC_clf.estimators_[1]
tree_3 = RFC_clf.estimators_[2]

# .subplots関数: 個数の表示を　『縦1個×横３個』 表示させる
fig, axes = plt.subplots(1, 3, figsize=(50, 50))

# ３つの木をプロットさせる
plot_tree(tree_1, feature_names=x.columns.values, class_names =True, proportion=True, filled=True, precision=3, ax=axes[0])
plot_tree(tree_2, feature_names=x.columns.values, class_names =True, filled=True, precision=3, ax=axes[1])
plot_tree(tree_3, feature_names=x.columns.values, class_names =True, filled=True, precision=3, ax=axes[2])


# プロットした内容を表示させる
plt.show()



#  学習済みモデルから予測 → その結果を表示

print(f" 予測結果 =  {prediction} \n")
print(f"予測割合: {predict_rate}", end="\n\n\n")

output = "output/output.csv"

predict_data.insert(0, 'predicted_result', prediction)
predict_data.insert(1, 'train_score', Score)
predict_data.to_csv(output, index=False)