'''
1.Numpy    たくさんの数値（配列）を扱うのに便利なライブラリです。
2.Pandas   データを表として扱います。データの変換などがやりやすくなります。
3.Matplotlib          データや結果を図にして可視化するために使います。
4.Sklearn.datasets    このライブラリから、今回学習に使うサンプルのデータをとってきます。
5.Sklearn.neural_network.MLPClassifier        今回使う【ニューラルネットワーク】
6.Sklearn.model_selection.train_test_split    【訓練用】と【テスト用】に分けることができるライブラリ。
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree


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
'''
clf = DecisionTreeClassifier(max_depth=5)


'''
fit関数で、『訓練用データ』を使用してモデルの学習を実行
fit(X,Y)
'''
clf.fit(x, y)


'''
1. 訓練用データで行った学習の正解率を確認
2. 作成したモデルで『テスト用データ』から予測してみる
'''
study_score = clf.score(x, y)

# 1. 精度(正解率)を計算 → 表示
print(study_score)

# 2. 学習済みモデルと新たに読み取った『predict_csv』の値から予測 → その結果を表示
predict_csv = "input/predict/prediction_data.csv"  # 予測用CSVファイル名を指定
predict_data = pd.read_csv(predict_csv)

prediction = clf.predict(predict_data)
predict_rate = clf.predict_proba(predict_data)

# 表示サイズ
# グラフのサイズを幅15インチ、高さ10インチに指定
plt.figure(figsize=(15, 10))


#プロット内容の設定
plot_tree(clf, 
          feature_names=x.columns.values, 
          class_names =True, 
          filled=True,
          proportion=True  # 各ノードのsamplesを%表示, value を各クラスの比率で表示する
          )

# プロットした内容を表示させる
plt.show()

#  学習済みモデルから予測 → その結果を表示

print(f" 予測結果 =  {prediction} \n")
print(f"予測割合: {predict_rate}", end="\n\n\n")

output = "output/output.csv"

predict_data.insert(0, 'predicted_result', prediction)
predict_data.insert(1, 'train_score', study_score)
predict_data.to_csv(output, index=False)