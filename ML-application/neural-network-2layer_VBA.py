'''
1.Pandas   データを表として扱います。データの変換などがやりやすくなります。
2.Matplotlib          データや結果を図にして可視化するために使います。
3.Sklearn.neural_network.MLPClassifier        今回使う【ニューラルネットワーク】
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


'''
アヤメのデータをライブラリーから取り込み
『学習用データ』と『テスト用データ』に分割
'''
# CSVデータの読み込み
input_csv = "input/train/train_data.csv"  # 入力CSVファイル名を指定
data = pd.read_csv(input_csv,header=0)



# .ilocは"Integer location"の略
# iloc[行,列]のように指定する
# [:, 0] で『全ての行＋0列目』    [:, 1:] で『全ての行＋1列目から最後の列まで』
y = data.iloc[:, 0]
x = data.iloc[:, 1:]

scaler = StandardScaler()
scaler.fit(x)

'''
　ニューラルネットワークの作成
　隠れ層：10  活性化関数：Reru関数   重み更新などの最適化手法：adam  学習の繰り返し回数：1000
 ※実は、引数なしで作成可能！　　　例）　clf=MLPClassifier()
'''
clf = MLPClassifier(hidden_layer_sizes=40, activation='relu', solver='adam', max_iter=1000)



'''
fit関数で、『訓練用データ』を使用してモデルの学習を実行
書き方 : fit(X,Y)
'''
clf.fit(x, y)

train_score = clf.score(x, y)

'''
1. 訓練用データで行った学習の正解率を確認
2. 作成したモデルで『テスト用データ』から予測してみる
'''
# 1. 精度(正解率)を計算 → 表示


print(train_score)
print("=================================================")


# 2. 学習済みモデルと新たに読み取った『predict_csv』の値から予測 → その結果を表示
predict_csv = "input/predict/prediction_data.csv"  # 予測用CSVファイル名を指定
predict_data = pd.read_csv(predict_csv)

#scaler2 = StandardScaler()
#predict_data_scaler = scaler2.fit(predict_data)

prediction = clf.predict(predict_data)
predict_rate = clf.predict_proba(predict_data)



'''
誤差関数(損失関数)を算出し
グラフにプロット
'''
# 損失関数で誤差をプロット → 損失関数曲線
plt.plot(clf.loss_curve_)

#図のタイトルに損失関数曲線
plt.title("Loss Curve")

#横軸(x軸)に学種回数
plt.xlabel("Iteration")

#縦軸(y軸)に損失関数値
plt.ylabel("Loss")

#グラフにグリッド線(格子状に引く)を表示
plt.grid()

#プロットしたグラフを表示させる
plt.show()





#  学習済みモデルから予測 → その結果を表示

print(f" 予測結果 =  {prediction} \n")
print(f"予測割合: {predict_rate}", end="\n\n\n")

output = "output/output.csv"

predict_data.insert(0, 'predicted_result', prediction)
predict_data.insert(1, 'train_score', train_score)
predict_data.to_csv(output, index=False)
