
 
 # MachineLearning_testフォルダ内の分析モデルについて

このフォルダ内の５つの分析モデルは、インターネット経由でサンプルデータを所得、または自作したダミーのデータを用いて分析を行っています。
『ML-application』フォルダ内の分析モデルは、csvファイルからデータを読み込み、ダミーの1レコードデータを用いて予測を行っています。

	k-means_lesson.py  
	➡k-means法を用いたクラスター分析モデルです。

	k-nearest-neighbor(knn)_lesson.py  
	➡k近傍法を用いたクラスター分析です。

	Market_Basket_Analysis.py  
	➡バスケット分析(アソシエーションの一種)を用いた分析モデル。
	データは練習の為に架空のものを作成致しました。

	recommend_cos0_similarity.ipynb  
	➡コサイン類似度を利用した協調フィルタリングモデルとなっています。
	映画のレコメンドシステムを想定したものになります。

	CNN_RGB_CIFAR-10.ipynb
	➡畳み込み層とニューラルネットワークを利用した画像認識モデルです。
	CIFAR-10の画像を利用して、分析するモデルとなっています。


# ML-applicationフォルダ内の分析モデルについて

    『input』フォルダ内にある『train』フォルダ内にあるcsvファイルで学習を行い、
    『input』フォルダ内にある『predict』フォルダ内にあるcsvファイルの1レコードを利用して、正解値を予測するようにしています。
           ↓
     『予測結果』及び『正解率』と予測元のデータがcsv形式で出力され、『output』フォルダ内に格納されます。

	neural-network.py
	➡ニューラルネットワーク(隠れ層１０)のモデル
     
	decisionTree.py
	➡決定木

	RadomForest.py
	➡ランダムフォレスト
