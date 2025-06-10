import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# トランザクションデータの例
transactions = [['牛乳', 'パン'], ['牛乳', 'オムツ', 'ビール', 'パン'], ['パン', 'オムツ'], ['牛乳', 'オムツ', 'ビール']]

# データの準備
encoder = TransactionEncoder()
trans_encoded = encoder.fit(transactions).transform(transactions)
df = pd.DataFrame(trans_encoded, columns=encoder.columns_)

# 頻出アイテムセットの発見
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# アソシエーションルールの生成
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

print(rules)


output = "output/output.csv"

rules.to_csv(output, index=False)
