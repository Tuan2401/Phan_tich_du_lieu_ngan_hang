import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def find_cross_sell_rules(df):

    basket = pd.get_dummies(df[["housing","loan","contact","poutcome"]])

    freq_items = apriori(basket, min_support=0.05, use_colnames=True)

    rules = association_rules(freq_items, metric="lift", min_threshold=1)

    rules = rules.sort_values("lift", ascending=False)

    print("\nTop luật bán chéo sản phẩm (Cross-Sell Rules):")
    print("\nGiải thích:")
    print("Antecedents  : Điều kiện ban đầu")
    print("Consequents  : Sản phẩm gợi ý")
    print("Support      : Tần suất xuất hiện")
    print("Confidence   : Xác suất xảy ra")
    print("Lift         : Mức độ liên kết")
    print(rules[["antecedents","consequents","support","confidence","lift"]].head(10))

    return rules