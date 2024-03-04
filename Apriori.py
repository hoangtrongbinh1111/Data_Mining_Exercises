from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Dataset
dataset = [['Milk', 'Eggs', 'Butter'],
           ['Milk', 'Cookies', 'Butter'],
           ['Milk', 'Bread', 'Cookies', 'Butter'],
           ['Jam', 'Eggs', 'Bread'],
           ['Milk', 'Eggs', 'Cookies', 'Butter'],
           ['Jam', 'Bread']]

# convert data to a dataframe
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# apriori instance
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True, verbose=1)

# create rules from frequent itemsets
association_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)

# show results
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(association_rules)