from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import time

# Dataset
dataset = [['Milk', 'Eggs', 'Butter'],
           ['Milk', 'Cookies', 'Butter'],
           ['Milk', 'Bread', 'Cookies', 'Butter'],
           ['Jam', 'Eggs', 'Bread'],
           ['Milk', 'Eggs', 'Cookies', 'Butter'],
           ['Jam', 'Bread']]
def Apriori():
    """
    This function performs the Apriori algorithm on a given dataset.

    Parameters:
    dataset (list of lists): A list of transactions, where each transaction is a list of items.
    min_support (float): The minimum support threshold. Items that occur in at least this many transactions are considered frequent.
    min_confidence (float): The minimum confidence threshold. Rules that have a confidence greater than or equal to this value are considered significant.
    min_lift (float): The minimum lift threshold. Rules that have a lift greater than or equal to this value are considered significant.
    use_colnames (bool): A boolean value indicating whether the items in the dataset are represented by column names (True) or by their indices (False).
    verbose (int): An integer value that controls the verbosity of the output.

    Returns:
    frequent_itemsets (list of dicts): A list of frequent itemsets, where each itemset is represented as a dictionary with the item names as keys and the support count as the value.
    association_rules (list of dicts): A list of association rules, where each rule is represented as a dictionary with the antecedents and the consequent as keys, and the confidence and lift values as the values.
    elapsed_time (float): The total time taken by the algorithm to complete.

    """
    # Record the starting time
    start_time = time.time()

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
    # Record the ending time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Time taken:", elapsed_time, "seconds")