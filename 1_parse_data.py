import os
import pickle
import sys

import numpy as np
from sklearn.model_selection import KFold

from get_data import get_data

data_name = sys.argv[1]  # yelp, netflix, amazon
min_items_in_categ = int(sys.argv[2])
min_reviews_of_user = int(sys.argv[3])
min_reviews_of_item = int(sys.argv[4])

n_folds = 3
n_tests = 50
help_folder = "data_splits"

for i in range(n_folds):
    os.makedirs(f"{help_folder}/{data_name}/{i}", exist_ok=True)

df, categs = get_data(data_name)

categ_order = sorted([c for c in categs.keys() if len(categs[c]) >= min_items_in_categ])
groups = [categs[cat] for cat in categ_order]

user_rws = df.groupby(["user_id"]).size()
users = [
    user
    for user, review_count in user_rws.items()
    if review_count >= min_reviews_of_user
]
print(f"Users from {len(user_rws)} to {len(users)}")
df_sub = df[df["user_id"].isin(users)]

item_rws = df.groupby(["item_id"]).size()
items = {i: False for i in item_rws.keys()}
for g in groups:
    for i in g:
        items[i] = True
ok_items = [
    i
    for i, v in items.items()
    if v and i in item_rws and item_rws[i] >= min_reviews_of_item
]
print(f"Items from {len(item_rws)} to {len(ok_items)}")

df_sub = df_sub[df_sub["item_id"].isin(ok_items)]
df_sub.to_csv(f"{help_folder}/{data_name}/df.csv", index=False)
print(f"Reviews from {df.shape[0]} to {df_sub.shape[0]}")

groups = [[i for i in g if i in ok_items] for g in groups]

folds = KFold(3, shuffle=True, random_state=0)
np.random.seed(0)
for fold, (train_i, test_i) in enumerate(folds.split(users)):
    train_users = np.array(users)[train_i]
    with open(f"{help_folder}/{data_name}/{fold}/train.pickle", "wb") as f:
        pickle.dump(train_users, f)
    test_users = np.random.choice(np.array(users)[test_i], n_tests, replace=False)
    with open(f"{help_folder}/{data_name}/{fold}/test.pickle", "wb") as f:
        pickle.dump(test_users, f)
    df_sub_fold = df_sub[df_sub["user_id"].isin(train_users)]

    items = df_sub_fold["item_id"].unique()
    groups_fold = []
    categ_order_fold = []
    for j, g in enumerate(groups):
        g_items = [i for i in g if i in items]
        if len(g_items) > 0:
            groups_fold.append(g_items)
            categ_order_fold.append(categ_order[j])
    with open(f"{help_folder}/{data_name}/{fold}/groups.pickle", "wb") as f:
        pickle.dump((groups_fold, categ_order_fold), f)
