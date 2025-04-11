from ease_model import EASE
import pickle
import pandas as pd
from LiCE.spn.SPN import SPN
from LiCE.data.DataHandler import DataHandler
from scipy.sparse import hstack, csc_matrix
import os
import sys
import numpy as np

data_name = sys.argv[1]
fold = int(sys.argv[2])
group_f = sys.argv[3]
rating_used = sys.argv[4] == "rating"

help_folder = "data_splits"
max_samples_for_spn = 50_000
min_instances_spn = 5000

os.makedirs(f"{help_folder}/{data_name}/{fold}/models", exist_ok=True)

df = pd.read_csv(f"{help_folder}/{data_name}/df.csv")
with open(f"{help_folder}/{data_name}/{fold}/groups.pickle", "rb") as f:
    (groups, categ_order) = pickle.load(f)

max_rating = df['rating'].max() if rating_used else 1

with open(f"{help_folder}/{data_name}/{fold}/train.pickle", "rb") as f:
    train_users = pickle.load(f)

df = df[df["user_id"].isin(train_users)]

ease = EASE()
ease.fit(df, implicit=not rating_used) # implicit = True means that rating is disregarded (always 1)

with open(f"{help_folder}/{data_name}/{fold}/models/ease_{'rating' if rating_used else 'binary'}.pickle", "wb") as f:
    pickle.dump(ease, f)

# -- train SPN
groups = [ease.item_enc.transform(items) for items in groups]
bounds = {c:(0, 1) for c in categ_order}
discrete = [] if rating_used else categ_order
cat_map = {}
if group_f == "sum":
    spn_train_data = [csc_matrix(ease.X[:,g].sum(axis=1)) for g in groups]
    bounds = {c:(0, spn_train_data[i].max()) for i, c in enumerate(categ_order)}
elif group_f == "mean":
    spn_train_data = [csc_matrix(ease.X[:,g].mean(axis=1)) for g in groups]
    discrete = []
elif group_f == "disjunction":
    spn_train_data = [csc_matrix(ease.X[:,g].sum(axis=1) > 0).astype(int) for g in groups]
    cat_map = {c: [0, 1] for c in categ_order}
    discrete = categ_order
elif group_f == "none":
    spn_train_data = [(ease.X * max_rating).astype(int)]
    bounds = {c:(0, max_rating) for c in categ_order}
    if not rating_used:
        cat_map = {c: [0, 1] for c in categ_order}

spn_train_data = hstack(spn_train_data).toarray()
data_handler = DataHandler(spn_train_data, categ_map=cat_map, bounds_map=bounds, discrete=discrete, feature_names=categ_order)

if spn_train_data.shape[0] > max_samples_for_spn:
    np.random.seed(fold)
    subset = np.random.choice(spn_train_data.shape[0], max_samples_for_spn, replace=False)
    spn = SPN(spn_train_data[subset, :], data_handler, learn_mspn_kwargs={"min_instances_slice": min_instances_spn})
    lls = spn.compute_ll(spn_train_data[subset, :])
else:
    spn = SPN(spn_train_data, data_handler, learn_mspn_kwargs={"min_instances_slice": min_instances_spn})
    lls = spn.compute_ll(spn_train_data)

with open(f"{help_folder}/{data_name}/{fold}/models/spn_{group_f}.pickle", "wb") as f:
    pickle.dump((spn, lls), f)