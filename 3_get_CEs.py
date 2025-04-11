import pickle
import os
import sys
from LiCE.lice.LiCE import LiCE
import numpy as np
from collections import defaultdict
import pandas as pd

data_name = sys.argv[1]
fold = int(sys.argv[2])
group_f = sys.argv[3]
rating_used = sys.argv[4] == "rating"
thresholding = sys.argv[5] == "median"
no_spn = sys.argv[5] == "no_spn"
if not thresholding and not no_spn:
    opt_coeff = float(sys.argv[5]) # 0.05
nth = sys.argv[6] == "nth"

time_limit = 600
topk = 10
help_folder = "data_splits"

os.makedirs(f"{help_folder}/{data_name}/{fold}/CEs", exist_ok=True)

df = pd.read_csv(f"{help_folder}/{data_name}/df.csv")
with open(f"{help_folder}/{data_name}/{fold}/groups.pickle", "rb") as f:
    (groups, categ_order) = pickle.load(f)

max_rating = df['rating'].max() if rating_used else 1

with open(f"{help_folder}/{data_name}/{fold}/models/ease_{'rating' if rating_used else 'binary'}.pickle", "rb") as f:
    ease = pickle.load(f)

with open(f"{help_folder}/{data_name}/{fold}/test.pickle", "rb") as f:
    test_users = pickle.load(f)

df = df[df["user_id"].isin(test_users)]

with open(f"{help_folder}/{data_name}/{fold}/models/spn_{group_f}.pickle", "rb") as f:
    spn, lls = pickle.load(f)
    if thresholding:
        median_ll = np.median(lls)
if no_spn:
    lice = LiCE(ease.B, max_rating)
else:
    lice = LiCE(ease.B, max_rating, spn, groups, group_f)
allstats = defaultdict(dict)
for user_id, rows in df.groupby(["user_id"]):
    factual = np.zeros((ease.B.shape[0],))
    for i, item in enumerate(ease.item_enc.transform(rows["item_id"])):
        factual[item] = rows.iloc[i]["rating"] / max_rating

    preds = ease.predict(df, [user_id], ease.item_enc.classes_, topk)
    for j, expl_i in enumerate(ease.item_enc.transform(preds.iloc[:3]["item_id"])):
        if nth:
            if thresholding:
                res = lice.generate_counterfactual(factual, expl_i, nth=topk+1, time_limit=time_limit, ll_threshold=median_ll)
            elif not no_spn:
                res = lice.generate_counterfactual(factual, expl_i, nth=topk+1, time_limit=time_limit, ll_opt_coefficient=opt_coeff)
            else:
                res = lice.generate_counterfactual(factual, expl_i, nth=topk+1, time_limit=time_limit)
        else:
            if thresholding:
                res = lice.generate_counterfactual(factual, expl_i, score=preds.iloc[-1]["score"], time_limit=time_limit, ll_threshold=median_ll)
            elif not no_spn:
                res = lice.generate_counterfactual(factual, expl_i, score=preds.iloc[-1]["score"], time_limit=time_limit, ll_opt_coefficient=opt_coeff)
            else:
                res = lice.generate_counterfactual(factual, expl_i, score=preds.iloc[-1]["score"], time_limit=time_limit)

        scores = res[0] @ ease.B
        true_position = np.sum(scores > scores[expl_i])
        stats = lice.stats
        stats["factual_position"] = j + 1
        stats["true_cf_position"] = true_position + 1
        stats["factual"] = factual
        stats["counterfactual"] = res[0]
        cf = np.array(res[0])
        if group_f == "sum":
            spn_cf = [cf[g].sum(axis=1) for g in groups]
            spn_f = [factual[g].sum(axis=1) for g in groups]
        elif group_f == "mean":
            spn_cf = [cf[g].mean(axis=1) for g in groups]
            spn_f = [factual[g].mean(axis=1) for g in groups]
        elif group_f == "disjunction":
            spn_cf = [(cf[g].sum(axis=1) > 0).astype(int) for g in groups]
            spn_f = [(factual[g].sum(axis=1) > 0).astype(int) for g in groups]
        elif group_f == "none":
            spn_cf = cf * max_rating
            spn_f = factual * max_rating
        stats["couterfactual_ll"] = spn.compute_ll(spn_cf)
        stats["factual_ll"] = spn.compute_ll(spn_f)
        allstats[user_id][expl_i](stats)
