import pandas as pd
import json
import os
import csv
from collections import defaultdict
from datasets import load_dataset

def get_data(data_name: str):
    categs = defaultdict(list)
    categ_map = {}
    if data_name == "yelp":
        items_to_prune = set()
        items_to_keep = set()
        with open('data/yelp/yelp_academic_dataset_business.json', 'r') as file:
            for line in file:
                business = json.loads(line)
                if business['review_count'] < 20: # -> 61k remains
                # if business['review_count'] < 30: # -> 45k remains
                # if business['review_count'] < 40: # -> 35k remains
                # if business['review_count'] < 50: # -> 29k remains
                # if business['review_count'] < 100: # -> 14k remains
                    # items_to_prune.add(business["business_id"])
                    continue
                else:
                    items_to_keep.add(business["business_id"])
                if "categories" not in business or business['categories'] is None:
                    cats = []
                else:
                    cats = [cat.strip() for cat in business['categories'].split(",")]
                    for cat in cats:
                        categs[cat].append(business["business_id"])
                categ_map[business['business_id']] = cats

        reviews = []
        with open('data/yelp/yelp_academic_dataset_review.json', 'r') as file:
            for line in file:
                review = json.loads(line)
                # if review['business_id'] not in items_to_prune:
                if review['business_id'] in items_to_keep:
                    reviews.append({
                        'rating': review['stars'],
                        'user_id': review['user_id'],
                        'item_id': review['business_id']
                    })

        df = pd.DataFrame(reviews)

    elif data_name == "netflix":
        data_dir = "data/netflix/training_set"
        data = []

        with open("data/netflix/movie_titles.txt", 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                item_id = row[0]
                year = row[1]
                try:
                    if year != "NULL":
                    # if int(year) > 2003:
                        categs[year].append(item_id)
                        categ_map[item_id] = [year]
                except ValueError: # when year cannot be parsed
                    pass

        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                item_id = next(reader)[0].split(":")[0]  # Extract item_id from the first line
                if item_id not in categ_map:
                    continue
                for row in reader:
                    if len(row) >= 2:  # Ensure there are at least two columns
                        user_id = row[0]
                        rating = row[1]
                        data.append({'user_id': user_id, 'item_id': item_id, 'rating': int(rating)})

        df = pd.DataFrame(data)

    elif data_name == "amazon":
        metadata = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Books", split="full", trust_remote_code=True)
        for d in metadata:
            for cat in d["categories"]:
                categs[cat].append(d['parent_asin'])

        data = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "0core_rating_only_Books", trust_remote_code=True)
        dataset = []
        for d in data["full"]:
            print(d)
            dataset.append({
                'user_id': d['user_id'],
                'item_id': d['parent_asin'],
                'rating': d['rating'],
            })

        df = pd.DataFrame(dataset)

    return df, categs
