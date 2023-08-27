import pandas as pd
import json
import os
from tqdm import tqdm
from collections import defaultdict

def load_dataset_texts(root_dir = "./MASSIVE/", test_set=False):
    def load_jsonl(filepath):
        with open(filepath, 'r', encoding='utf-8') as json_file:
            json_list = list(json_file)

        jsons = []

        for json_str in json_list:
            jsons.append(json.loads(json_str))

        return jsons

    train_filelist = os.listdir(f"{root_dir}/train_data")
    test_filelist = os.listdir(f"{root_dir}/test_data")


    train_texts = []
    train_labels = []

    for item in train_filelist:
        data = load_jsonl(f"{root_dir}/train_data/{item}")

        for example in tqdm(data, f"loading {item}"):
            train_texts.append(example["utt"])
            train_labels.append(example["intent"])

    if not test_set:
        return train_texts, train_labels

    test_texts = []

    for item in test_filelist:
        data = load_jsonl(f"{root_dir}/test_data/{item}")

        for example in tqdm(data, f"loading {item}"):
            test_texts.append(example["utt"])

    return (train_texts, train_labels), (test_texts)

def load_indoml_data(_dir, with_scenario=True):
    train_cols = ['indoml_id', 'utt', 'locale']

    if with_scenario:
        train_cols += ["scenario"]

    train_cols += ['intent']

    train_data = defaultdict(list)
    test_data = defaultdict(list)
    valid_data = []

    with open(_dir + "/massive_train.data", "r", encoding="utf-8") as train_data_file:
        json_list = list(train_data_file)

    with open(_dir + "/massive_train.solution", "r", encoding="utf-8") as train_data_file:
        train_data["intent"] = [json.loads(item)["intent"] for item in train_data_file]

    for item in tqdm(json_list, desc="loading train data"):
        item = json.loads(item)
        for col in train_cols[:-1]:
            train_data[col].append(item[col])

    with open(_dir + "/massive_valid.data", "r", encoding="utf-8") as train_data_file:
        json_list = list(train_data_file)

    for item in tqdm(json_list, desc="loading test data"):
        item = json.loads(item)
        for col in train_cols[:-1]:
            test_data[col].append(item[col])

    return pd.DataFrame(train_data), pd.DataFrame(test_data)