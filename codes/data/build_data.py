import argparse
import array
import gzip
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def dataset_merge_and_split(path, core):
    if not os.path.exists(folder + "%d-core" % core):
        os.makedirs(folder + "%d-core" % core)

    df = pd.read_csv(path + "/train.csv", index_col=None, usecols=None)
    # Construct matrix
    ui = defaultdict(list)
    for _, row in df.iterrows():
        user, item = int(row["userID"]), int(row["itemID"])
        ui[user].append(item)

    df = pd.read_csv(path + "/test.csv", index_col=None, usecols=None)
    for _, row in df.iterrows():
        user, item = int(row["userID"]), int(row["itemID"])
        ui[user].append(item)

    train_json = {}
    val_json = {}
    test_json = {}
    for u, items in ui.items():
        if len(items) < 10:
            testval = np.random.choice(len(items), 2, replace=False)
        else:
            testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

        test = testval[: len(testval) // 2]
        val = testval[len(testval) // 2 :]
        train = [i for i in list(range(len(items))) if i not in testval]
        train_json[u] = [items[idx] for idx in train]
        val_json[u] = [items[idx] for idx in val.tolist()]
        test_json[u] = [items[idx] for idx in test.tolist()]

    with open(path + "/5-core/train.json", "w") as f:
        json.dump(train_json, f)
    with open(path + "/5-core/val.json", "w") as f:
        json.dump(val_json, f)
    with open(path + "/5-core/test.json", "w") as f:
        json.dump(test_json, f)


def load_textual_image_features(data_path):
    import json
    import os

    from gensim.models.doc2vec import Doc2Vec

    asin_dict = json.load(open(os.path.join(data_path, "asin_sample.json"), "r"))

    # Prepare textual feture data.
    doc2vec_model = Doc2Vec.load(os.path.join(data_path, "doc2vecFile"))
    vis_vec = np.load(
        os.path.join(data_path, "image_feature.npy"), allow_pickle=True
    ).item()
    text_vec = {}
    for asin in asin_dict:
        text_vec[asin] = doc2vec_model.docvecs[asin]

    all_dict = {}
    num_items = 0
    filename = data_path + "/train.csv"
    df = pd.read_csv(filename, index_col=None, usecols=None)
    for _, row in df.iterrows():
        asin, i = row["asin"], int(row["itemID"])
        all_dict[i] = asin
        num_items = max(num_items, i)
    filename = data_path + "/test.csv"
    df = pd.read_csv(filename, index_col=None, usecols=None)
    for _, row in df.iterrows():
        asin, i = row["asin"], int(row["itemID"])
        all_dict[i] = asin
        num_items = max(num_items, i)

    t_features = []
    v_features = []
    for i in range(num_items + 1):
        t_features.append(text_vec[all_dict[i]])
        v_features.append(vis_vec[all_dict[i]])

    np.save(data_path + "/text_feat.npy", np.asarray(t_features, dtype=np.float32))
    np.save(data_path + "/image_feat.npy", np.asarray(v_features, dtype=np.float32))


parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--name",
    nargs="?",
    default="MenClothing",
    help="Choose a dataset folder from {MenClothing, WomenClothing, Beauty, Toys_and_Games}.",
)

np.random.seed(123)

args = parser.parse_args()
folder = args.name + "/"
name = args.name
core = 5
if folder in ["MenClothing/", "WomenClothing/"]:
    dataset_merge_and_split(folder, core)
    load_textual_image_features(folder)
else:
    bert_path = "sentence-transformers/stsb-roberta-large"
    bert_model = SentenceTransformer(bert_path)

    if not os.path.exists(folder + "%d-core" % core):
        os.makedirs(folder + "%d-core" % core)

    def parse(path):
        g = gzip.open(path, "r")
        for line in g:
            yield json.dumps(eval(line))

    print("----------parse metadata----------")
    if not os.path.exists(folder + "meta-data/meta.json"):
        with open(folder + "meta-data/meta.json", "w") as f:
            for line in parse(folder + "meta-data/" + "meta_%s.json.gz" % (name)):
                f.write(line + "\n")

    print("----------parse data----------")
    if not os.path.exists(folder + "meta-data/%d-core.json" % core):
        with open(folder + "meta-data/%d-core.json" % core, "w") as f:
            for line in parse(
                folder + "meta-data/" + "reviews_%s_%d.json.gz" % (name, core)
            ):
                f.write(line + "\n")

    print("----------load data----------")
    jsons = []
    for line in open(folder + "meta-data/%d-core.json" % core).readlines():
        jsons.append(json.loads(line))

    print("----------Build dict----------")
    items = set()
    users = set()
    for j in jsons:
        items.add(j["asin"])
        users.add(j["reviewerID"])
    print("n_items:", len(items), "n_users:", len(users))

    item2id = {}
    with open(folder + "%d-core/item_list.txt" % core, "w") as f:
        for i, item in enumerate(items):
            item2id[item] = i
            f.writelines(item + "\t" + str(i) + "\n")

    user2id = {}
    with open(folder + "%d-core/user_list.txt" % core, "w") as f:
        for i, user in enumerate(users):
            user2id[user] = i
            f.writelines(user + "\t" + str(i) + "\n")

    ui = defaultdict(list)
    review2id = {}
    review_text = {}
    ratings = {}
    with open(folder + "%d-core/review_list.txt" % core, "w") as f:
        for j in jsons:
            u_id = user2id[j["reviewerID"]]
            i_id = item2id[j["asin"]]
            ui[u_id].append(i_id)  # ui[u_id].append(i_id)
            review_text[len(review2id)] = j["reviewText"].replace("\n", " ")
            ratings[len(review2id)] = int(j["overall"])
            f.writelines(str((u_id, i_id)) + "\t" + str(len(review2id)) + "\n")
            review2id[u_id, i_id] = len(review2id)
    with open(folder + "%d-core/user-item-dict.json" % core, "w") as f:
        f.write(json.dumps(ui))
    with open(folder + "%d-core/rating-dict.json" % core, "w") as f:
        f.write(json.dumps(ratings))

    review_texts = []
    with open(folder + "%d-core/review_text.txt" % core, "w") as f:
        for i, j in review2id:
            f.write(review_text[review2id[i, j]] + "\n")
            review_texts.append(review_text[review2id[i, j]] + "\n")
    review_embeddings = bert_model.encode(review_texts)
    assert review_embeddings.shape[0] == len(review2id)
    np.save(folder + "review_feat.npy", review_embeddings)

    print("----------Split Data----------")
    train_json = {}
    val_json = {}
    test_json = {}
    for u, items in ui.items():
        if len(items) < 10:
            testval = np.random.choice(len(items), 2, replace=False)
        else:
            testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

        test = testval[: len(testval) // 2]
        val = testval[len(testval) // 2 :]
        train = [i for i in list(range(len(items))) if i not in testval]
        train_json[u] = [items[idx] for idx in train]
        val_json[u] = [items[idx] for idx in val.tolist()]
        test_json[u] = [items[idx] for idx in test.tolist()]

    with open(folder + "%d-core/train.json" % core, "w") as f:
        json.dump(train_json, f)
    with open(folder + "%d-core/val.json" % core, "w") as f:
        json.dump(val_json, f)
    with open(folder + "%d-core/test.json" % core, "w") as f:
        json.dump(test_json, f)

    jsons = []
    with open(folder + "meta-data/meta.json", "r") as f:
        for line in f.readlines():
            jsons.append(json.loads(line))

    print("----------Text Features----------")
    raw_text = {}
    for _json in jsons:
        if _json["asin"] in item2id:
            string = " "
            if "categories" in _json:
                for cates in _json["categories"]:
                    for cate in cates:
                        string += cate + " "
            if "title" in _json:
                string += _json["title"]
            if "brand" in _json:
                string += _json["title"]
            if "description" in _json:
                string += _json["description"]
            raw_text[item2id[_json["asin"]]] = string.replace("\n", " ")
    texts = []
    with open(folder + "%d-core/raw_text.txt" % core, "w") as f:
        for i in range(len(item2id)):
            f.write(raw_text[i] + "\n")
            texts.append(raw_text[i] + "\n")
    sentence_embeddings = bert_model.encode(texts)
    assert sentence_embeddings.shape[0] == len(item2id)
    np.save(folder + "text_feat.npy", sentence_embeddings)

    print("----------Image Features----------")

    def readImageFeatures(path):
        f = open(path, "rb")
        while True:
            asin = f.read(10).decode("UTF-8")
            if asin == "":
                break
            a = array.array("f")
            a.fromfile(f, 4096)
            yield asin, a.tolist()

    data = readImageFeatures(folder + "meta-data/" + "image_features_%s.b" % name)
    feats = {}
    avg = []
    for d in data:
        if d[0] in item2id:
            feats[int(item2id[d[0]])] = d[1]
            avg.append(d[1])
    avg = np.array(avg).mean(0).tolist()

    ret = []
    for i in range(len(item2id)):
        if i in feats:
            ret.append(feats[i])
        else:
            ret.append(avg)

    assert len(ret) == len(item2id)
    np.save(folder + "image_feat.npy", np.array(ret))
