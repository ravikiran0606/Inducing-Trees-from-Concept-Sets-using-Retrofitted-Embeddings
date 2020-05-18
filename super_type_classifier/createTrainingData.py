import argparse
import numpy as np
import string
import pickle as pkl
import json
import random

def getURI(input_category):
    input_category = input_category.split("_")[-1]

    result = input_category.lower()
    for cur_punch in string.punctuation:
        result = result.replace(cur_punch, " ")
    uri = "_".join(result.split())
    return uri


def createParentChildDicts(input_taxonomy):
    parents_dict = {}
    children_dict = {}
    unique_uris = set()

    with open(input_taxonomy, "r") as f:
        for cur_line in f:
            cur_line = cur_line.lower()
            tokens_list = cur_line.split(" > ")

            prev_token_uri = None
            for token in tokens_list:
                cur_token_uri = getURI(token.strip())
                if len(cur_token_uri) == 0:
                    continue

                unique_uris.add(cur_token_uri)

                if parents_dict.get(cur_token_uri) is None:
                    parents_dict[cur_token_uri] = set()

                if children_dict.get(cur_token_uri) is None:
                    children_dict[cur_token_uri] = set()

                if prev_token_uri is not None and prev_token_uri != cur_token_uri:
                    children_dict[prev_token_uri].add(cur_token_uri)
                    parents_dict[cur_token_uri].add(prev_token_uri)

                prev_token_uri = cur_token_uri

    for key, val in children_dict.items():
        children_dict[key] = list(val)

    for key, val in parents_dict.items():
        parents_dict[key] = list(val)

    return parents_dict, children_dict


def createTrainingData(word_list, parents_dict, child_dict):
    data = []
    for key in word_list:
        cur_node = key
        cur_parents = parents_dict[key]

        for cur_parent in cur_parents:
            cur_siblings = child_dict[cur_parent].copy()
            cur_siblings.remove(cur_node)

            if len(cur_siblings) != 0:
                cur_sibling = np.random.choice(cur_siblings, replace=False)

                # Positive Example
                data.append([cur_node, cur_parent, 1])

                # Negative Example
                data.append([cur_node, cur_sibling, 0])

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-st", "--src_taxonomy", type=str, default=None,
                        help="Source Taxonomy which is used as training dataset")
    parser.add_argument("-of", "--out_file", type=str, default=None,
                        help="Output Training Dataset file name")

    args = parser.parse_args()

    src_taxonomy = args.src_taxonomy
    out_file = args.out_file

    parents_dict, child_dict = createParentChildDicts(src_taxonomy)
    data_nodes = []
    for key in parents_dict.keys():
        if len(parents_dict[key]) !=0:
            data_nodes.append(key)

    data_full = createTrainingData(data_nodes, parents_dict, child_dict)

    with open(out_file, "wb") as f:
        pkl.dump(data_full, f)





