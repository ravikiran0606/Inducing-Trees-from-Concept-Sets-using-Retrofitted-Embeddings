import argparse
import numpy as np
import string
import pickle as pkl
import json
import jsonlines
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

    unique_uris = sorted(list(unique_uris))
    lexicon_eval = list()
    for cur_uri in unique_uris:
        cur_dict = {}
        cur_dict[cur_uri] = dict()
        cur_dict[cur_uri]["parent"] = parents_dict[cur_uri]
        cur_dict[cur_uri]["children"] = children_dict[cur_uri]
        lexicon_eval.append(cur_dict)

    return lexicon_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-st", "--src_taxonomy", type=str, default=None,
                        help="Source Taxonomy which is used as training dataset")
    parser.add_argument("-olf", "--out_lexicon_file", type=str, default=None,
                        help="Output Lexicon with children and parents information for evaluation")
    args = parser.parse_args()

    src_taxonomy = args.src_taxonomy
    out_lexicon_file = args.out_lexicon_file

    lexicon_eval = createParentChildDicts(src_taxonomy)

    with jsonlines.open(out_lexicon_file, "w") as f:
        f.write_all(lexicon_eval)





