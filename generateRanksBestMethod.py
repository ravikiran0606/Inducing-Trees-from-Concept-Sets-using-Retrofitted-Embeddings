import numpy as np
import sys
import json
import math
import jsonlines
import argparse
from scipy.stats import rankdata
import pickle as pkl
sys.path.append("..")
sys.path.append(".")
from modelUtils.LanguageModels import WordEmbeddingsModel

def rerankListTopK(cur_node, cur_neighbors_list, embeddings_model, classifier_model, vector_feat, top_k=1):
    # Create the instances:
    X_test = []
    cur_node_embed = embeddings_model.getEmbeddingsForWord(cur_node)
    for cur_n, cur_n_prob in cur_neighbors_list:
        cur_n_embed = embeddings_model.getEmbeddingsForWord(cur_n)
        if vector_feat == "vector_concat":
            cur_x = np.concatenate((cur_node_embed, cur_n_embed))
        elif vector_feat == "vector_diff":
            cur_x = (cur_node_embed - cur_n_embed)
        else:
            cur_x = None
        X_test.append(cur_x)

    X_test = np.array(X_test)
    y_pred_probs = classifier_model.predict_proba(X_test)
    y_pred_probs_pos = np.array([x[1] for x in y_pred_probs])
    new_ranks = rankdata(-y_pred_probs_pos, method="ordinal")
    topk_idx = np.where(new_ranks<=top_k)[0]

    topk_nodes = [None] * top_k
    for i in topk_idx:
        topk_nodes[new_ranks[i] - 1] = cur_neighbors_list[i]

    new_neighbors_list = cur_neighbors_list.copy()
    for cur_n_tuple in topk_nodes:
        new_neighbors_list.remove(cur_n_tuple)
    new_neighbors_list = topk_nodes + new_neighbors_list

    return new_neighbors_list

def computeRanks_with_classifier(lexicon_evaluation, embeddings_file_path, embeddings_dims, classifier_model_path):

    # Load the classifier model
    with open(classifier_model_path, "rb") as f:
        classifier_model = pkl.load(f)
        pdict = {'verbose': 0, 'n_jobs': -2}
        classifier_model.set_params(**pdict)

    # Load the embeddings model
    embeddings_model = WordEmbeddingsModel(embeddings_file_path, embeddings_dims)

    gpt_fobj = open(lexicon_evaluation, "r")

    rank_results = []
    index_cnt = 0

    for cur_line in gpt_fobj:
        cur_dict = json.loads(cur_line)
        cur_query = list(cur_dict.keys())[0]

        index_cnt += 1
        if index_cnt % 100 == 0:
            print("Progress count = ", index_cnt)

        try:
            word_neighbors_all = embeddings_model.getMostSimilarWords(cur_query, -1, probs=True)
            word_neighbors_with_probs = rerankListTopK(cur_query, word_neighbors_all, embeddings_model, classifier_model, "vector_diff", top_k=3)

            word_neighbors = []
            word_neighbors_scores = []
            for x in word_neighbors_with_probs:
                word_neighbors.append(x[0])
                word_neighbors_scores.append(round(x[1], 5))

            cur_rank_dict = dict()
            cur_rank_dict[cur_query] = [word_neighbors, word_neighbors_scores]
            rank_results.append(cur_rank_dict)
        except:
            pass

    return rank_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lte", "--lexicon_to_evaluate", type=str, default=None,
                        help="Lexicon to evaluate")
    parser.add_argument("-ep", "--embeddings_path", type=str, default=None,
                        help="Embeddings path")
    parser.add_argument("-ed", "--embeddings_dims", type=int, default=300,
                        help="Embeddings dimensions")
    parser.add_argument("-cmp", "--classifier_model_path", type=str, default=None,
                        help="Parent Classifier model path")
    parser.add_argument("-orf", "--out_ranks_file", type=str, default=None,
                        help="Output ranks file")
    args = parser.parse_args()

    lexicon_evaluation = args.lexicon_to_evaluate
    embeddings_file_path = args.embeddings_path
    classifier_model_path = args.classifier_model_path
    out_ranks_file = args.out_ranks_file
    embeddings_dims = args.embeddings_dims

    rank_list = computeRanks_with_classifier(lexicon_evaluation, embeddings_file_path, embeddings_dims, classifier_model_path)

    with jsonlines.open(out_ranks_file, "w") as f:
        f.write_all(rank_list)









