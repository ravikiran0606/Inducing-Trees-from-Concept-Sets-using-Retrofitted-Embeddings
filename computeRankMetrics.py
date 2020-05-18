import argparse
import json
import numpy as np
import math
from scipy import stats

def computeMAP(rank_list):
    rank_list.sort()
    average_precision = 0
    i = 0
    for cur_rank in rank_list:
        i = i + 1
        average_precision += (float(i) / cur_rank)

    average_precision /= len(rank_list)
    return average_precision

def computeNDCG(rank_list):
    rank_list.sort()
    idcg_score = 0
    dcg_score = 0

    i = 0
    for cur_rank in rank_list:
        i += 1
        dcg_score += 1.0 / math.log(cur_rank + 1, 2)
        idcg_score += 1.0 / math.log(i + 1, 2)

    ndcg_score = dcg_score / idcg_score
    return ndcg_score

def meanRankMetric(ground_truth_lexicon, baseline_ranks_file, comparison_ranks_file):
    baseline_ndcg_results = {}
    baseline_map_results = {}

    comparison_ndcg_results = {}
    comparison_map_results = {}

    with open(baseline_ranks_file, "r") as f:
        for cur_line in f:
            cur_dict = json.loads(cur_line)
            cur_query = list(cur_dict.keys())[0]
            word_neighbors = list(cur_dict.values())[0][0]
            gt_neighbors_list = ground_truth_lexicon[cur_query]["parent"] + ground_truth_lexicon[cur_query]["children"]

            cur_ranks_list = []
            for cur_neighbor in gt_neighbors_list:
                try:
                    cur_rank = word_neighbors.index(cur_neighbor) + 1
                    cur_ranks_list.append(cur_rank)
                except:
                    pass

            baseline_ndcg_results[cur_query] = computeNDCG(cur_ranks_list)
            baseline_map_results[cur_query] = computeMAP(cur_ranks_list)

    with open(comparison_ranks_file, "r") as f:
        for cur_line in f:
            cur_dict = json.loads(cur_line)
            cur_query = list(cur_dict.keys())[0]
            word_neighbors = list(cur_dict.values())[0][0]
            gt_neighbors_list = ground_truth_lexicon[cur_query]["parent"] + ground_truth_lexicon[cur_query]["children"]

            cur_ranks_list = []
            for cur_neighbor in gt_neighbors_list:
                try:
                    cur_rank = word_neighbors.index(cur_neighbor) + 1
                    cur_ranks_list.append(cur_rank)
                except:
                    pass

            comparison_ndcg_results[cur_query] = computeNDCG(cur_ranks_list)
            comparison_map_results[cur_query] = computeMAP(cur_ranks_list)

    # Sort based on keys:
    baseline_ndcg_results = dict(sorted(baseline_ndcg_results.items(), key=lambda x: x[0]))
    baseline_map_results = dict(sorted(baseline_map_results.items(), key=lambda x: x[0]))
    comparison_ndcg_results = dict(sorted(comparison_ndcg_results.items(), key=lambda x: x[0]))
    comparison_map_results = dict(sorted(comparison_map_results.items(), key=lambda x: x[0]))

    baseline_map_list = list(baseline_map_results.values())
    baseline_ndcg_list = list(baseline_ndcg_results.values())
    baseline_MAP = np.mean(baseline_map_list) * 100
    baseline_NDCG = np.mean(baseline_ndcg_list) * 100

    comparison_map_list = list(comparison_map_results.values())
    comparison_ndcg_list = list(comparison_ndcg_results.values())
    comparison_MAP = np.mean(comparison_map_list) * 100
    comparison_NDCG = np.mean(comparison_ndcg_list) * 100

    print("\nBaseline Method MAP metric = ", baseline_MAP)
    print("Comparison Method MAP metric = ", comparison_MAP)
    print("MAP significance test = ", stats.ttest_rel(baseline_map_list, comparison_map_list), "\n")

    print("Baseline Method NDCG metric = ", baseline_NDCG)
    print("Comparison Method NDCG metric = ", comparison_NDCG)
    print("nDCG significance test = ", stats.ttest_rel(baseline_ndcg_list, comparison_ndcg_list), "\n")


def constructFromJL(jl_file):
    result_dict = {}
    with open(jl_file, "r") as f:
        for cur_line in f:
            cur_dict = json.loads(cur_line)
            key = list(cur_dict.keys())[0]
            val = list(cur_dict.values())[0]
            result_dict[key] = val

    return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--ground_truth", type=str, default=None,
                        help="Ground truth Lexicon with children and parents information")
    parser.add_argument("-brf", "--baseline_ranks_file", type=str, default=None,
                        help="Ranks file of the Baseline method")
    parser.add_argument("-crf", "--comparison_ranks_file", type=str, default=None,
                        help="Ranks file of the Comparison method")


    args = parser.parse_args()

    ground_truth = args.ground_truth
    baseline_ranks_file = args.baseline_ranks_file
    comparison_ranks_file = args.comparison_ranks_file

    ground_truth_lexicon = constructFromJL(ground_truth)
    meanRankMetric(ground_truth_lexicon, baseline_ranks_file, comparison_ranks_file)












