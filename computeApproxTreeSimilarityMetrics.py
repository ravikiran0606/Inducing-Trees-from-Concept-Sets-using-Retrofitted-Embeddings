import argparse
import json
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import matplotlib.pyplot as plt
import time
import rltk
import numpy as np
from scipy import stats
import zss

def createGraphFromTruth(ground_truth, addRoot=True):
    node_list = []
    edge_list = []
    artificial_edge_list = []
    with open(ground_truth, "r") as f:
        for cur_line in f:
            cur_dict = json.loads(cur_line)
            key = list(cur_dict.keys())[0]
            node_list.append(key)

            parents = list(cur_dict.values())[0]["parent"]
            children = list(cur_dict.values())[0]["children"]

            # Add artificial root node to the existing topmost nodes
            if addRoot and len(parents) == 0:
                artificial_edge_list.append(("root", key))

            for cur_n in parents:
                edge_list.append((cur_n, key))
            for cur_n in children:
                edge_list.append((key, cur_n))

    cur_graph = nx.Graph()
    cur_graph.add_nodes_from(node_list)
    cur_graph.add_edges_from(edge_list)
    cur_graph.add_edges_from(artificial_edge_list)
    return cur_graph, artificial_edge_list

def str2bool(input_str):
    if input_str is None:
        return None
    elif input_str.lower() == "true":
        return True
    elif input_str.lower() == "false":
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bmf", "--baseline_mst_file", type=str, default=None,
                        help="Baseline Minimum Spanning Tree file")
    parser.add_argument("-cmf", "--comparison_mst_file", type=str, default=None,
                        help="Comparison Minimum Spanning Tree file")
    parser.add_argument("-gt", "--ground_truth", type=str, default=None,
                        help="Ground truth Lexicon with children and parents information")
    parser.add_argument("-gtsns", "--gt_sample_nodes_set", type=str, default=None,
                        help="Sample nodes set for the Ground truth Lexicon which is used for computing the similarity")
    parser.add_argument("-sp", "--save_plots", type=str, default="False",
                        help="Whether to save the plots of the ground truth and constructed trees")
    parser.add_argument("-pd", "--plot_dir", type=str, default="plots/",
                        help="Directory to store plots")
    args = parser.parse_args()

    baseline_mst_file = args.baseline_mst_file
    comparison_mst_file = args.comparison_mst_file
    ground_truth = args.ground_truth
    save_plots = str2bool(args.save_plots)
    plot_dir = args.plot_dir

    with open(args.gt_sample_nodes_set, "r") as f:
        gt_sample_nodes_set = [x.strip() for x in f]
        print("Number of Sample Nodes for Evaluation = ", len(gt_sample_nodes_set))

    start_time = time.time()

    # Ground Truth Tree construction:
    gt_graph, artificial_edge_list = createGraphFromTruth(ground_truth)

    if save_plots:
        # Save the Plot:
        gt_file_name = ground_truth.split("/")[-1].split(".")[0]
        plt.figure(figsize=(200, 200))
        nx.draw(gt_graph, with_labels=True)
        plt.savefig(plot_dir + gt_file_name + ".png")

    gt_tree = nx.bfs_tree(gt_graph, source="root")
    gt_graph = nx.Graph(gt_tree)

    # Baseline Method Generated Tree construction:
    base_gen_graph = nx.read_weighted_edgelist(baseline_mst_file)
    base_gen_graph.add_edges_from(artificial_edge_list)

    if save_plots:
        # Save the Plot:
        base_gen_file_name = baseline_mst_file.split("/")[-1].split(".")[0]
        plt.figure(figsize=(200, 200))
        nx.draw(base_gen_graph, with_labels=True)
        plt.savefig(plot_dir + base_gen_file_name + ".png")

    base_gen_tree = nx.bfs_tree(base_gen_graph, source="root")
    base_gen_graph = nx.Graph(base_gen_tree)

    # Comparison Method Generated Tree construction:
    comp_gen_graph = nx.read_weighted_edgelist(comparison_mst_file)
    comp_gen_graph.add_edges_from(artificial_edge_list)

    if save_plots:
        # Save the Plot:
        comp_gen_file_name = comparison_mst_file.split("/")[-1].split(".")[0]
        plt.figure(figsize=(200, 200))
        nx.draw(comp_gen_graph, with_labels=True)
        plt.savefig(plot_dir + comp_gen_file_name + ".png")

    comp_gen_tree = nx.bfs_tree(comp_gen_graph, source="root")
    comp_gen_graph = nx.Graph(comp_gen_tree)

    # Computing the Jaccard Similarity between their shortest paths:
    baseline_jaccard_sims = []
    comparison_jaccard_sims = []
    for cur_node in gt_sample_nodes_set:
        gt_sp = nx.shortest_path(gt_graph, source=cur_node, weight=None)
        base_gen_sp = nx.shortest_path(base_gen_graph, source=cur_node, weight=None)
        comp_gen_sp = nx.shortest_path(comp_gen_graph, source=cur_node, weight=None)

        for cur_key in gt_sp.keys():
            if cur_key == cur_node:
                continue

            try:
                gt_path = set(gt_sp[cur_key])
            except:
                gt_path = set()

            try:
                base_gen_path = set(base_gen_sp[cur_key])
            except:
                base_gen_path = set()

            try:
                comp_gen_path = set(comp_gen_sp[cur_key])
            except:
                comp_gen_path = set()

            base_jaccard_sim = rltk.jaccard_index_similarity(gt_path, base_gen_path)
            baseline_jaccard_sims.append(base_jaccard_sim)

            comp_jaccard_sim = rltk.jaccard_index_similarity(gt_path, comp_gen_path)
            comparison_jaccard_sims.append(comp_jaccard_sim)

    baseline_JACCARD = np.mean(baseline_jaccard_sims) * 100
    comparison_JACCARD = np.mean(comparison_jaccard_sims) * 100

    print("\nBaseline Method Jaccard Similarity metric = ", baseline_JACCARD)
    print("Comparison Method Jaccard Similarity metric = ", comparison_JACCARD)
    print("MAP significance test = ", stats.ttest_rel(baseline_jaccard_sims, comparison_jaccard_sims), "\n")

    cur_time = time.time()
    seconds_elapsed = cur_time-start_time
    print("Seconds Elapsed = ", seconds_elapsed)












