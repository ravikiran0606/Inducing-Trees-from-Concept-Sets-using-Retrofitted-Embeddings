import argparse
import networkx as nx
import matplotlib.pyplot as plt
import json

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
    parser.add_argument("-gt", "--ground_truth", type=str, default=None,
                        help="Ground truth Lexicon with children and parents information")
    parser.add_argument("-mf", "--mst_file", type=str, default=None,
                        help="Input Maximum Spanning Tree file")
    parser.add_argument("-sn", "--source_node", type=str, default=None,
                        help="Source node to run the BFS")
    parser.add_argument("-dpt", "--directed_plot", type=str, default="False",
                        help="Whether to produce directed tree plots or not")
    parser.add_argument("-md", "--max_depth", type=int, default=2,
                        help="Maximum depth of the BFS")
    parser.add_argument("-pd", "--plot_dim", type=int, default=10,
                        help="Square Plot dimensions (dim X dim)")
    parser.add_argument("-pdir", "--plot_dir", type=str, default="plots/",
                        help="Directory to store plots")
    args = parser.parse_args()

    mst_file = args.mst_file
    ground_truth = args.ground_truth
    source_node = args.source_node
    max_depth = args.max_depth
    plot_dim = args.plot_dim
    plot_dir = args.plot_dir
    directed_plot = str2bool(args.directed_plot)

    # Ground Truth Tree Plot:
    gt_graph, artificial_edge_list = createGraphFromTruth(ground_truth)
    gt_short_tree = nx.bfs_tree(gt_graph, source=source_node, depth_limit=max_depth)
    if directed_plot:
        gt_short_graph = gt_short_tree
    else:
        gt_short_graph = nx.Graph(gt_short_tree)


    plt.figure(figsize=(plot_dim, plot_dim))
    nx.draw(gt_short_graph, with_labels=True)
    gt_file_name = ground_truth.split("/")[-1].split(".")[0] + "_with_" + source_node
    plt.savefig(plot_dir + gt_file_name + ".png")

    # Generated Tree Plot:
    gen_graph = nx.read_weighted_edgelist(mst_file)
    gen_graph.add_edges_from(artificial_edge_list)
    short_tree = nx.bfs_tree(gen_graph, source=source_node, depth_limit=max_depth)
    if directed_plot:
        short_graph = short_tree
    else:
        short_graph = nx.Graph(short_tree)


    plt.figure(figsize=(plot_dim, plot_dim))
    nx.draw(short_graph, with_labels=True)
    gen_file_name = mst_file.split("/")[-1].split(".")[0] + "_with_" + source_node
    plt.savefig(plot_dir + gen_file_name + ".png")












