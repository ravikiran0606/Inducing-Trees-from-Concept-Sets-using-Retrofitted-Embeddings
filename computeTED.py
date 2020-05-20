import argparse
import json
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import matplotlib.pyplot as plt
import time
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

def removeCost(cur_node):
    return 1

def insertCost(cur_node):
    return 1

def updateCost(cur_node1, cur_node2):
    return 1

def str2bool(input_str):
    if input_str is None:
        return None
    elif input_str.lower() == "true":
        return True
    elif input_str.lower() == "false":
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mf", "--mst_file", type=str, default=None,
                        help="Input Maximum Spanning Tree file")
    parser.add_argument("-gt", "--ground_truth", type=str, default=None,
                        help="Ground truth Lexicon with children and parents information")
    parser.add_argument("-sp", "--save_plots", type=str, default="False",
                        help="Whether to save the plots of the ground truth and constructed trees")
    parser.add_argument("-pd", "--plot_dir", type=str, default="plots/",
                        help="Directory to store plots")
    args = parser.parse_args()

    mst_file = args.mst_file
    ground_truth = args.ground_truth
    save_plots = str2bool(args.save_plots)
    plot_dir = args.plot_dir

    start_time = time.time()

    # Ground Truth Tree construction:
    gt_graph, artificial_edge_list = createGraphFromTruth(ground_truth)

    if save_plots:
        # Save the Plot:
        gt_file_name = ground_truth.split("/")[-1].split(".")[0]
        plt.figure(figsize=(300, 300))
        nx.draw(gt_graph, with_labels=True)
        plt.savefig(plot_dir + gt_file_name + ".png")

    gt_tree = nx.bfs_tree(gt_graph, source="root")
    gt_nodes_dict = {}
    for edge in gt_tree.edges():
        if edge[0] not in gt_nodes_dict:
            gt_nodes_dict[edge[0]] = zss.Node(edge[0])
        if edge[1] not in gt_nodes_dict:
            gt_nodes_dict[edge[1]] = zss.Node(edge[1])
        gt_nodes_dict[edge[0]].addkid(gt_nodes_dict[edge[1]])

    # Generated Tree construction:
    gen_graph = nx.read_weighted_edgelist(mst_file)
    gen_graph.add_edges_from(artificial_edge_list)

    if save_plots:
        # Save the Plot:
        plt.figure(figsize=(300, 300))
        nx.draw(gen_graph, with_labels=True)
        gen_file_name = mst_file.split("/")[-1].split(".")[0]
        plt.savefig(plot_dir + gen_file_name + ".png")

    gen_tree = nx.bfs_tree(gen_graph, source="root")

    gen_nodes_dict = {}
    for edge in gen_tree.edges():
        if edge[0] not in gen_nodes_dict:
            gen_nodes_dict[edge[0]] = zss.Node(edge[0])
        if edge[1] not in gen_nodes_dict:
            gen_nodes_dict[edge[1]] = zss.Node(edge[1])
        gen_nodes_dict[edge[0]].addkid(gen_nodes_dict[edge[1]])

    # Computing the Tree edit distance:
    tree_edit_distance = zss.distance(gt_nodes_dict['root'], gen_nodes_dict['root'], zss.Node.get_children, insert_cost=insertCost, remove_cost=removeCost, update_cost=updateCost)
    print("Tree Edit Distance = ", tree_edit_distance)


    # print(zss.simple_distance(gt_nodes_dict['root'], gen_nodes_dict['root']))
    cur_time = time.time()
    seconds_elapsed = cur_time-start_time
    print("Seconds Elapsed = ", seconds_elapsed)












