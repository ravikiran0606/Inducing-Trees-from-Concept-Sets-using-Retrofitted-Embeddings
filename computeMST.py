import argparse
import networkx as nx
import json

def str2bool(input_str):
    if input_str is None:
        return None
    elif input_str.lower() == "true":
        return True
    elif input_str.lower() == "false":
        return False

def createGraph(ranks_file, max_edges):
    node_list = []
    edge_list = []
    with open(ranks_file, "r") as f:
        for cur_line in f:
            cur_dict = json.loads(cur_line)
            key = list(cur_dict.keys())[0]
            node_list.append(key)

            neighbors = list(cur_dict.values())[0][0]
            for i in range(0, max_edges):
                cur_n = neighbors[i]
                edge_list.append((key, cur_n, 1))

    cur_graph = nx.Graph()
    cur_graph.add_nodes_from(node_list)
    cur_graph.add_weighted_edges_from(edge_list)
    return cur_graph

def createGraphWeighted(ranks_file, max_edges):
    node_list = []
    edge_list = []
    with open(ranks_file, "r") as f:
        for cur_line in f:
            cur_dict = json.loads(cur_line)
            key = list(cur_dict.keys())[0]
            node_list.append(key)

            neighbors = list(cur_dict.values())[0][0]
            scores = list(cur_dict.values())[0][1]
            for i in range(0, max_edges):
                cur_n = neighbors[i]
                edge_list.append((key, cur_n, scores[i]))

    cur_graph = nx.Graph()
    cur_graph.add_nodes_from(node_list)
    cur_graph.add_weighted_edges_from(edge_list)
    return cur_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rf", "--ranks_file", type=str, default=None,
                        help="Ranks file")
    parser.add_argument("-mne", "--max_num_edges", type=int, default=50,
                        help="Maximum number of edges to consider while creating the graph from rank lists")
    parser.add_argument("-w", "--weighted", type=str, default=False,
                        help="Whether to create MST from a weighted graph with rank lists scores as weights")
    parser.add_argument("-omf", "--out_mst_file", type=str, default=None,
                        help="Output Minimum Spanning Tree file")
    args = parser.parse_args()

    ranks_file = args.ranks_file
    max_num_edges = args.max_num_edges
    weighted = str2bool(args.weighted)
    out_mst_file = args.out_mst_file

    if weighted:
        graph = createGraphWeighted(ranks_file, max_num_edges)
    else:
        graph = createGraph(ranks_file, max_num_edges)

    mst = nx.maximum_spanning_tree(graph)
    nx.write_weighted_edgelist(mst, out_mst_file)

    # Testing:
    # for (u, v, wt) in graph.edges.data('weight'):
    #     print('(%s, %s, %.3f)' % (u, v, wt))



















