import argparse
import json

def computeStats(input_taxonomy):
    with open(input_taxonomy, "r") as f:
        num_nodes = 0
        num_childs_total = 0
        num_nodes_with_children = 0
        for cur_line in f:
            cur_dict = json.loads(cur_line)
            children = list(cur_dict.values())[0]["children"]
            num_nodes += 1
            num_childs_total += len(children)
            if len(children) != 0:
                num_nodes_with_children += 1

        print("Num. Concepts = ", num_nodes)
        print("Num. Edges = ", num_childs_total)
        print("Avg. Num. Children/Node = ", (num_childs_total/num_nodes_with_children))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-it", "--input_taxonomy", type=str, default=None,
                        help="Input Taxonomy for which the stats will be computed")
    args = parser.parse_args()

    input_taxonomy = args.input_taxonomy
    computeStats(input_taxonomy)













