# Zero-Shot Taxonomy Induction using Representation Learning
Code Repository for the Paper: Zero-Shot Taxonomy Induction using Representation Learning: An Empirical Study


## Modules

### Retrofitting the embeddings

Retrofitting is implemented based on this paper: [Retrofitting Word Vectors to Semantic Lexicons](https://arxiv.org/abs/1411.4166)


### Super-type classifier

To create the training data from the taxonomy, use the following command,
 
```
python3 createTrainingData.py --src_taxonomy=<input_taxonomy> --out_file=<training_data>.pkl
```

where, <br>

<input_taxonomy> = Input taxonomy file path. Format of taxonomy file should be similar to the Google Product Taxonomy as found here: [https://www.google.com/basepages/producttype/taxonomy.en-US.txt](https://www.google.com/basepages/producttype/taxonomy.en-US.txt) <br>
<training_data> = Generated training data which is used for training the super-type classifier model <br>


To train the super-type classifier model, use the following command,

```
python3 trainClassifier.py --input_data=<training_data>.pkl --model_type=<model_type> 
--embeddings_path=<embeddings_path> --embeddings_dims=<embeddings_dims> --vector_features=<feature_type> 
--out_classifier_model=<output_model_path>.pkl
```

where, <br>

<training_data> = Training data created from taxonomy <br>
<model_type> = Classifier model to use (Valid options are SVM, Logistic, LogisticCV, RandomForest). Random forest works best <br>
<embeddings_path>, <embeddings_dims> = Embeddings to use (Should be in Word2Vec format) and its dimensions <br>
<feature_type> = Feature type to use (Valid options are vector_concat, vector_diff). 
vector_concat concatenates the word embeddings of the node pairs. 
vector_diff computes the difference between the word embeddings of the node pairs. <br>
<output_model_path> = Output model path to store the trained model <br>


### Ranking the neighbors

To generate the ranks file for the methods (Pre-trained Embeddings (method 1), Domain-specific Corpus-trained Embeddings (method 2), Super-type Classification using Pre-trained Embeddings (method 4), Super-type Classification using Retrofitted Embeddings (method 5)), use the following command,

```
python3 generateRanks.py --lexicon_to_evaluate=<ground_truth_lexicon> --embeddings_path=<embeddings_path> 
--embeddings_dims=<embeddings_dims> --classifier_model_path=<super_type_classifier_model_path>.pkl 
--out_ranks_file=<output_ranks_file_path>.jl
```

To generate the ranks file for the method (Retrofitted Embedding Transfer with Super-Type Classification (method 5)), use the following command,

```
python3 generateRanksBestMethod.py --lexicon_to_evaluate=<ground_truth_lexicon> --embeddings_path=<embeddings_path> 
--embeddings_dims=<embeddings_dims> --classifier_model_path=<super_type_classifier_model_path>.pkl 
--out_ranks_file=<output_ranks_file_path>.jl
```

where, <br>

<ground_truth_lexicon> = Ground truth lexicon <br>
<embeddings_path>, <embeddings_dims> = Embeddings to use (Should be in Word2Vec format) and its dimensions <br>
<super_type_classifier_model_path> = Trained super type classifier model (optional argument needed only for methods 3, 4 and 5)
<output_ranks_file_path> = Output path to store the generated ranks file (needed for computing the MAP and nDCG metrics in the Rank metrics module)


### Computing Rank metrics (MAP and nDCG)

To compute the MAP and nDCG metric, use the following command,

```
python3 computeRankMetrics.py --ground_truth=<ground_truth_lexicon> --baseline_ranks_file=<baseline_ranks>.jl 
--comparison_ranks_file=<comparison_ranks>.jl
```

where, <br>

<ground_truth_lexicon> = Ground Truth lexicon <br>
<baseline_ranks> = Baseline Ranks file <br>
<comparison_ranks> = Comparison Ranks file <br>


### Generating MST Tree

To generate the MST Tree from the ranks file, use the following command,

```
python3 computeMST.py --ranks_file=<input_ranks_file> --max_num_edges=<max_num_edges> --weighted=<weight_condition> 
--out_mst_file=<output_mst_file>
```

where, <br>

<input_ranks_file> = Input ranks file (generated in the "Ranking the neighbors" module) <br>
<max_num_edges> = Maximum number of edges considered for creating the lexicon graph (default: 50) <br>
<weight_condition> = Whether to use weights in the graph when computing the Maximum Spanning Tree (MST) from the graph. <br>
<output_mst_file> = Generated MST file (stored as weighted edge list representation as in NetworkX package) <br>


### Computing Approximate Tree Similarity Metrics

To compute the approximate tree similarity metrics, use the following command,

```
python3 computeApproxTreeSimilarity.py --baseline_mst_file=<baseline_mst_tree> --comparison_mst_file=<comparison_mst_tree> 
--ground_truth=<ground_truth_lexicon> --gt_sample_nodes_set=<sample_nodes_set>
```

where, <br>

<baseline_mst_tree> = Baseline MST tree (Should be in weighted edge list representation as in NetworkX package) <br>
<comparison_mst_tree> = Comparision MST tree (Should be in weighted edge list representation as in NetworkX package) <br>
<ground_truth_lexicon> = Ground Truth lexicon <br>
<sample_nodes_set> = Sample nodes set from the Ground Truth tree which are used as source nodes in the computation of Jaccard Similarity metric on the shortest paths. <br>


### Computing Tree Edit Distance Metric

To compute the Tree Edit distance metric, use the following command,

```
python3 computeTED.py --save_plots=<save_condition> --plot_dir=<plot_directory> --mst_file=<mst_tree> 
--ground_truth=<ground_truth_lexicon>
```

where, <br>

<save_condition> = Whether to save the visualization of the Ground truth tree and Maximum Spanning Tree (MST) as image files. (True or False) <br>
<plot_directory> = Director to save the plots <br>
<mst_tree> = MST tree (Should be in weighted edge list representation as in NetworkX package) <br>
<ground_truth_lexicon> = Ground Truth lexicon <br>

Note that it is not feasible to compute the Tree Edit Distance metric for a large tree (with number of nodes >=1000) because of its time complexity.

### Additional Utilities

To compute the various statistics about the taxonomy, use the following command,

```
python3 computeTaxonomyStats.py --input_taxonomy=<input_taxonomy>
```


To generate the lexicon from the taxonomy tree, use the following command,

```
python3 generateGroundTruth.py --src_taxonomy=<input_taxonomy> --out_lexicon_file=<output_lexicon>
```

where, <br>

<input_taxonomy> = Input taxonomy file path. Format of taxonomy file should be similar to the Google Product Taxonomy as found here: [https://www.google.com/basepages/producttype/taxonomy.en-US.txt](https://www.google.com/basepages/producttype/taxonomy.en-US.txt) <br>
<output_lexicon> = Output lexicon generated based on the input taxonomy tree <br>

## License

[MIT License](LICENSE)