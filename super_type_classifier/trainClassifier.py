import argparse
import numpy as np
import pickle as pkl
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

sys.path.append("..")
from modelUtils.LanguageModels import WordEmbeddingsModel

def prepareTrainingData(data, embeddings_model, vector_feat):
    X_data = []
    y_data = []

    for i in range(len(data)):
        node_1 = data[i][0]
        node_2 = data[i][1]
        node_1_embed = embeddings_model.getEmbeddingsForWord(node_1)
        node_2_embed = embeddings_model.getEmbeddingsForWord(node_2)
        if node_1_embed is not None and node_2_embed is not None:
            if vector_feat == "vector_concat":
                cur_x = np.concatenate((node_1_embed, node_2_embed))
            elif vector_feat == "vector_diff":
                cur_x = (node_1_embed-node_2_embed)
            else:
                cur_x = None
            X_data.append(cur_x)
            y_data.append(data[i][2])

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--input_data", type=str, default=None,
                        help="Input Training data")
    parser.add_argument("-mt", "--model_type", type=str, default="RandomForest",
                        help="Classification model type for training. Options: SVM, Logistic, LogisticCV, RandomForest")
    parser.add_argument("-ep", "--embeddings_path", type=str, default=None,
                        help="Embeddings path")
    parser.add_argument("-ed", "--embeddings_dims", type=int, default=300,
                        help="Embeddings dimensions")
    parser.add_argument("-vf", "--vector_features", type=str, default="vector_diff",
                        help="Type of features. Options: vector_concat, vector_diff")
    parser.add_argument("-ocm", "--out_classifier_model", type=str, default=None,
                        help="Output Parent Classifier model")
    args = parser.parse_args()

    input_data_file = args.input_data
    model_type = args.model_type
    embeddings_path = args.embeddings_path
    vector_feat = args.vector_features
    embeddings_dims = args.embeddings_dims
    out_model_path = args.out_classifier_model

    embeddings_model = WordEmbeddingsModel(embeddings_path, embeddings_dims)

    with open(input_data_file, "rb") as f:
        data = pkl.load(f)

    X_data, y_data = prepareTrainingData(data, embeddings_model, vector_feat)

    if model_type == "RandomForest":
        # Random forest classifier model:
        model = RandomForestClassifier(verbose=1, random_state=0, n_estimators=200)
    elif model_type == "SVM":
        # SVM:
        model = SVC(verbose=1, kernel="rbf", decision_function_shape="ovr", random_state=0, gamma="auto")
    elif model_type == "LogisticCV":
        # Logistic Regression model with 5-fold cross validation:
        model = LogisticRegressionCV(verbose=1, cv=5, random_state=0, max_iter=1000)
    elif model_type == "Logistic":
        # Logistic Regression model
        model = LogisticRegression(verbose=1, random_state=0, max_iter=1000)
    else:
        model = None

    # Train the model
    model.fit(X_data, y_data)

    # Store the model
    with open(out_model_path, "wb") as f:
        pkl.dump(model, f)



