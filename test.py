import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from annoy import AnnoyIndex
import torch.nn as nn
import torch
import argparse

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.pretrained_model = config["pretrained_model"]
        self.model_path_question = config["model_path_question"]
        self.model_path_answer = config["model_path_answer"]
        self.max_len = config["max_len"]
        self.data_file_operator_messages = config["data_file_operator_messages"]
        self.data_file_triplet_test = config["data_file_triplet_test"]
        self.annoy_file_path = config["annoy_file_path"]
        self.k_values = config["k_values"]
        self.device = config["device"]

class BertEmbedder:
    def __init__(self, pretrained_model, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModel.from_pretrained(pretrained_model)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def embed_bert_cls(self, model_output):
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings

    def get_embeddings(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.embed_bert_cls(model_output)
        return embeddings

def create_annoy_index(embedder_answer, df_operator_messages, annoy_file_path):
    f = embedder_answer.get_embeddings(df_operator_messages["message"][0]).shape[1]  # Dimensionality of embeddings
    ANN = AnnoyIndex(f, 'angular')

    for i in range(len(df_operator_messages)):
        message = df_operator_messages["message"][i]
        vector = embedder_answer.get_embeddings(message)[0]
        ANN.add_item(i, vector)

    ANN.build(100)
    ANN.save(f'{annoy_file_path}.ann')

    return ANN

def cosine_similarity_transform(angular_distance):
    return (2 - (angular_distance ** 2)) / 2

def calculate_total_cosine_similarity(df_test, embedder_question, ANN):
    total_cosine = 0
    for i in df_test.index:
        message = df_test["anchor_one_str"][i]
        vector = embedder_question.get_embeddings(message)[0]
        _, distances = ANN.get_nns_by_vector(vector, 1, include_distances=True)
        total_cosine += cosine_similarity_transform(distances[0])
    return total_cosine / df_test.shape[0]

def calculate_total_cosine_similarity_answer(df_test, embedder_question, embedder_answer):
    total_cosine_ans = 0
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for i in df_test.index:
        vector_q = embedder_question.get_embeddings(df_test["anchor_one_str"][i])
        vector_a = embedder_answer.get_embeddings(df_test["positive"][i])
        total_cosine_ans += cos(vector_q, vector_a)
    return float((total_cosine_ans / df_test.shape[0])[0])

def find_among_first_k(df_test, df_operator_messages, embedder_question, ANN, k):
    total_recall = 0
    for i in df_test.index:
        message = df_test["anchor_one_str"][i]
        vector = embedder_question.get_embeddings(message)
        response = ANN.get_nns_by_vector(vector[0], k, include_distances=True)
        total_recall += find_among_first_k_helper(df_test['positive'][i], response, k, df_operator_messages)
    return total_recall / df_test.shape[0]

def find_among_first_k_helper(to_find, response, k, df_operator_messages):
    ids, _ = response
    for i in range(len(ids)):
        id_cur = ids[i]
        text = df_operator_messages['message'][id_cur]
        if text == to_find:
            return 1
    return 0

def main():
    embedder_question = BertEmbedder(config.pretrained_model, config.model_path_question)
    embedder_answer = BertEmbedder(config.pretrained_model, config.model_path_answer)

    df_operator_messages = pd.read_csv(config.data_file_operator_messages)
    df_operator_messages = df_operator_messages.drop_duplicates(subset=['message'])

    df_test = pd.read_csv(config.data_file_triplet_test)

    # Create Annoy index
    ANN = create_annoy_index(embedder_answer, df_operator_messages, config.annoy_file_path)

    # Calculate and print total cosine similarity
    total_cosine = calculate_total_cosine_similarity(df_test, embedder_question, ANN)
    print(f"Total Cosine Similarity: {total_cosine}")

    # Calculate and print total cosine similarity (answer)
    total_cosine_ans = calculate_total_cosine_similarity_answer(df_test, embedder_question, embedder_answer)
    print(f"Total Cosine Similarity (Answer): {total_cosine_ans}")

    # Calculate and print recall@k for each k value
    for k in config.k_values:
        recall_at_k = find_among_first_k(df_test, df_operator_messages, embedder_question, ANN, k)
        print(f"Recall@{k}: {recall_at_k}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triplet Training Script")
    parser.add_argument("--config", default="config.json", help="")
    args = parser.parse_args()

    config = Config(args.config)
    main(config)
