from transformers import AutoTokenizer, AutoModel
import torch
import pandas
from annoy import AnnoyIndex

def embed_bert_cls(model_output):
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings

def get_question_embs(question):
    encoded_input = tokenizer(question, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        model_output = model_question(**encoded_input)
    question_embeddings = embed_bert_cls(model_output)
    return question_embeddings.tolist()[0]

def pretty_response(response):
    to_output = {}
    ids, distances = response
    for i in range(len(ids)):
        j = ids[i]
        score = distances[i]
        text = df['message'][j]
        to_output[j] = text + ' [ ' + str(score) + ' ]'
    return to_output
        
def get_response(question):
    vector = get_question_embs(question)
    return pretty_response(ANN.get_nns_by_vector(vector, 10, include_distances=True))

print("Started loading")
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model_question = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
model_answer = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

version = ''
model_question.load_state_dict(torch.load(f'models/model_anchor{version}.bin', map_location=torch.device('cpu')))
model_answer.load_state_dict(torch.load(f'models/model_pos_neg{version}.bin', map_location=torch.device('cpu')))
print("Model loaded")

f = 312
ANN = AnnoyIndex(f, 'angular')
ANN.load(f'{version[1:]}.ann')
df = pandas.read_csv('data/grammar_fix.csv')
print("ANN & dataframe loaded")