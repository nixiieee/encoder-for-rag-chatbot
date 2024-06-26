# Encoder for RAG Chatbot
## About the project

[Presentation for the project (in Russian)](https://drive.google.com/file/d/1wwlGdvPL2OwPJ7blrKwunxKaeXRsrx_6/view?usp=sharing)

This is a R&D project on developing a RAG-model chatbot for techsupport for telecommunications company. The project focues on developing encoder, performing various experiments on loss function and context understanding.

The main model which was used for the experiments is BERT (tuned on Russian datasets). Context understanding is implemented by concatenating last n messages of the user.

Losses (used in experiments):

- Triplet loss
- Softmax Margin loss. A classical softmax loss, but with a constant margin added to enforce a separation between classes 
- Softmax Margin loss + relevance score. Relevance score is retrieved from outer scorer model and added just like the margin constant. The idea behind this loss is to add ranking inside negative and positive classes.

More about loss formulas can be found in `losses.py` file.

The best performance was reached with Softmax Margin loss and the context window of last 6 messages.

## Dataset

The dataset original dataset - logs of techsupport chats with users - was modified into a "anchor-positive-negative" triplet format, where "anchor" is the user's message, "positive" is the correct answer (from the operator, used in a real dialog) and "negative" is an incorrect answer. A sample (with mock data) can be found in `data/triplet_data.csv`.

## Installing dependencies 

To install all of the required dependencies for the project run this command:

```
pip install -r requirements.txt
```

## Reproducing results

Unfortunately, the oridinal dataset used for the project is under NDA, so to reproduce the result you need to provide your own. Good quality on any language besides Russian is not gurateed (since the original dataset was in Russian). A small sample file with mock data can be found in folder `data`. If you want to reproduce the results, your dataset should resemble it in terms of format.

You can change the configuration for train (`config.json`) and test (`config_test.json`) according to your preferences. By default, Softmax margin loss is used.

To run train:

```
python3 train.py --config=config.json
```

To run test:

```
python3 test.py --config=config.json
```
