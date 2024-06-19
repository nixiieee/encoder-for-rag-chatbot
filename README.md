# Encoder for RAG Chatbot
## Project description

[Presentation for the project (in Russian)](https://drive.google.com/file/d/1wwlGdvPL2OwPJ7blrKwunxKaeXRsrx_6/view?usp=sharing)

This is a R&D project on developing a RAG-model chatbot for techsupport for telecommunications company. The project focues on developing encoder, performing various experiments on loss function and context understanding.

The main model which was used for the experiments is BERT (tuned on Russian datasets). Context understanding is implemented by concatenating last 6 messages of the user.

Losses (used in experiments):

- Triplet loss

- Softmax margin ($\alpha$, a hyperparameter) loss:

\begin{align*}
 \frac{1}{N} \sum_{i=1}^{N} \max \left( 0, \log \left( \sum_{j=1}^{k} \exp (\mathbf{a}_i \cdot \mathbf{n}_{ij}) \right) - (\mathbf{a}_i \cdot \mathbf{p}_i) + \alpha \right)
\end{align*}

- Softmax margin loss + relevance score ($r_i$, retrieved from outer scorer model):

  ![Softmax Margin Loss + Relevance Score](https://latex.codecogs.com/png.latex?\frac{1}{N}\sum_{i=1}^{N}\max\left(0,\log\left(\sum_{j=1}^{k}\exp(\mathbf{a}_i\cdot\mathbf{n}_{ij})\right)-(\mathbf{a}_i\cdot\mathbf{p}_i)+r_i+\alpha\right))




## Installing dependencies 

To install all of the required dependencies for the project run this command:

```
pip install -r requirements.txt
```

## Reproducing results

Unfortunately, the oridinal dataset used for the project is under NDA, so to reproduce the result you need to provide your own. A small sample file with mock data can be found in folder `data`. If you want to reproduce the results, your dataset should resemble it in terms of format.
