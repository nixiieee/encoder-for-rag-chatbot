from torch.utils.data import Dataset

class TripletDataset(Dataset):

    def __init__(self, anchor, positive, negative, score, tokenizer, max_len):
        self.anchor = anchor
        self.positive = positive
        self.negative = negative
        self.score = score
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, item):

        anchor = str(self.anchor[item])
        positive = str(self.positive[item])
        negative = str(self.negative[item])
        score = self.score[item]

        anchor_encoding = self.tokenizer(
            anchor,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        positive_encoding = self.tokenizer(
            positive,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        negative_encoding = self.tokenizer(
            negative,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'anchor_ids': anchor_encoding['input_ids'].flatten(),
            'anchor_attention_mask': anchor_encoding['attention_mask'].flatten(),
            'positive_ids': positive_encoding['input_ids'].flatten(),
            'positive_attention_mask': positive_encoding['attention_mask'].flatten(),
            'negative_ids': negative_encoding['input_ids'].flatten(),
            'negative_attention_mask': negative_encoding['attention_mask'].flatten(),
            'score': score,
        }