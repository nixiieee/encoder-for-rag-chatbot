import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AdamW
import time
import datetime
import random
import numpy as np
import argparse
from dataset import TripletDataset 
from losses import TripletLoss, SoftmaxLoss, SoftmaxLossRScore, TripletMarginLossRScore 

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.pretrained_model = config["pretrained_model"]
        self.data_file = config["data_file"]
        self.test_file = config["test_file"]
        self.model_save_path_anchor = config["model_save_path_anchor"]
        self.model_save_path_pos_neg = config["model_save_path_pos_neg"]
        self.max_len = config["max_len"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.adam_epsilon = config["adam_epsilon"]
        self.epochs = config["epochs"]
        self.seed = config["seed"]
        self.train_split = config["train_split"]
        self.val_split = config["val_split"]
        self.loss_function = config["loss_function"]  # Added loss_function parameter

class TripletTrainer:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
        self.model_anchor = AutoModel.from_pretrained(config.pretrained_model).to(self.device)
        self.model_pos_neg = AutoModel.from_pretrained(config.pretrained_model).to(self.device)
        self.optimizer_a = AdamW(self.model_anchor.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)
        self.optimizer_pn = AdamW(self.model_pos_neg.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)
        self._set_seed(config.seed)

        # Initialize losses based on config
        self.triplet_loss = TripletLoss()
        self.softmax_loss = SoftmaxLoss()
        self.softmax_loss_rscore = SoftmaxLossRScore()
        self.triplet_margin_loss_rscore = TripletMarginLossRScore()

        # Initialize the selected loss function based on the config
        if config.loss_function == "softmax":
            self.loss_fn = self.softmax_loss
        elif config.loss_function == "softmax_rscore":
            self.loss_fn = self.softmax_loss_rscore
        else:
            raise ValueError(f"Unsupported loss function: {config.loss_function}")

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def load_data(self):
        df = pd.read_csv(self.config.data_file)
        ds = TripletDataset(
            anchor=df.anchor_one_str.to_numpy(),
            positive=df.positive.to_numpy(),
            negative=df.negative.to_numpy(),
            score=df.score,
            tokenizer=self.tokenizer,
            max_len=self.config.max_len
        )

        train_size = int(self.config.train_split * len(ds))
        val_size = int(self.config.val_split * len(ds))
        test_size = len(ds) - train_size - val_size

        generator = torch.Generator()
        generator.manual_seed(self.config.seed)

        train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size], generator=generator)

        self.train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=self.config.batch_size)
        self.validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=self.config.batch_size)
        
        indices = list(test_dataset.indices)
        df_test = df.iloc[indices]
        df_test.to_csv(self.config.test_file)

    def train(self):
        training_stats = []
        total_t0 = time.time()

        for epoch_i in range(self.config.epochs):
            print(f"\n======== Epoch {epoch_i + 1} / {self.config.epochs} ========")
            print('Training...')

            t0 = time.time()
            total_train_loss = 0.0
            train_losses = []

            self.model_anchor.train()
            self.model_pos_neg.train()

            for step, batch in enumerate(self.train_dataloader):
                self.optimizer_a.zero_grad()
                self.optimizer_pn.zero_grad()

                anchor_ids = batch["anchor_ids"].to(self.device)
                anchor_attention_mask = batch["anchor_attention_mask"].to(self.device)
                anchor_outputs = self.model_anchor(input_ids=anchor_ids, attention_mask=anchor_attention_mask)
                anchor_embeddings = self.embed_bert_cls(anchor_outputs)

                positive_ids = batch["positive_ids"].to(self.device)
                positive_attention_mask = batch["positive_attention_mask"].to(self.device)
                positive_outputs = self.model_pos_neg(input_ids=positive_ids, attention_mask=positive_attention_mask)
                positive_embeddings = self.embed_bert_cls(positive_outputs)

                negative_ids = batch["negative_ids"].to(self.device)
                negative_attention_mask = batch["negative_attention_mask"].to(self.device)
                negative_outputs = self.model_pos_neg(input_ids=negative_ids, attention_mask=negative_attention_mask)
                negative_embeddings = self.embed_bert_cls(negative_outputs)

                # Calculate loss using the selected loss function
                if isinstance(self.loss_fn, SoftmaxLoss):
                    loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
                elif isinstance(self.loss_fn, SoftmaxLossRScore):
                    loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings, batch["score"].to(self.device))
                else:
                    raise ValueError("Unsupported loss function for training")

                total_train_loss += loss.detach()
                loss.backward()

                train_losses.append(float(loss))
                
                nn.utils.clip_grad_norm_(self.model_anchor.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.model_pos_neg.parameters(), 1.0)

                self.optimizer_a.step()
                self.optimizer_pn.step()

                if step % 40 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print(f'  Batch {step:>5,}  of  {len(self.train_dataloader):>5,}.    Elapsed: {elapsed}.')
                    print(f"  Loss at step {step} = {loss} \n")

            avg_train_loss = total_train_loss / len(self.train_dataloader)
            training_time = self.format_time(time.time() - t0)

            print(f"\n  Average training loss: {avg_train_loss:.2f}")
            print(f"  Training epoch took: {training_time}")

            self.evaluate(training_stats, epoch_i, avg_train_loss, train_losses, training_time)

        print("\nTraining complete!")
        print(f"Total training took {self.format_time(time.time() - total_t0)} (h:mm:ss)")

        torch.save(self.model_anchor.state_dict(), self.config.model_save_path_anchor)
        torch.save(self.model_pos_neg.state_dict(), self.config.model_save_path_pos_neg)

    def embed_bert_cls(self, model_output):
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings

    def format_time(self, elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def evaluate(self, training_stats, epoch_i, avg_train_loss, train_losses, training_time):
        print("\nRunning Validation...")

        t0 = time.time()
        self.model_anchor.eval()
        self.model_pos_neg.eval()

        total_val_loss = 0.0
        val_losses = []

        with torch.no_grad():
            for step, batch in enumerate(self.validation_dataloader):
                anchor_ids = batch["anchor_ids"].to(self.device)
                anchor_attention_mask = batch["anchor_attention_mask"].to(self.device)
                anchor_outputs = self.model_anchor(input_ids=anchor_ids, attention_mask=anchor_attention_mask)
                anchor_embeddings = self.embed_bert_cls(anchor_outputs)

                positive_ids = batch["positive_ids"].to(self.device)
                positive_attention_mask = batch["positive_attention_mask"].to(self.device)
                positive_outputs = self.model_pos_neg(input_ids=positive_ids, attention_mask=positive_attention_mask)
                positive_embeddings = self.embed_bert_cls(positive_outputs)

                negative_ids = batch["negative_ids"].to(self.device)
                negative_attention_mask = batch["negative_attention_mask"].to(self.device)
                negative_outputs = self.model_pos_neg(input_ids=negative_ids, attention_mask=negative_attention_mask)
                negative_embeddings = self.embed_bert_cls(negative_outputs)

                # Calculate loss using the selected loss function
                if isinstance(self.loss_fn, SoftmaxLoss):
                    loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
                elif isinstance(self.loss_fn, SoftmaxLossRScore):
                    loss = self.loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings, batch["score"].to(self.device))
                else:
                    raise ValueError("Unsupported loss function for evaluation")

                total_val_loss += loss.detach()
                val_losses.append(float(loss))

            avg_val_loss = total_val_loss / len(self.validation_dataloader)
            validation_time = self.format_time(time.time() - t0)

            print(f"Validation Loss: {avg_val_loss}")
            print(f"  Validation took: {validation_time}")

            training_stats.append({
                'epoch': epoch_i + 1,
                'Training Loss': float(avg_train_loss),
                'Valid. Loss': float(avg_val_loss),
                'List of train losses': train_losses,
                'List of val losses': val_losses,
                'Training Time': training_time,
                'Validation Time': validation_time
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triplet Training Script")
    parser.add_argument("--config", default="config.json", help="")
    args = parser.parse_args()

    config = Config(args.config)
    trainer = TripletTrainer(config)
    trainer.load_data()
    trainer.train()
