from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import random

from newsapi import const
import json


def embed(word):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)
    with torch.no_grad():
        input_id = tokenizer.encode(word, return_tensors="pt")
        outputs = model(input_id.to(device))
        hidden_states = outputs.last_hidden_state
        return hidden_states[0]


def check_input(preferences):
    category_embeddings_serializable = {}
    genres = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("category_embeddings.json", "r") as f:
        category_embeddings_serializable = json.load(f)
    category_embeddings = {
        category: torch.tensor(embedding).to(device)
        for category, embedding in category_embeddings_serializable.items()
    }
    for pr in preferences:
        emb = embed(pr)
        max = 0
        genre = None
        for cat, embedding in category_embeddings.items():
            print(
                torch.nn.functional.cosine_similarity(emb, embedding).mean().item(), cat
            )
            if (
                torch.nn.functional.cosine_similarity(emb, embedding).mean().item()
                >= 0.84
            ):
                if (
                    torch.nn.functional.cosine_similarity(emb, embedding).mean().item()
                    > max
                ):
                    max = (
                        torch.nn.functional.cosine_similarity(emb, embedding)
                        .mean()
                        .item()
                    )
                    genre = cat
        genres.append(genre)
    return genres


def summarize(news: str):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    tokenizer.save_pretrained("summary_model")
    model_path = "summary_model/summary_model/checkpoint-1780"  # Assuming the checkpoint directory
    sum_model = T5ForConditionalGeneration.from_pretrained(
        model_path
    )  # Don't use from_pretrained
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sum_model.to(device)
    text = news
    inputs = tokenizer(
        text, truncation=True, padding="max_length", return_tensors="pt"
    )  # Convert to PyTorch tensors
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():  # Disable gradient calculation during inference
        outputs = sum_model.generate(**inputs)  # Pass tokenized input to the model
        summary_ids = outputs[0]  # Access the generated summary token ids
        summary_text = tokenizer.decode(
            summary_ids, skip_special_tokens=True
        )  # Decode token ids back to text

    return summary_text


def classify(text: str):
    classes = {3: "sport", 1: "entertainment", 0: "business", 2: "politics", 4: "tech"}
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    class_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=5
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_model.to(device)
    class_model.load_state_dict(torch.load("classify_model.pth"))
    class_model.eval()
    text_encoding = tokenizer.encode_plus(
        text, padding=True, truncation=True, return_tensors="pt"
    )
    input_ids, attention_mask = text_encoding["input_ids"].to(device), text_encoding[
        "attention_mask"
    ].to(device)

    with torch.no_grad():
        outputs = class_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # print(logits.shape)
        probabilities = torch.softmax(logits, dim=1)
        # print(probabilities.shape)
        max_prob, pred_label = torch.max(probabilities, dim=1)
        max_prob_overall = torch.max(max_prob)
        if max_prob_overall.item() > 0.6:
            return classes[pred_label[0].item()]
        else:
            return None
