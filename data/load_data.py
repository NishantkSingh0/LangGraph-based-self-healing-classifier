import pandas as pd
from transformers import AutoTokenizer

def load_data(path: str, chunksize=1000):
    data=pd.read_csv(path, chunksize=chunksize)
    return next(data)

def tokenize_data(data, tokenizer_name="distilbert-base-uncased"):
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized=tokenizer(
        data["review"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=256
    )
    return tokenized, tokenizer
