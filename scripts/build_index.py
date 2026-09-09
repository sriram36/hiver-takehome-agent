import pandas as pd
import numpy as np
import faiss
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def build_index():
    print("Loading data...")
    df = pd.read_csv("data/processed/amazonhelp_subsample.csv")
    
    try:
        test_df = pd.read_csv("data/golden/golden_set_test.csv")
        test_tweet_ids = set(test_df['tweet_id'].astype(str))
    except Exception as e:
        print("Could not load test set, assuming empty test set.")
        test_tweet_ids = set()
        
    id_to_text = dict(zip(df['tweet_id'].astype(str), df['text']))
    id_to_author = dict(zip(df['tweet_id'].astype(str), df['author_id']))
    
    pairs = []
    
    for _, row in df.iterrows():
        if row['author_id'] == 'AmazonHelp' and pd.notna(row['in_response_to_tweet_id']):
            parent_id = str(row['in_response_to_tweet_id']).replace('.0', '')
            if parent_id in id_to_text and id_to_author.get(parent_id) != 'AmazonHelp':
                if parent_id not in test_tweet_ids:
                    pairs.append({
                        "customer_text": id_to_text[parent_id],
                        "amazon_reply": row['text']
                    })
                    
    print(f"Found {len(pairs)} historical pairs (excluding test set).")
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv("data/retrieval_pairs.csv", index=False)
    
    print("Building TF-IDF FAISS index...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1500)
    X = vectorizer.fit_transform(pairs_df['customer_text']).toarray().astype('float32')
    
    faiss.normalize_L2(X)
    
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    
    faiss.write_index(index, "data/faiss.index")
    with open("data/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("Index built and saved to data/faiss.index")

if __name__ == "__main__":
    build_index()
