import faiss
import pickle
import pandas as pd
import numpy as np
import os
from .state import AgentState

class Retriever:
    def __init__(self):
        index_path = os.path.join("data", "faiss.index")
        vectorizer_path = os.path.join("data", "vectorizer.pkl")
        pairs_path = os.path.join("data", "retrieval_pairs.csv")
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            self.pairs = pd.read_csv(pairs_path)
        else:
            self.index = None
            
    def retrieve(self, state: AgentState) -> AgentState:
        if not self.index:
            state["retrieved_exemplars"] = []
            state["max_similarity"] = 0.0
            return state
            
        text = state["customer_text"]
        
        vec = self.vectorizer.transform([text]).toarray().astype('float32')
        faiss.normalize_L2(vec)
        
        D, I = self.index.search(vec, k=3)
        
        exemplars = []
        max_sim = float(D[0][0]) if len(D[0]) > 0 else 0.0
        
        for idx in I[0]:
            if idx != -1:
                exemplars.append({
                    "customer": self.pairs.iloc[idx]['customer_text'],
                    "reply": self.pairs.iloc[idx]['amazon_reply']
                })
                
        state["retrieved_exemplars"] = exemplars
        state["max_similarity"] = max_sim
        return state
