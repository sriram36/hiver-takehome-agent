import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, f1_score

def main():
    df = pd.read_csv("data/golden/golden_set_dev.csv")
    
    texts = df['customer_text'].fillna('')
    y_intent = df['human_intent']
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    clf = LogisticRegression(class_weight='balanced', random_state=42)
    
    # Predict using 5-fold cross validation on dev set to avoid overfitting
    print("Running 5-fold CV on dev set for simple TF-IDF intent prediction...")
    y_pred = cross_val_predict(clf, X, y_intent, cv=5)
    
    print("\n--- Intent Classification Report (Simple TF-IDF) ---")
    print(classification_report(y_intent, y_pred, zero_division=0))
    
    f1_macro = f1_score(y_intent, y_pred, average='macro', zero_division=0)
    print(f"Macro F1: {f1_macro:.3f}")

if __name__ == "__main__":
    main()
