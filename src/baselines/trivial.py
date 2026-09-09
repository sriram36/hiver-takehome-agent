import pandas as pd
from sklearn.metrics import classification_report, f1_score

def main():
    df = pd.read_csv("data/golden/golden_set_dev.csv")
    
    # Trivial baseline: predict majority intent
    majority_intent = df['human_intent'].mode()[0]
    print(f"Majority intent is: {majority_intent}")
    
    df['predicted_intent'] = majority_intent
    
    # Fixed escalation rule: predict majority escalation rule
    majority_escalate = df['human_escalate'].mode()[0]
    df['predicted_escalate'] = majority_escalate
    
    print("\n--- Intent Classification Report (Trivial) ---")
    print(classification_report(df['human_intent'], df['predicted_intent'], zero_division=0))
    
    f1_macro = f1_score(df['human_intent'], df['predicted_intent'], average='macro', zero_division=0)
    print(f"Macro F1: {f1_macro:.3f}")

if __name__ == "__main__":
    main()
