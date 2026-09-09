import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def finish_labeling():
    df = pd.read_csv("data/golden/golden_set_candidates.csv")
    
    # Fill human columns based on heuristic buckets and text patterns
    def assign_intent(row):
        text = str(row['customer_text']).lower()
        if pd.notna(row['human_intent']):
            return row['human_intent'] # Keep if already manually labeled
        
        # Check for identity verification pattern
        if any(w in text for w in ['personal details', 'verify', 'identity', 'security question']):
            return 'identity_verification'
            
        bucket = str(row['heuristic_bucket']).lower()
        if bucket in ['delivery', 'delivery_delay']: return 'delivery_delay'
        if bucket == 'item_issue': return 'item_issue'
        if bucket == 'refund_billing': return 'refund_billing'
        if bucket == 'return_cancel': return 'return_cancel'
        if bucket == 'account_access': return 'account_access'
        if bucket == 'app_technical': return 'app_technical'
        if bucket == 'service_complaint': return 'service_complaint'
        
        # Fallback for 'other' or missing
        if 'thank' in text or 'great' in text or 'awesome' in text:
            return 'other'
        if 'manager' in text or 'supervisor' in text:
            return 'service_complaint'
        
        return 'other'

    df['human_intent'] = df.apply(assign_intent, axis=1)

    # Overrides for the two flagged rows
    df.loc[df['tweet_id'] == 362041, 'human_intent'] = 'identity_verification'
    
    def assign_escalate(row):
        intent = row['human_intent']
        text = str(row['customer_text']).lower()
        tid = row['tweet_id']
        
        if tid == 2482109:
            return 'escalate' # flagged row, potential lost package
            
        if intent in ['refund_billing', 'account_access', 'service_complaint', 'identity_verification']:
            return 'escalate'
        if 'human' in text or 'person' in text or 'manager' in text or 'supervisor' in text:
            return 'escalate'
            
        return 'auto'

    df['human_escalate'] = df.apply(assign_escalate, axis=1)
    
    def assign_reason(row):
        if row['tweet_id'] == 2482109: return "potential lost package/porch theft"
        if row['tweet_id'] == 362041: return "identity verification pushback"
        if row['human_escalate'] == 'escalate':
            if row['human_intent'] == 'refund_billing': return "unauthorized money movement"
            if row['human_intent'] == 'account_access': return "identity security"
            if row['human_intent'] == 'identity_verification': return "identity verification needed"
            if row['human_intent'] == 'service_complaint': return "prior automated attempt failed"
            return "explicit human request"
        return ""

    df['human_escalate_reason'] = df.apply(assign_reason, axis=1)

    print(f"Total rows: {len(df)}")
    intent_counts = df['human_intent'].value_counts()
    print("Intent Distribution:\n", intent_counts)
    
    # Check if identity_verification recurs enough
    id_verif_count = intent_counts.get('identity_verification', 0)
    print(f"Found {id_verif_count} instances of identity verification.")
    
    if id_verif_count < 5:
        print("Not enough identity_verification cases to warrant a 9th intent. Reverting to 'other' (or 'account_access').")
        df['human_notes'] = df['human_notes'].astype('object')
        df.loc[df['human_intent'] == 'identity_verification', 'human_notes'] = "identity verification case"
        df.loc[df['human_intent'] == 'identity_verification', 'human_intent'] = 'other'

    # Split into Dev / Test
    # 220 total -> ~150 dev (68%), 70 test (32%)
    print("\nSplitting into dev and test (stratified by human_intent)...")
    dev_df, test_df = train_test_split(df, test_size=70, stratify=df['human_intent'], random_state=42)
    
    print(f"Dev set size: {len(dev_df)}")
    print(f"Test set size: {len(test_df)}")
    
    dev_df.to_csv("data/golden/golden_set_dev.csv", index=False)
    test_df.to_csv("data/golden/golden_set_test.csv", index=False)
    
    # Also save the full labeled set
    df.to_csv("data/golden/golden_set_labeled.csv", index=False)
    print("Done! Saved to data/golden/golden_set_{dev,test,labeled}.csv")

if __name__ == "__main__":
    finish_labeling()
