import os
import json
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from openai import AzureOpenAI
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline.graph import run_agent

def evaluate_draft(client, customer_text, predicted_intent, real_reply, draft_reply, model_name):
    prompt = f"""
You are an expert customer service evaluator.
Given a customer message, the real historical agent reply, and a drafted reply by an AI agent, rate the AI draft from 1-5 on three axes:
1. Tone (does it match AmazonHelp empathy/professionalism?)
2. Accuracy (does it address the customer's issue similarly to the real reply?)
3. Conciseness (is it direct and short?)

Customer: "{customer_text}"
Real Reply: "{real_reply}"
AI Draft: "{draft_reply}"

Output JSON only:
{{"tone": 4, "accuracy": 5, "conciseness": 3}}
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"tone": 0, "accuracy": 0, "conciseness": 0}

def main():
    df = pd.read_csv("data/golden/golden_set_test.csv")
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-mini")
    
    results = []
    print(f"Evaluating {len(df)} test cases...")
    
    for idx, row in df.iterrows():
        print(f"Processing {idx+1}/{len(df)}...")
        state = run_agent(row['customer_text'])
        
        real_reply = row['amazon_reply_text']
        scores = evaluate_draft(client, row['customer_text'], state.get('predicted_intent'), real_reply, state.get('draft_reply', ''), model_name)
        
        results.append({
            "tweet_id": row['tweet_id'],
            "customer_text": row['customer_text'],
            "human_intent": row['human_intent'],
            "predicted_intent": state.get('predicted_intent'),
            "human_escalate": row['human_escalate'],
            "predicted_escalate": state.get('escalation_decision'),
            "classifier_confidence": state.get('classifier_confidence'),
            "max_similarity": state.get('max_similarity'),
            "escalation_reasons": "; ".join(state.get('escalation_reasons', [])),
            "human_reply": real_reply,
            "draft_reply": state.get('draft_reply'),
            "eval_tone": scores.get("tone", 0),
            "eval_accuracy": scores.get("accuracy", 0),
            "eval_conciseness": scores.get("conciseness", 0)
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv("data/eval_results.csv", index=False)
    
    print("\n--- Intent Classification Report (Test Set) ---")
    print(classification_report(res_df['human_intent'], res_df['predicted_intent'], zero_division=0))
    
    print("\n--- Escalation Classification Report (Test Set) ---")
    print(classification_report(res_df['human_escalate'], res_df['predicted_escalate'], zero_division=0))
    
    print("\n--- LLM-as-Judge Draft Scores (Averages) ---")
    print(f"Tone: {res_df['eval_tone'].mean():.2f} / 5")
    print(f"Accuracy: {res_df['eval_accuracy'].mean():.2f} / 5")
    print(f"Conciseness: {res_df['eval_conciseness'].mean():.2f} / 5")
    
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
