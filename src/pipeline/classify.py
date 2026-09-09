import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from sklearn.metrics import classification_report, f1_score
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_llm_client():
    load_dotenv()
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    elif os.getenv("OPENAI_API_KEY"):
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError("No LLM API key found in environment variables.")

def classify_text(client, model_name, text, few_shot_prompt):
    prompt = f"""
{few_shot_prompt}

Now classify the following incoming customer tweet into exactly one of the 8 intents.
Also provide your confidence score between 0.0 and 1.0.

Customer Tweet:
"{text}"

Output strictly as JSON:
{{"intent": "intent_name", "confidence": 0.95}}
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an AmazonHelp routing agent. Only reply with the requested JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        return result.get("intent", "other"), result.get("confidence", 0.0)
    except Exception as e:
        print(f"Error classifying: {e}")
        return "other", 0.0

def main():
    df = pd.read_csv("data/golden/golden_set_dev.csv")
    
    # 1. Select one few-shot example per intent
    intents = df['human_intent'].unique()
    few_shot_examples = []
    few_shot_indices = []
    
    for intent in intents:
        sample = df[df['human_intent'] == intent].iloc[0]
        few_shot_examples.append(f"Intent: {intent}\nCustomer: {sample['customer_text']}\n")
        few_shot_indices.append(sample.name)
        
    few_shot_prompt = "Here are some examples of intents:\n\n" + "\n".join(few_shot_examples)
    
    # 2. Setup LLM
    try:
        client = get_llm_client()
    except ValueError as e:
        print(e)
        print("Please configure your .env file with your Microsoft Foundry keys.")
        return
        
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-mini")
    
    # 3. Predict the rest of the dev set (excluding the few-shot examples so we don't evaluate on them)
    eval_df = df.drop(index=few_shot_indices).copy()
    
    print(f"Classifying {len(eval_df)} dev examples with {model_name}...")
    
    predictions = []
    confidences = []
    
    # Parallel processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_idx = {executor.submit(classify_text, client, model_name, row['customer_text'], few_shot_prompt): idx for idx, row in eval_df.iterrows()}
        
        results = {}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            intent, conf = future.result()
            results[idx] = (intent, conf)
            
    for idx in eval_df.index:
        intent, conf = results[idx]
        predictions.append(intent)
        confidences.append(conf)
        
    eval_df['predicted_intent'] = predictions
    eval_df['predicted_confidence'] = confidences
    
    print("\n--- Intent Classification Report (Few-Shot LLM) ---")
    print(classification_report(eval_df['human_intent'], eval_df['predicted_intent'], zero_division=0))
    
    f1_macro = f1_score(eval_df['human_intent'], eval_df['predicted_intent'], average='macro', zero_division=0)
    print(f"Macro F1: {f1_macro:.3f}")

if __name__ == "__main__":
    main()
