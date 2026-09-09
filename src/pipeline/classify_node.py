import os
import json
import pandas as pd
from openai import AzureOpenAI
from .state import AgentState

try:
    df = pd.read_csv("data/golden/golden_set_dev.csv")
    intents = df['human_intent'].unique()
    few_shot_examples = []
    for intent in intents:
        sample = df[df['human_intent'] == intent].iloc[0]
        few_shot_examples.append(f"Intent: {intent}\nCustomer: {sample['customer_text']}\n")
    FEW_SHOT_PROMPT = "Here are some examples of intents:\n\n" + "\n".join(few_shot_examples)
except Exception:
    FEW_SHOT_PROMPT = ""

def classify_intent(state: AgentState) -> AgentState:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    prompt = f"""
{FEW_SHOT_PROMPT}

Classify the following incoming customer tweet into exactly one of the 8 intents.
Also provide your confidence score between 0.0 and 1.0.

Customer Tweet:
"{state['customer_text']}"

Output strictly as JSON:
{{"intent": "intent_name", "confidence": 0.95}}
"""
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-mini")
    
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
        state["predicted_intent"] = result.get("intent", "other")
        state["classifier_confidence"] = float(result.get("confidence", 0.0))
    except Exception as e:
        print(f"Classification error: {e}")
        state["predicted_intent"] = "other"
        state["classifier_confidence"] = 0.0
        
    return state
