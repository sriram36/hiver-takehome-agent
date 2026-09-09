import os
from openai import AzureOpenAI
from .state import AgentState

def load_brand_voice():
    try:
        with open("data/brand_voice.md", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "Be helpful and concise."

BRAND_VOICE = load_brand_voice()

def draft_reply(state: AgentState) -> AgentState:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    exemplars = state.get('retrieved_exemplars', [])
    exemplars_text = "\n".join([f"Customer: {ex['customer']}\nAmazonHelp: {ex['reply']}\n" for ex in exemplars])
    
    prompt = f"""
You are an AI customer support agent for AmazonHelp.

Brand Voice Guidelines:
{BRAND_VOICE}

Here are some historical examples of how similar issues were resolved:
{exemplars_text}

Draft a reply to the following customer message. Keep it under 280 characters.
Customer message: "{state['customer_text']}"
"""
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-mini")
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        state["draft_reply"] = response.choices[0].message.content
    except Exception as e:
        print(f"Drafting error: {e}")
        state["draft_reply"] = ""
        
    return state
