import pandas as pd
from dotenv import load_dotenv
import os
from openai import AzureOpenAI

def main():
    load_dotenv()
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    df = pd.read_csv("data/processed/amazonhelp_subsample.csv")
    brand_replies = df[df['author_id'] == 'AmazonHelp']['text'].dropna().sample(50, random_state=42).tolist()

    prompt = f"""
Analyze the following 50 real customer service replies from AmazonHelp on Twitter. 
Identify the core tone, formatting patterns, and constraints (e.g., greetings, apologies, sentence length, sign-offs like ^TR, empathy markers). 

Draft a concise 'Brand Voice Guide' (max 200 words) that can be inserted into a system prompt for an AI agent to emulate this exact style.

Replies:
{"\n".join(brand_replies)}
"""

    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-mini")
    print("Generating brand voice...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    with open("data/brand_voice.md", "w", encoding="utf-8") as f:
        f.write(response.choices[0].message.content)
    print("Brand voice guide generated and saved to data/brand_voice.md!")

if __name__ == "__main__":
    main()
