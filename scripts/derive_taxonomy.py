import os
import argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

def get_llm_client():
    load_dotenv()
    
    # Check for Azure OpenAI keys first
    if os.getenv("AZURE_OPENAI_API_KEY"):
        print("Using Azure OpenAI Client...")
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    elif os.getenv("OPENAI_API_KEY"):
        print("Using Standard OpenAI Client...")
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError("Neither AZURE_OPENAI_API_KEY nor OPENAI_API_KEY found in environment variables.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/amazonhelp_subsample.csv", help="Path to subsample dataset")
    parser.add_argument("--samples", type=int, default=300, help="Number of customer tweets to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/taxonomy.txt")
    args = parser.parse_args()

    # 1. Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # 2. Filter customer-side tweets
    # inbound=True means customer to brand
    customer_tweets = df[df['inbound'] == True]['text'].dropna().tolist()
    
    if len(customer_tweets) == 0:
        print("No customer tweets found in the dataset.")
        return

    # 3. Sample
    rng = np.random.default_rng(args.seed)
    sample_size = min(args.samples, len(customer_tweets))
    sampled_tweets = rng.choice(customer_tweets, size=sample_size, replace=False)
    
    print(f"Sampled {sample_size} customer tweets for taxonomy derivation.")
    
    # Create the prompt for the LLM
    tweets_text = "\n".join([f"- {text.replace('\n', ' ')}" for text in sampled_tweets])
    
    prompt = f"""
You are an expert customer support analyst. 
I have provided a random sample of {sample_size} customer support tweets sent to AmazonHelp.

Your task is to perform an open-coding thematic analysis to derive a clean, distinct set of 6 to 10 "Intents" (categories) that classify these customer requests. 

The intents should be mutually exclusive and cover the vast majority of the volume. Some typical e-commerce intents might be (but are not limited to):
- Order Status / Where is my stuff?
- Delivery Delay / Missed Delivery
- Refund / Return
- Wrong / Damaged Item
- Account Access / Payment Issue
- App / Website Bug

For each intent you discover, provide:
1. Intent Name
2. Description
3. 2-3 Example quotes from the sample data that fit this intent.

Data sample:
{tweets_text}

Output format: Please output the taxonomy in a clear markdown format.
"""

    client = get_llm_client()
    
    # Note: If using Azure OpenAI, model parameter usually needs to be the deployment name.
    # We default to gpt-4o-mini, but user might need to change it depending on their Azure setup.
    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini") if os.getenv("AZURE_OPENAI_API_KEY") else "gpt-4o-mini"
    
    print(f"Calling LLM ({model_name}) to derive taxonomy...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert data analyst building an intent taxonomy for customer support."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    taxonomy_output = response.choices[0].message.content
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(taxonomy_output)
        
    print(f"\nTaxonomy successfully written to {args.output}")
    print("\n--- Derived Taxonomy Preview ---\n")
    print(taxonomy_output[:500] + "\n...\n")

if __name__ == "__main__":
    main()
