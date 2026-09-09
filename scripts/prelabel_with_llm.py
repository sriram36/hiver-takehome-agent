"""
prelabel_with_llm.py

Adds ai_suggested_intent / ai_suggested_escalate / ai_suggested_reason
columns to golden_set_candidates.csv using an LLM, so the human labelling
pass (you) is reading-and-correcting rather than starting from a blank
sheet. This does NOT replace human labelling -- see labeling_rubric.md.

Requires:
    pip install openai

Azure OpenAI, set these env vars (matches your usual setup):
    AZURE_OPENAI_ENDPOINT     e.g. https://<resource>.openai.azure.com/
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_DEPLOYMENT   the deployment/model name, e.g. gpt-4o-mini
    AZURE_OPENAI_API_VERSION  e.g. 2024-08-01-preview

Plain OpenAI instead, set:
    OPENAI_API_KEY
and pass --provider openai --model gpt-4o-mini

Usage:
    python prelabel_with_llm.py \
        --input golden_set_candidates.csv \
        --rubric ../eval/labeling_rubric.md \
        --output golden_set_candidates.csv \
        --provider azure
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd

INTENTS = [
    "delivery_delay", "item_issue", "refund_billing", "return_cancel",
    "account_access", "app_technical", "service_complaint", "other",
]

SYSTEM_PROMPT_TEMPLATE = """You are labelling customer support tweets sent to \
Amazon's support account for a research/eval dataset. You are given the \
same rubric a human labeller is using. Follow it exactly.

RUBRIC:
{rubric}

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"intent": "<one of: {intents}>", "escalate": "auto" or "escalate", "reason": "<one short phrase>"}}
"""

USER_TEMPLATE = """Prior customer turn (may be empty): {prior}

Customer message to classify: {text}
"""


def call_azure(client, deployment, system_prompt, user_prompt):
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=150,
    )
    return resp.choices[0].message.content


def parse_json_response(raw: str):
    # models sometimes wrap JSON in ```json fences despite instructions
    cleaned = re.sub(r"```json|```", "", raw).strip()
    return json.loads(cleaned)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=["azure", "openai"], default="azure")
    parser.add_argument("--model", default=None, help="overrides AZURE_OPENAI_DEPLOYMENT / model name for openai provider")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows, for a quick test")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds between calls, if you hit rate limits")
    args = parser.parse_args()

    with open(args.rubric, "r", encoding="utf-8") as f:
        rubric_text = f.read()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(rubric=rubric_text, intents=", ".join(INTENTS))

    df = pd.read_csv(args.input)
    for col in ["ai_suggested_intent", "ai_suggested_escalate", "ai_suggested_reason"]:
        if col not in df.columns:
            df[col] = ""

    if args.provider == "azure":
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )
        model = args.model or os.environ["AZURE_OPENAI_DEPLOYMENT"]
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = args.model or "gpt-4o-mini"

    n = len(df) if args.limit is None else min(args.limit, len(df))
    for i in range(n):
        row = df.iloc[i]
        if str(row["ai_suggested_intent"]).strip():
            continue  # already labelled, e.g. resuming after a crash
        user_prompt = USER_TEMPLATE.format(
            prior=row.get("prior_customer_turn", "") or "",
            text=row["customer_text"],
        )
        try:
            raw = call_azure(client, model, system_prompt, user_prompt)
            parsed = parse_json_response(raw)
            df.at[i, "ai_suggested_intent"] = parsed.get("intent", "")
            df.at[i, "ai_suggested_escalate"] = parsed.get("escalate", "")
            df.at[i, "ai_suggested_reason"] = parsed.get("reason", "")
        except Exception as e:
            print(f"[row {i}] failed: {e}", file=sys.stderr)
            df.at[i, "ai_suggested_intent"] = "ERROR"
        if (i + 1) % 20 == 0 or i == n - 1:
            df.to_csv(args.output, index=False)  # checkpoint periodically
            print(f"labelled {i + 1}/{n}")
        if args.sleep:
            time.sleep(args.sleep)

    df.to_csv(args.output, index=False)
    print(f"Done. Wrote {args.output}")


if __name__ == "__main__":
    main()
