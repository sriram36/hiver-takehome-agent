"""
build_golden_candidates.py

Turns the raw thread subsample into a stratified candidate pool for the
golden evaluation set. Produces a CSV with everything a human labeller
needs, plus a *non-authoritative* heuristic bucket used only to make sure
the sample covers all intents and a mix of easy/ambiguous cases -- it is
NOT the ground truth label.

Pipeline:
  1. Load the thread subsample (output of extract_subsample.py).
  2. Find customer tweets that AmazonHelp directly replied to.
  3. Keep English-language ones (langdetect) -- multilingual is explicitly
     out of scope for this build (see report: "what I chose not to build").
  4. Attach the immediately preceding customer turn (if any) as context,
     and the real historical AmazonHelp reply (for reference only).
  5. Tag each with a cheap keyword-heuristic intent bucket + an
     ambiguity flag (matched 0 or 2+ buckets = ambiguous).
  6. Stratified-sample N candidates across (bucket x difficulty).

Usage:
    python build_golden_candidates.py \
        --input amazonhelp_subsample.csv \
        --n 220 \
        --output golden_set_candidates.csv
"""
import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

# Keyword buckets -- for STRATIFICATION ONLY, never used as a final label.
BUCKET_KEYWORDS = {
    "refund_billing": [
        r"\brefund", r"\bcharge[ds]?\b", r"\bbilled\b", r"double charg",
        r"unauthori[sz]ed", r"membership fee", r"overcharg",
    ],
    "item_issue": [
        r"\bdamaged?\b", r"\bbroken\b", r"wrong item", r"\bmissing\b",
        r"\bdefective\b", r"\bfake\b", r"counterfeit", r"wrong (product|order)",
    ],
    "account_access": [
        r"log ?in", r"\bpassword\b", r"locked out", r"account access",
        r"can'?t access", r"account.*hack",
    ],
    "return_cancel": [
        r"\breturn(ed|ing)?\b", r"\bcancel(led|ling)?\b",
    ],
    "app_technical": [
        r"\bapp\b", r"\bcrash", r"\bkindle\b", r"\balexa\b", r"\becho\b",
        r"wifi", r"won'?t (work|connect)", r"doesn'?t work", r"\bbug\b",
    ],
    "service_complaint": [
        r"\bstill waiting\b", r"\buseless\b", r"\brude\b", r"\bworst\b",
        r"\bterrible\b", r"\bawful\b", r"\bhorrible\b", r"waste of time",
        r"no update", r"\bcomplain",
    ],
    "delivery": [
        r"\bdeliver", r"\barriv", r"\btracking\b", r"\bshipped\b",
        r"\bdispatch", r"\bpackage\b", r"where is my order",
    ],
}


def bucket_text(text: str):
    hits = []
    low = text.lower()
    for bucket, patterns in BUCKET_KEYWORDS.items():
        if any(re.search(p, low) for p in patterns):
            hits.append(bucket)
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--brand", default="AmazonHelp")
    parser.add_argument("--n", type=int, default=220)
    parser.add_argument("--min-per-bucket", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="golden_set_candidates.csv")
    args = parser.parse_args()

    dtype = {"tweet_id": str, "in_response_to_tweet_id": str, "response_tweet_id": str}
    df = pd.read_csv(args.input, dtype=dtype)
    if df["inbound"].dtype != bool:
        df["inbound"] = df["inbound"].astype(str).str.strip().str.lower() == "true"

    id_to_row = df.set_index("tweet_id")

    brand_rows = df[df["author_id"] == args.brand]
    parent_ids = set(brand_rows["in_response_to_tweet_id"].dropna())
    candidates = df[df["tweet_id"].isin(parent_ids) & (df["inbound"] == True)].copy()

    # keep one AmazonHelp reply per candidate (first, if several replied to the same tweet)
    reply_lookup = {}
    for _, row in brand_rows.iterrows():
        parent = row["in_response_to_tweet_id"]
        if pd.notna(parent) and parent not in reply_lookup:
            reply_lookup[parent] = row["text"]
    candidates["amazon_reply_text"] = candidates["tweet_id"].map(reply_lookup)

    # prior customer turn, if this tweet itself was a reply to an earlier customer message
    def prior_turn(row):
        parent = row["in_response_to_tweet_id"]
        if pd.notna(parent) and parent in id_to_row.index:
            parent_row = id_to_row.loc[parent]
            if isinstance(parent_row, pd.DataFrame):
                parent_row = parent_row.iloc[0]
            if parent_row["inbound"]:
                return parent_row["text"]
        return ""

    candidates["prior_customer_turn"] = candidates.apply(prior_turn, axis=1)

    def is_english(t):
        try:
            return detect(t) == "en"
        except Exception:
            return False

    candidates["is_english"] = candidates["text"].apply(is_english)
    candidates = candidates[candidates["is_english"]].copy()

    candidates["bucket_hits"] = candidates["text"].apply(bucket_text)
    candidates["heuristic_bucket"] = candidates["bucket_hits"].apply(
        lambda hits: hits[0] if len(hits) == 1 else ("other" if len(hits) == 0 else "ambiguous")
    )
    candidates["difficulty"] = candidates["bucket_hits"].apply(
        lambda hits: "clear" if len(hits) == 1 else "ambiguous"
    )

    # stratified sample: guarantee min_per_bucket per heuristic_bucket, then fill
    # remaining slots proportionally to bucket size
    rng = np.random.default_rng(args.seed)
    picked_idx = []
    remaining_budget = args.n
    buckets = candidates["heuristic_bucket"].unique()

    bucket_frames = {b: candidates[candidates["heuristic_bucket"] == b] for b in buckets}
    guaranteed = {}
    for b, frame in bucket_frames.items():
        take = min(len(frame), args.min_per_bucket)
        guaranteed[b] = take
    total_guaranteed = sum(guaranteed.values())

    for b, frame in bucket_frames.items():
        take = guaranteed[b]
        if take > 0:
            idx = rng.choice(frame.index, size=take, replace=False)
            picked_idx.extend(idx)

    remaining_budget = max(0, args.n - total_guaranteed)
    remaining_pool = candidates.drop(index=picked_idx)
    if remaining_budget > 0 and len(remaining_pool) > 0:
        take = min(remaining_budget, len(remaining_pool))
        idx = rng.choice(remaining_pool.index, size=take, replace=False)
        picked_idx.extend(idx)

    sample = candidates.loc[picked_idx].sample(frac=1, random_state=args.seed)  # shuffle order

    out_cols = [
        "tweet_id", "author_id", "created_at", "prior_customer_turn", "text",
        "amazon_reply_text", "heuristic_bucket", "difficulty",
    ]
    out = sample[out_cols].rename(columns={"text": "customer_text", "author_id": "customer_id"})
    out["human_intent"] = ""
    out["human_escalate"] = ""
    out["human_escalate_reason"] = ""
    out["human_notes"] = ""

    out.to_csv(args.output, index=False)

    print("Bucket distribution in full English candidate pool:")
    print(candidates["heuristic_bucket"].value_counts())
    print()
    print("Bucket distribution in sampled golden-set candidates:")
    print(out["heuristic_bucket"].value_counts())
    print()
    print("Difficulty distribution in sample:")
    print(out["difficulty"].value_counts())
    print(f"\nWrote {len(out)} candidates to {args.output}")


if __name__ == "__main__":
    main()
