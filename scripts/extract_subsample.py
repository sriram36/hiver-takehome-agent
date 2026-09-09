"""
extract_subsample.py

Filters the Kaggle "Customer Support on Twitter" dataset (twcs.csv) down to
AmazonHelp conversation threads and writes a manageable, upload-sized
subsample for building/testing the support agent pipeline.

A "thread" here is: a customer's root inbound tweet -> every tweet reachable
by following in_response_to_tweet_id links forward (brand replies, customer
follow-ups, further brand replies, ...), capped at depth 6.

Usage:
    python extract_subsample.py \
        --input twcs.csv \
        --brand AmazonHelp \
        --threads 2000 \
        --output amazonhelp_subsample.csv
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to the raw twcs.csv")
    parser.add_argument("--brand", default="AmazonHelp", help="author_id of the brand handle")
    parser.add_argument("--threads", type=int, default=2000, help="number of root threads to sample")
    parser.add_argument("--max-depth", type=int, default=6, help="max hops to follow forward from the root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="subsample.csv")
    args = parser.parse_args()

    dtype = {
        "tweet_id": str,
        "author_id": str,
        "in_response_to_tweet_id": str,
        "response_tweet_id": str,
    }
    df = pd.read_csv(args.input, dtype=dtype)

    # inbound comes in as "True"/"False" strings when forced through generic read; normalize
    if df["inbound"].dtype != bool:
        df["inbound"] = df["inbound"].astype(str).str.strip().str.lower() == "true"

    # child map: tweet_id -> list of tweet_ids that replied to it (built once, O(n))
    children = defaultdict(list)
    for tid, parent in zip(df["tweet_id"], df["in_response_to_tweet_id"]):
        if isinstance(parent, str) and parent.lower() != "nan":
            children[parent].append(tid)

    id_to_inbound = dict(zip(df["tweet_id"], df["inbound"]))

    brand_ids = set(df.loc[df["author_id"] == args.brand, "tweet_id"])
    if not brand_ids:
        raise SystemExit(f"No tweets found for author_id == {args.brand!r}. Check the handle spelling.")

    # root customer tweets: inbound tweets that some AmazonHelp tweet directly replied to
    parents_of_brand_replies = set(
        df.loc[df["tweet_id"].isin(brand_ids), "in_response_to_tweet_id"].dropna()
    )
    root_ids = [tid for tid in parents_of_brand_replies if id_to_inbound.get(tid) is True]

    if not root_ids:
        raise SystemExit("Found brand tweets but no inbound root tweets — check column names/values.")

    rng = np.random.default_rng(args.seed)
    sample_size = min(args.threads, len(root_ids))
    sampled_roots = rng.choice(root_ids, size=sample_size, replace=False)

    # BFS forward from each sampled root using the prebuilt children map (fast)
    keep_ids = set()
    for root in sampled_roots:
        if root in keep_ids:
            continue
        keep_ids.add(root)
        frontier = [root]
        for _ in range(args.max_depth):
            next_frontier = []
            for node in frontier:
                for child in children.get(node, []):
                    if child not in keep_ids:
                        keep_ids.add(child)
                        next_frontier.append(child)
            frontier = next_frontier
            if not frontier:
                break

    out = df[df["tweet_id"].isin(keep_ids)].sort_values(["tweet_id"]).reset_index(drop=True)
    out.to_csv(args.output, index=False)

    n_brand_rows = (out["author_id"] == args.brand).sum()
    print(f"Sampled {sample_size} root threads")
    print(f"Total rows: {len(out)} ({n_brand_rows} from {args.brand}, {len(out) - n_brand_rows} customer-side)")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
