# AmazonHelp AI Support Agent

An end-to-end AI support pipeline that classifies incoming customer tweets, retrieves grounding exemplars from history, drafts contextually-appropriate replies, and decides whether to auto-respond or escalate to a human.

---

## Quick Start — Reproduce Results in < 15 Minutes

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/sriram36/hiver-takehome-agent.git
cd hiver-takehome-agent

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Edit `.env` with your Azure OpenAI credentials:
```env
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.services.ai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-mini
```

### 3. Run the Evaluation Harness
The golden set, pre-built FAISS index, and brand voice guide are already committed. Just run:
```bash
python scripts/eval.py
```

This will:
- Run the full LangGraph pipeline (classify → retrieve → draft → route) on 70 held-out test cases
- Print classification report (intent + escalation)
- Run the LLM-as-a-judge on each draft
- Save full results to `data/eval_results.csv`

**Expected output:**
```
--- Intent Classification Report (Test Set) ---
Macro F1: 0.69

--- LLM-as-Judge Draft Scores ---
Tone: 4.13 / 5  |  Accuracy: 4.80 / 5  |  Conciseness: 3.36 / 5
```

---

## Repository Structure

```
hiver-takehome-agent/
├── src/
│   ├── pipeline/
│   │   ├── state.py           # LangGraph AgentState TypedDict
│   │   ├── classify_node.py   # Few-shot LLM intent classification
│   │   ├── retrieve.py        # FAISS TF-IDF retrieval node
│   │   ├── draft.py           # LLM reply drafting node
│   │   ├── route.py           # Deterministic escalation routing
│   │   └── graph.py           # LangGraph state machine assembly
│   └── baselines/
│       ├── trivial.py         # Majority-class baseline
│       └── simple_tfidf.py    # TF-IDF + Logistic Regression baseline
│
├── scripts/
│   ├── extract_subsample.py   # Extract AmazonHelp threads from twcs.csv
│   ├── derive_taxonomy.py     # Open-coding of intents
│   ├── build_golden_candidates.py  # Stratified 220-candidate sampling
│   ├── prelabel_with_llm.py   # LLM pre-labeling for human correction
│   ├── finish_labeling.py     # Resolves flagged rows, stratified dev/test split
│   ├── generate_brand_voice.py # One-time brand voice guide generation
│   ├── build_index.py         # Build TF-IDF FAISS index
│   └── eval.py                # End-to-end evaluation harness
│
├── data/
│   ├── golden/
│   │   ├── golden_set_candidates.csv  # 220 labeled candidates
│   │   ├── golden_set_dev.csv         # 150 dev examples
│   │   ├── golden_set_test.csv        # 70 held-out test examples
│   │   └── labeling_rubric.md         # Intent definitions + escalation rules
│   ├── processed/
│   │   └── amazonhelp_subsample.csv   # 7,947 extracted rows
│   ├── faiss.index            # Pre-built retrieval index (3,483 pairs)
│   ├── vectorizer.pkl         # TF-IDF vectorizer for the FAISS index
│   ├── retrieval_pairs.csv    # Historical (customer, reply) pairs
│   ├── brand_voice.md         # Distilled AmazonHelp brand voice guide
│   └── eval_results.csv       # Last evaluation run outputs
│
├── report.md                  # Problem framing, results, failure analysis
├── decision_log.md            # 13 key technical decisions with reasoning
├── requirements.txt
├── .env.example
└── README.md
```

---

## Running Individual Steps (If Starting from Raw Data)

If you have the raw `twcs.csv` from Kaggle, you can re-run the full pipeline:

```bash
# 1. Extract subsample
python scripts/extract_subsample.py --input twcs.csv --brand AmazonHelp --threads 2000 --output data/processed/amazonhelp_subsample.csv

# 2. Build golden candidates (220)
python scripts/build_golden_candidates.py

# 3. [Manual] Label candidates using data/golden/labeling_rubric.md

# 4. Finish labeling + dev/test split
python scripts/finish_labeling.py

# 5. Generate brand voice guide
python scripts/generate_brand_voice.py

# 6. Build FAISS index
python scripts/build_index.py

# 7. Run baselines on dev set
python src/baselines/trivial.py
python src/baselines/simple_tfidf.py

# 8. Run LLM classifier on dev set
python src/pipeline/classify.py

# 9. Run full evaluation on test set
python scripts/eval.py
```

---

## Key Results Summary

| Metric | Value |
|---|---|
| Trivial Baseline Macro F1 | 0.071 |
| TF-IDF Baseline Macro F1 | 0.547 |
| **LLM Few-Shot Macro F1** | **0.69** |
| Escalation Recall | 0.93 |
| Draft Tone (LLM Judge) | 4.13 / 5 |
| Draft Accuracy (LLM Judge) | 4.80 / 5 |

See `report.md` for full per-class breakdown, failure analysis, and the mandatory "What is misleading about my headline number?" section.
