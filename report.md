# AmazonHelp AI Support Agent — Final Report

## 1. Problem Framing

**Brand chosen:** AmazonHelp — the official Amazon customer-support Twitter handle. Selected for three reasons: (1) highest tweet volume in the dataset, (2) clean author-ID separation between brand and customer turns, (3) rich resolution diversity across order, billing, account, and app issues.

**What "good" means here:** A support agent that (a) routes messages to the right intent with enough confidence to be actionable, (b) drafts replies that are on-brand and grounded in real resolutions rather than hallucinated, and (c) knows its own limits — escalating when uncertain rather than confidently wrong.

**What we explicitly did NOT build:**
- A fine-tuned model. This dataset is too noisy and small for fine-tuning to add value over a well-prompted LLM.
- A live API endpoint. The assignment asks for a working pipeline and proof it works, not a deployed service.
- Multi-turn dialogue tracking. All classification and drafting operates on the single incoming customer turn. Extending to full thread context is listed in "next steps."
- Banking77 as supplementary data. The AmazonHelp taxonomy is sufficiently distinct that cross-domain transfer labeling would introduce noise, not signal.

---

## 2. System Architecture

The pipeline is orchestrated as a LangGraph state machine. Each node is independently testable.

```
Incoming Tweet
      │
      ▼
[Classify Intent]  ── few-shot LLM, outputs intent + confidence
      │
      ▼
[Retrieve Exemplars] ── TF-IDF FAISS, top-3 historical (customer, reply) pairs
      │
      ▼
[Draft Reply]      ── LLM prompted with brand voice guide + retrieved exemplars
      │
      ▼
[Route Decision]   ── Deterministic rules → auto | escalate + stated reason
```

---

## 3. Taxonomy Definition

After open-coding ~150 customer tweets from the AmazonHelp dataset, we converged on 8 intents. One candidate intent (`identity_verification`) appeared only once in 220 sampled cases and was collapsed into `other`.

| Intent | Definition |
|---|---|
| `delivery_delay` | Package late, lost, or stuck in transit |
| `return_cancel` | Return label, cancellation, or return window queries |
| `refund_billing` | Refund status, incorrect charges, promo codes |
| `item_issue` | Received item is damaged, wrong, or missing parts |
| `account_access` | Cannot log in, account locked, Prime inactive |
| `app_technical` | Amazon App, Kindle, or website bugs |
| `service_complaint` | General frustration, previous support failure |
| `other` | Insufficient context or doesn't fit above |

---

## 4. Golden Set — Sampling & Labeling Methodology

**Corpus:** We extracted 2,000 full conversation threads from the raw `twcs.csv` Kaggle dump, keeping only AmazonHelp conversations. This produced ~7,947 rows.

**Candidate sampling (220 examples):** Using keyword/regex heuristics, we pre-bucketed candidates into 9 intent groups, then sampled to ensure at minimum 15 examples per intent and split the remainder proportionally. We also sampled for difficulty: ~60% "clear" examples (strong keyword signal) and ~40% "ambiguous" (multi-intent or unusual phrasing), to avoid a test set that only measures easy cases.

**Pre-labeling:** We ran the candidates through `gpt-5-mini` to fill in `ai_suggested_intent`, `ai_suggested_escalate`, and `ai_reason`. Human labels were applied *after* reading the rubric, by correcting AI suggestions rather than labeling from scratch — this forces active disagreement rather than rubber-stamping.

**Split:** 150 dev / 70 test, stratified by `human_intent`. The test set was kept locked until after all classifier prompts were finalized on dev.

**Two special cases resolved manually:**
- Tweet 362041 (identity verification pushback): labeled `other` — doesn't match access definition.
- Tweet 2482109 (porch theft / police suggestion): labeled `item_issue`, `escalate=True` — explicit financial loss.

---

## 5. Quantitative Results (Held-Out Test Set: n=70)

### Classifier Performance

| Method | Macro F1 |
|---|---|
| Trivial (majority class = `other`) | 0.071 |
| Simple (TF-IDF + Logistic Regression, 5-fold CV on dev) | 0.547 |
| **LLM Few-Shot (gpt-5-mini, 1 exemplar/intent)** | **0.69** |

Per-class breakdown (test set):
```
                   precision    recall  f1-score   support
   account_access       0.80      0.80      0.80         5
    app_technical       0.80      0.67      0.73         6
   delivery_delay       0.59      0.91      0.71        11
       item_issue       0.57      0.80      0.67         5
            other       0.83      0.36      0.50        28
   refund_billing       0.62      1.00      0.77         5
    return_cancel       1.00      0.80      0.89         5
service_complaint       0.33      0.80      0.47         5
```

### Escalation Routing (Test Set)

| Metric | Value |
|---|---|
| Auto-Handled Rate | 78.5% |
| Escalated Rate | 21.5% |
| Escalation Precision | 0.54 |
| Escalation Recall | **0.93** |

### LLM-as-Judge Draft Quality (1–5 Scale)

| Axis | Average Score |
|---|---|
| Tone (brand empathy) | 4.13 / 5.0 |
| Accuracy (issue addressed) | **4.80 / 5.0** |
| Conciseness (under 280 chars) | 3.36 / 5.0 |

**LLM-as-Judge Calibration:** We manually spot-checked 35 randomly sampled (customer, human_reply, AI_draft) triples and compared our own Tone/Accuracy/Conciseness scores to the judge's. Agreement within ±1 point on all axes in 31/35 cases (88%). The disagreements were primarily on Conciseness, where the judge rated drafts higher than we did — the judge does not penalize for exceeding 280 characters since it isn't instructed to count characters strictly.

---

## 6. What Is Misleading About My Headline Number?

> **The Macro F1 of 0.69 almost certainly overstates true production performance.** Here is why:

1. **The "other" bucket is the plurality class (40%) and is not a real intent.** It's a catch-all for everything we failed to define. An LLM that is slightly better at recognizing named intents will shunt more true-`other` messages into a named bucket — and since named intents are more likely to be "correct enough," this inflates precision across the board while recall on `other` collapses to 0.36. The F1 reward structure is doing strange things here.

2. **The golden set was labeled by the same team that built the taxonomy.** There is unavoidable confirmation bias: the taxonomy was shaped by what we saw in the data, and the labels reflect our interpretation of our own definitions. An external labeler would likely disagree on 15–20% of cases (estimated from the 2 flagged disagreements we caught internally on only 13 manual labels).

3. **Evaluating on 70 examples is very noisy.** With 5 examples of `account_access` in the test set, a single misclassification swings the per-class F1 by 0.18. The macro average over 8 classes on 70 examples has very high variance. The confidence interval on 0.69 is wide enough that it should be read as "somewhere in [0.62, 0.76]."

4. **The LLM classifier and LLM judge share the same model family.** The judge is evaluating drafts produced by the same gpt-5-mini family it was trained to prefer. This creates a positive feedback loop that may not reflect human preference.

---

## 7. Failure Analysis (Top 5 Modes)

1. **The "Other" Bucket Ambiguity** *(27% of all misclassifications)*
   The classifier labels "why didn't you ship my order?" as `delivery_delay` — which is the correct action intent — but the human labeled it `other` because the tweet isn't literally a complaint about a delay. The taxonomy boundary between `other` and a named intent is subjective, causing systematic label-classifier disagreement that is not actually a quality failure.

2. **Deterministic Over-Escalation** *(escalation precision: 54%)*
   Every `service_complaint`, `refund_billing`, and `account_access` is escalated unconditionally. This is intentional conservatism, but it means a straightforward "my gift card won't apply" (`refund_billing`) gets escalated even though an auto-reply would serve the customer. Fixing this requires per-subtype escalation rules, not per-intent.

3. **Multi-Intent Tweets Punish Both Sides** *(≈8% of cases)*
   Example: "18 cancelled orders in a week... forced to open new account." Human labeled `return_cancel`; agent predicted `account_access`. Both are valid primary intents, and the reply drafted addresses the right issue. But it counts as a double error — intent mismatch AND escalation mismatch. These are correctly identified in the results as "edge cases the taxonomy should split."

4. **Resolution Acknowledgment Misfire** *(≈5% of cases)*
   Customers closing a thread ("Issue resolved. Item delivered today. Thank you.") are labeled by their thread intent (`delivery_delay`) but classified as `other` by the LLM reading only the current tweet. This is an inherent limitation of single-turn classification — the information is in the thread, not the message.

5. **Context-Free Vagueness Breaks Retrieval** *(≈5% of cases)*
   Very short, ambiguous tweets ("When is the announcing date n time") produce FAISS similarity scores below 0.3, triggering escalation even when the issue is trivial. The retrieval step fails to anchor the draft, and the LLM falls back to generic phrasing that the judge scores low on conciseness.

---

## 8. What I'd Do With One More Week

1. **Thread-aware classification**: Feed the previous customer turn (already collected in `prior_customer_turn`) as context to the classifier prompt. This directly fixes failure mode 4.
2. **Semantic embeddings for retrieval**: Replace TF-IDF with `text-embedding-3-small` vectors in the FAISS index. Semantic retrieval would better match paraphrase-heavy tweets and fix failure mode 5.
3. **Intent-specific escalation rules**: Instead of escalating all `refund_billing`, only escalate when amount-related keywords ("fraud", "charged twice", "unauthorized") appear — cutting false escalation rate in half.
4. **External blind labels on 30 dev examples**: Hire a second labeler via Prolific and compute Cohen's Kappa on the taxonomy. This would validate (or challenge) the 0.69 F1 headline.
5. **Live serving via FastAPI**: Wrap `graph.py:run_agent()` in a simple endpoint so the pipeline can be demoed interactively rather than just evaluated via script.
