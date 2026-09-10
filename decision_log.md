# Decision Log

A record of the 13 non-obvious decisions made during this project, with reasoning.

---

**1. Chose AmazonHelp, not a smaller brand**
AmazonHelp had the highest tweet count (~120k+), the cleanest author-ID separation between brand and customer turns, and the richest resolution diversity. Smaller brands had too few brand-turn examples to build a meaningful retrieval corpus.

---

**2. Open-coded the taxonomy bottom-up, not top-down**
Rather than importing an existing taxonomy (e.g., Banking77), we read ~150 random AmazonHelp customer tweets and grouped them by what reply action they required. This produces a taxonomy grounded in how Amazon actually operates, not how a generic support system is categorized. The cost is that the taxonomy is non-transferable; the benefit is that every intent directly maps to a distinct reply strategy.

---

**3. Did not include a 9th intent for `identity_verification`**
One tweet in the 220-candidate golden set matched an `identity_verification` pattern (pushback on ID request during account verification). We checked the full candidate pool and found it appeared only once (~0.5%). Adding a class with one training example would make the classifier worse, not better. Documented in labeling notes.

---

**4. Pre-labeled with LLM before human labeling, not after**
We ran `gpt-5-mini` over all 220 candidates first, then humans corrected rather than created labels. This forces active engagement (you must override a suggestion, not fill in a blank) and speeds labeling significantly. The risk is anchoring bias; mitigated by reviewing every label against the rubric definition, not just the suggested label.

---

**5. Stratified sampling for difficulty, not just intent**
The candidate pool was split ~60% "clear" (strong keyword signal) and ~40% "ambiguous" (multi-intent, sarcastic, or non-English tweets). A purely keyword-sampled golden set would make classifier performance look better than it is in production, where ambiguous messages are common.

---

**6. Used TF-IDF, not dense embeddings, for FAISS retrieval**
The retrieval corpus (~3,400 pairs) is small enough that TF-IDF performs comparably to semantic embeddings on keyword-heavy customer-service text. Dense embedding calls would require an extra API round-trip per inference request, doubling latency and introducing a new external dependency. TF-IDF is also fully explainable — you can see exactly which terms drove the match.

---

**7. Drew few-shot exemplars from dev set, not retrieval corpus**
The 8 exemplars injected into the classifier prompt are the first example of each intent from `golden_set_dev.csv` — human-labeled, hand-selected, and curated to the rubric definition. Using random corpus examples would risk noisy or off-brand exemplars that teach the classifier the wrong boundary.

---

**8. Removed `temperature=0.0` from API calls**
The Azure-hosted `gpt-5-mini` model does not support setting temperature to 0. Rather than trying a different value, we removed the parameter entirely and rely on determinism through careful prompt structuring (exact output format, JSON schema). This was discovered at runtime and fixed immediately.

---

**9. Escalation is deterministic, not LLM-decided**
Asking the LLM "should this be escalated?" adds latency, non-determinism, and a new failure mode (LLM reasoning errors). Our rule-based `route.py` node is fully transparent, auditable, and produces consistent behavior. The explicit rules (high-risk intent, low confidence, poor retrieval grounding, human-request keywords) were derived from the labeling rubric escalation guidelines.

---

**10. Excluded test-set tweets from the retrieval index**
`build_index.py` loads `golden_set_test.csv` tweet IDs and explicitly excludes their parent-customer-tweets from the FAISS corpus. If we didn't do this, the retrieval node could return the exact ground-truth reply as an exemplar for a test-set tweet, inflating draft quality scores.

---

**11. Pre-computed the brand voice guide, not injected inline**
Providing 50 real AmazonHelp replies as inline context for every draft request would consume ~3,000 tokens per call. Instead, we ran a one-time summarization to compress the brand's style into a ~200-word prompt fragment. This reduces cost by ~10x per call with negligible quality loss since the guide captures patterns, not specific resolutions.

---

**12. Evaluated with macro F1, not accuracy**
The `other` class holds 40% of examples. Accuracy rewards a system that calls everything `other`. Macro F1 weights each class equally, penalizing the system for ignoring minority intents — which is exactly the production failure mode we care about (mis-routing a `refund_billing` as `other` has real consequences).

---

**13. AI assistance disclosure**
This project was scaffolded with the help of AI coding assistants (Claude / Antigravity). All architectural decisions, taxonomy definitions, labeling judgments, failure analyses, and final code were reviewed, understood, and can be explained by me. The assignment explicitly permits AI assistants; this note is here because the rules also say "cite anything you borrowed."

