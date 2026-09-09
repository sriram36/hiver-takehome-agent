# Golden Set Labeling Rubric — AmazonHelp Support Agent

For every row in `golden_set_candidates.csv`, fill in:
- `human_intent` — one of the 8 labels below
- `human_escalate` — `auto` or `escalate`
- `human_escalate_reason` — one short phrase (this becomes the agent's stated reason)
- `human_notes` — anything odd: sarcasm, multiple issues in one tweet, needs
  more context than given, you disagreed with the AI draft suggestion, etc.

If a tweet plausibly fits two intents, pick the one the reply would need to
address *first*, and note the second in `human_notes`. If nothing fits, use
`other` rather than forcing it — and note why in `human_notes`; a pile-up of
genuine misfits is itself a finding for the report.

## Intents

**1. delivery_delay** — order hasn't arrived, tracking shows no movement,
wrong delivery location, missed promised window.
> "@AmazonHelp I paid for one day shipping and it's delayed... this doesn't
> even make sense" → `delivery_delay`, `auto` — status/apology template
> covers this; no money or safety risk.

**2. item_issue** — item received is damaged, wrong, incomplete, or
counterfeit/fake.
> "@AmazonHelp I received the hardcover book that I ordered and it is
> completely ripped and smashed" → `item_issue`, `auto` — standard
> replacement/refund offer, unless value is unusually high (see below).

**3. refund_billing** — wrong or duplicate charge, refund not received or
reversed, unauthorized charge, membership fee dispute.
> "@AmazonHelp I was recharged for an order that was refunded... I expect my
> refund immediately" → `refund_billing`, `escalate` — money is moving in a
> way the customer didn't authorize; needs a human to verify the account.

**4. return_cancel** — wants to cancel an order, wants to return an item,
or is asking about the status of an existing return/cancellation.
> "@AmazonHelp Can you help me check the status of my return?" →
> `return_cancel`, `auto` — policy/status lookup, no dispute yet.

**5. account_access** — can't log in, account locked, missing account data
(gift card balance, order history, content).
> "@AmazonHelp I'm unable to log into an account I probably haven't used in
> 3 years... Can you offer any support?" → `account_access`, `escalate` —
> can't verify identity safely over a public Twitter reply.

**6. app_technical** — app crashes, device (Kindle/Alexa/Echo/Fire TV)
won't connect or work, streaming/playback bugs, feature not functioning.
> "@AmazonHelp i tried uninstalling the app... nothing works" →
> `app_technical`, `auto` — troubleshooting steps first; escalate only if
> the customer says they've already tried the standard steps.

**7. service_complaint** — expresses frustration about a *prior* support
interaction (long hold, rude agent, "still waiting," no follow-up) with no
new substantive information. This is a continuation of an existing case,
not a fresh issue.
> "@AmazonHelp U have been working on this for last 10 days... Now I want my
> iphone or my money back" → `service_complaint`, `escalate` — automation
> already failed this customer once; don't let it fail twice.

**8. other** — general pre-purchase questions, praise/positive feedback,
off-topic, or anything genuinely not covered above.
> "Issue has kinda been solved... the one I spoke to was beyond helpful" →
> `other`, `auto` — no action needed, acknowledge only.

## Escalation call, as a rule of thumb

Escalate when: money/fraud is in question, identity/account security is
involved, the customer explicitly asks for a human/supervisor, or the
message itself signals a prior automated attempt already failed
(`service_complaint`, or any intent where `human_notes` says "second time
this has come up"). Otherwise default to `auto` and let the reply-drafting
step decide how much it can actually resolve versus just acknowledge.

## A note on the AI-suggested columns

If you ran `prelabel_with_llm.py`, you'll see `ai_suggested_intent` /
`ai_suggested_escalate` / `ai_suggested_reason` columns. Treat these as a
first draft to speed up reading, not as the answer. Label independently
where you can, then compare — your agreement rate with these suggestions
(and where you overruled them) is itself worth reporting, the same way the
LLM-judge agreement number is required for the reply-quality rubric.
