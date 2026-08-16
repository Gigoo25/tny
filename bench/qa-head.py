#!/usr/bin/env python3
"""F115: an extractive QA head instead of a generator, the one shape F76/F79 never measured.

The model tny ships does two jobs: pick which sentence in ~3,200 chars answers the question
(worth 21 cases over the best model-free extractor, F76) and phrase it. A SQuAD-tuned encoder is
trained for the first job specifically -- answerhood, not relevance -- returns a span rather than
prose, and runs in 100-300 ms on CPU. llama.cpp cannot serve one: no span-classification head.

Reads the *same* context tny hands its own model (`tny --context`, cached in .contexts.json), so
this is an arm of F79's bake-off and not a different experiment. Writes the cache format
`bench/answer-cli.mjs` already grades, so the verdicts come from `bench/grade.mjs` unchanged:

    TNY_ANSWERS=bench/.answers-qa-span.json bun bench/answer-cli.mjs --regrade

transformers 5.x deleted the `question-answering` pipeline, so the span decode is here: top-k
starts and ends per window, best pair within a length cap, scored against the CLS null span that
SQuAD2 uses to abstain. An abstention is printed the way tny prints a refusal, because that is
what it is -- and a refusal is the safe failure the whole grounding design is built around.

Two arms, because a bare span and a span in its sentence are different products:
  .answers-qa-span.json      the span alone -- "resize2fs", "115"
  .answers-qa-sentence.json  the sentence containing it, which is what a user reads
"""
import json
import os
import re
import subprocess
import sys
import time

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL = os.environ.get("QA_MODEL", "deepset/minilm-uncased-squad2")
MAX_ANS_TOK = 40

CASES = json.loads(subprocess.run(
    ["bun", "-e", """
const f=["instructional","qa","general","ambiguous"];const out=[];
for(const n of f){const {CASES}=await import(`./bench/fixture-${n}.mjs`);for(const c of CASES)out.push(c[0]);}
console.log(JSON.stringify(out));"""],
    capture_output=True, text=True, check=True).stdout)

CTX_CACHE = "bench/.contexts.json"
ctx = json.load(open(CTX_CACHE)) if os.path.exists(CTX_CACHE) else {}
env = {**os.environ, "TNY_MODE": "medium", "TNY_LEN": "medium", "TNY_MODEL": "0.8b"}
for i, q in enumerate(CASES):
    if ctx.get(q):
        continue
    print(f"  context {i + 1}/{len(CASES)}", file=sys.stderr)
    p = subprocess.run(["./target/release/tny", "--context", q], capture_output=True, text=True, env=env)
    ctx[q] = p.stdout.strip()
    json.dump(ctx, open(CTX_CACHE, "w"), indent=1)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL).eval()


def answer(question, context):
    """Best span across every window, or "" when the null span wins."""
    enc = tok(question, context, truncation="only_second", max_length=384, stride=128,
              return_overflowing_tokens=True, return_offsets_mapping=True,
              padding=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping", None)
    with torch.no_grad():
        out = model(**enc)
    best, best_span = float("-inf"), None
    for w in range(out.start_logits.shape[0]):
        s, e = out.start_logits[w], out.end_logits[w]
        # Per window, never pooled: the null span of a window that does not contain the answer
        # is high, and pooling it across windows vetoed every span found in another window.
        wnull = (s[0] + e[0]).item()
        # Only context tokens are candidates: a span inside the question is not an answer.
        keep = [j for j, sid in enumerate(enc.sequence_ids(w)) if sid == 1]
        if not keep:
            continue
        starts = sorted(keep, key=lambda j: -s[j])[:20]
        ends = sorted(keep, key=lambda j: -e[j])[:20]
        for a in starts:
            for b in ends:
                if b < a or b - a > MAX_ANS_TOK:
                    continue
                sc = (s[a] + e[b]).item()
                if sc > best and sc > wnull:
                    best, best_span = sc, (offsets[w][a][0].item(), offsets[w][b][1].item())
    if best_span is None:
        return ""
    return context[best_span[0]:best_span[1]].strip()


span_cache, sent_cache, t0 = {}, {}, time.time()
for i, q in enumerate(CASES):
    c = ctx.get(q, "")
    t = time.time()
    span = answer(q, c) if c else ""
    ms = (time.time() - t) * 1000
    if not span:
        span_cache[q] = sent_cache[q] = {"ans": "not found", "err": "rejected — qa head abstained"}
        print(f"  {i + 1:2} {ms:6.0f}ms ABSTAIN  {q[:52]}", file=sys.stderr)
        continue
    at = c.find(span)
    left = max(c.rfind(".", 0, at), c.rfind("\n", 0, at)) + 1
    ends = [x for x in (c.find(".", at + len(span)), c.find("\n", at + len(span))) if x > 0]
    right = min(ends) if ends else len(c)
    sentence = re.sub(r"\s+", " ", c[left:right + 1]).strip()
    span_cache[q] = {"ans": span, "err": ""}
    sent_cache[q] = {"ans": sentence, "err": ""}
    print(f"  {i + 1:2} {ms:6.0f}ms {span[:38]:40} | {sentence[:58]}", file=sys.stderr)

json.dump(span_cache, open("bench/.answers-qa-span.json", "w"), indent=1)
json.dump(sent_cache, open("bench/.answers-qa-sentence.json", "w"), indent=1)
print(f"\n{MODEL}: {len(CASES)} cases in {time.time() - t0:.1f}s", file=sys.stderr)
