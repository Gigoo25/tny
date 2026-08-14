# tny — small-model offline terminal assistant

## Goal

`tny "question"` — a Rust CLI for day-to-day terminal Q&A. Smallest possible model
that still gives coherent, *correct* answers. **Fully offline**: all knowledge from
local ZIM files, searched like a private search engine. The model's job is to
*adapt* retrieved material to the exact question asked, not to recall facts.

Portable: any computer, CPU-only, no GPU assumed.

## Status

Sessions 1–2 (2026-08-13). Model choice, retrieval architecture, and corpus plan
are settled **by measurement**. No Rust written yet — deliberately; the
measurements redirected the design five times.

- `NOTES.md` — findings, append-only (this file)
- `PLAN.md` — build sequence
- `bench/harness.mjs` — runnable measurement harness (rebuilds every benchmark)

## Where to pick up

1. Read "Verdict" below, then `PLAN.md` phase P0.
2. Start the three servers (see "Environment"), then `bun bench/harness.mjs all`
   to reproduce the numbers before changing anything.
3. First code: `PLAN.md` P0 (supervise the servers) — nothing else is blocked.

Open decisions worth resolving before/while coding are in "Open questions".

---

## The one-paragraph summary

Retrieval, not model size, produces correct answers — but it cannot substitute for
capability. On the same six terminal questions with the answer verified present in
context, LFM2.5-350M scored 2/6 while Qwen3.5-0.8B scored 5/6; without any corpus
they scored 1/6 and 3/6. So **350M with the entire corpus still loses to 0.8B with
none of it**: retrieval multiplies whatever capability exists. Selection, however,
must never be done by the chat model — asking it to choose among candidates scored
*worse than free Xapian rank-1*, while a 35 MB embedding model fused with Xapian by
Reciprocal Rank Fusion scored 9/10. Final shape: Xapian + bge-small select, and
Qwen3.5-0.8B (thinking disabled) adapts.

## Verdict

| Role | Choice | Size | Evidence |
|---|---|---|---|
| Answering | **Qwen3.5-0.8B Q8_0, `enable_thinking:false`** | ~0.9 GB | F20, F21, F23 |
| Selecting + section ranking | **bge-small-en-v1.5 Q8_0** | ~35 MB | F17, F22 |
| Knowledge | **ZIM files via `kiwix-serve`** | 82 MB → 18 GB tiers | F11–F14 |
| Rejected as answerer | LFM2.5-350M | — | 2/6 with perfect context (F20/F23) |
| Rejected outright | LFM2.5-1.2B-Thinking | — | never answered at all (F1) |
| Rejected mechanism | model-as-judge | — | worse than free rank-1 (F16) |

---

## Guiding result (external)

"Honey, I Shrunk the Coding Agent" (Itay Inbar, Apr 2026) —
<https://itayinbarr.substack.com/p/honey-i-shrunk-the-coding-agent>

Same 9B weights: Aider scaffold 19.11% → purpose-built `little-coder` 45.56% on
Aider Polyglot. **Scaffold–model fit, not model size, was the dominant variable.**
Transferable mechanisms: bounded thinking budget with abort-and-retry; structured
*directive* tool errors ("file exists, use Edit"); malformed tool-call repair;
loop/repetition abort; catch empty responses; small per-turn context injections
rather than one big preamble; retry-with-feedback worth 18% of passes; and
supervise the runtime — their Ollama server died mid-run and silently degraded
results.

Prior art for the UX: **`samtay/so`** <https://github.com/samtay/so> (Rust, 1.4k★,
terminal Stack Overflow browser). It **dropped DuckDuckGo as default because DDG
blocks requests** ([issue #16](https://github.com/samtay/so/issues/16)),
independently confirming what we measured. `tny` differs in kind: `so` shows an
existing answer; `tny` rewrites retrieved material for your specific question.

## Measurement rig ≠ target machine

A deliberately weak, busy box (browser + editor open), ±10% noise. Treat the
*shape* of each result as the finding, not the absolute seconds.

- Intel i5-5250U (Broadwell, 2 cores / 4 threads), AVX2 + FMA + F16C, no AVX-512
- 7.7 GiB RAM, ~1.3 GiB free during tests
- llama.cpp b10273 (`a6aa6f5`), auto-dispatch to `libggml-cpu-haswell.so`
- kiwix-tools 3.8.2 (libkiwix 14.2.1, libzim 9.8.1, **xapian 1.4.31**)

## Environment

```sh
nix-shell -p llama-cpp     # llama-cli, llama-server, llama-bench
nix-shell -p kiwix-tools   # kiwix-serve, kiwix-search, kiwix-manage
export LLAMA_CACHE=$PWD/models
```

The three servers used for every measurement:

```sh
# knowledge (port 8082)
kiwix-serve --port 8082 --address 127.0.0.1 zim/*.zim
# answering (port 8080)
llama-server -hf ggml-org/Qwen3.5-0.8B-GGUF:Q8_0 --no-mmproj -t 4 -c 8192 --jinja \
             --port 8080
# selection (port 8084)
llama-server -hf ggml-org/bge-small-en-v1.5-Q8_0-GGUF --embeddings --pooling cls \
             -c 512 -t 4 --port 8084
```

`--jinja` is required for the native LFM2.5 tool-call handler (F6). `--no-mmproj` is
required for every Qwen3.5 GGUF, which are vision models whose 671 MB projector
llama.cpp otherwise downloads and loads for nothing (F25). Add
`--reasoning-format none` only when debugging thinking output (F19).

## On disk

```
zim/devdocs_en_bash_2026-04.zim          545 KB   132 articles, _ftindex:no
zim/devdocs_en_rust_2026-07.zim          5.9 MB  2,983 articles, _ftindex:no
zim/archlinux_en_all_maxi_2026-07.zim     34 MB 14,497 articles, _ftindex:YES
models/  LFM2.5-350M Q8_0, LFM2.5-230M Q8_0, LFM2.5-1.2B-Thinking Q4_0+Q4_K_M,
         Qwen3.5-0.8B Q8_0, bge-small-en-v1.5 Q8_0        (all via llama.cpp -hf)
         Qwen3.5-2B Q4_K_M, byte-verified 1,280,835,840 B, at
           models/Qwen3.5-2B-Q4_K_M.gguf — kept for the harder fixture (F26);
           serve with `-m` since the `-hf` blob was the corrupt one
```

## Models evaluated

| Model | Quant | Size | Params | Notes |
|---|---|---|---|---|
| LFM2.5-1.2B-Thinking | Q4_0 / Q4_K_M | 661 / 695 MiB | 1.17 B | rejected, F1 |
| LFM2.5-350M | Q8_0 | 359 MiB | 354 M | rejected as answerer, F20 |
| LFM2.5-230M | Q8_0 | 233 MiB | 230 M | faster, weaker extraction |
| **Qwen3.5-0.8B** | **Q8_0** | **~0.9 GB** | 0.8 B | **chosen answerer** |
| **bge-small-en-v1.5** | **Q8_0** | **~35 MiB** | 33 M | **chosen selector**, 384-dim |

LFM2.5 are `lfm2` arch: hybrid, only 6 attention layers → small KV cache. Vocab
65536, context 32768, knowledge cutoff mid-2024. At these sizes **Q8_0 is the right
default** — 230M Q8_0 (233 MiB) is *smaller* than 1.2B Q4_0 (661 MiB) and
near-lossless; quantisation damage hurts tiny models proportionally more.

---

# Part 1 — model findings

## F1 — the 1.2B Thinking model is the wrong tool

"How do I find files larger than 100MB in bash?", `-n 900`, temp 0.05: spent **all
900 tokens thinking, never answered.** 107 s, truncated mid-sentence. Hallucinated
`-type l`, doubted whether `find` has `-size`, recomputed "100MB in bytes" three
times, converging on `-size +104857600` — **wrong** (bare `-size` counts 512-byte
blocks; correct is `+100M` or `+104857600c`). Both non-thinking models got it right
in ~10 s.

## F2 — reasoning-budget flags exist but do not fix deliberation

llama.cpp ships `--reasoning-budget N`, `--reasoning-budget-message`, `-rea
on|off|auto`, `--reasoning-format`, `--reasoning-preserve`. On the 1.2B:

| Config | Result |
|---|---|
| `--reasoning-budget 200` | **Works** — injects `</think>`. Model then deliberates in plain text. |
| `--reasoning-budget 0` | Emits `<think></think>`, rambles in content channel anyway. |
| `-rea off` | **Empty output.** Broken for this template. |
| budget + steering + terse prompt | Right answer first, then relapsed into "Duplicate? No, better:…" |

The habit is in the weights, not the tags.

## F3 — CPU knobs are machine-dependent → autotune, never hardcode

1.2B thread sweep, pp512 / tg64, r=3:

| Quant | t=1 | t=2 | t=3 | t=4 |
|---|---|---|---|---|
| Q4_0 pp512 | 11.92 | 20.85 | 24.77 | **29.54** |
| Q4_0 tg64 | 6.50 | 10.11 | 11.09 | **12.64** |
| Q4_K_M pp512 | 15.98 | 25.03 | **27.60** | 27.75 |
| Q4_K_M tg64 | 6.94 | 10.45 | **11.39** | 9.72 |

- Hyperthreads **helped Q4_0** (t=4 > t=2 on 2 physical cores) but **hurt Q4_K_M**.
- **BLAS is not worth it**: pp512 27.57 vs 27.31 t/s, inside noise. Ship CPU-only.
  (To test: copy `llama-bench` + `libggml-cpu-*.so` to an empty dir;
  `GGML_BACKEND_PATH` does not suppress backends in the binary's own dir.)
- **Flash attention ≈ +15–20% decode** (tg32 10.9–11.8 vs 8.1–9.6); `auto` enables it.
- `--poll 0` vs `50`: noise.

## F4 — small non-thinking models answer fast

temp 0.1 / top_k 50 / rep-pen 1.05, t=4:

| Model | Prefill t/s | Decode t/s | "files > 100MB" |
|---|---|---|---|
| 1.2B Thinking Q4_0 | 21.9 | 8.6 | **no answer** (burned 900 tok) |
| 350M Q8_0 | 53.5 | 15.4 | correct (`find . -type f -size +100M`) |
| 230M Q8_0 | 82.6 | 22.5 | correct, cleanest phrasing |

**Model load: 359 MiB Q8_0 in 1.5 s** — a resident server is a latency
optimisation, not a requirement.

## F5 — they fabricate confidently → lookup is mandatory

"What is C. elegans?" from memory:

- **350M**: "commonly known as the 'little worm'", "first discovered in 1878 by the
  German zoologist Karl von Frisch", "three main parts: head, thorax, and abdomen",
  "the thorax contains the exoskeleton". That is insect anatomy; von Frisch (b. 1886)
  studied bees.
- **230M**: "Contains only 2,000 genes" (actually ~20,000).
- Rust compiler version: **"4.0.0"** on one run. With retrieved text: **1.97.1** ✓.

## F6 — llama.cpp has native LFM2.5 tool-call support

`libllama-common.so` (b10273) contains `Using specialized template: LFM2.5`,
`tool_call_start_marker`, `<|tool_call_start|>`, `<|tool_call_end|>`, python-or-json
arg parsing. Verified live with `--jinja` + OpenAI `tools`:

```
"What is in the file ./Cargo.toml?"  -> read_file  {"path":"./Cargo.toml"}
"Who won the 2026 Super Bowl?"       -> web_search {"query":"who won the 2026 Super Bowl"}
"What is 12 * 7?"                    -> direct, "84"
```

No tool-call parser or GBNF grammar needed. **`tool_choice:"required"` is silently
ignored** — the harness must decide and inject results itself. (With ZIM-only
retrieval and harness-side routing, `tny` may not need tool calling at all; kept
here because it is verified and cheap.)

## F7 — routing: positive phrasing wins, negations backfire

| Variant | Score /10 | Failure |
|---|---|---|
| no system prompt | 7 | misses version lookups; never reads files |
| aggressive | 7 | over-searches `12 * 7`, `chmod 755` |
| **V2 (short, positive)** | **8, 8, 9** | misses "Summarize the file src/main.rs" |
| V3 (+ "Do NOT search for arithmetic") | 6 | **searched arithmetic more** |
| V4 (V2 + explicit read_file rule) | 7, 7 | more spurious searching |

**Negations backfire at this scale.** V2 is the ceiling for prompt-only routing, so
path-like queries must be handled deterministically in the harness.

## F8 — fabrication lives in the elaboration

Headline fact right, *tail* invented: Rust "**1.97.1**" ✓ then fake release dates;
"Linda Yaccarino" ✓ then "since November 2021", "CEO from 2015 to 2021", plus
"Before joining Yaccarino".

| Contract | max_tok | "How many neurons does C. elegans have?" |
|---|---|---|
| terse | 80 | "C. elegans has 302 neurons." ✓ |
| verbose | 250 | "302" ✓ then "a single brain and body wall muscles" ✗ |

## F9 — ⚠️ never give the model a refusal escape hatch

The contract ended "If the results do not contain the answer, reply exactly: not
found in results." On prose that looked safe. On **code it collapses**:

| System prompt | Context | Result |
|---|---|---|
| with escape clause | Stack Exchange answer, fenced code | "not found in results." |
| with escape clause | link-refs denoised | "not found in results." |
| with escape clause | fences stripped | "not found in results." |
| with escape clause | `>>>` prompts stripped | "not found in results." |
| **escape clause removed** | identical context | **correct answer + working code** |

The answer (`list(reversed(xs))`) was present every time. **Detect emptiness in the
harness; never delegate it to the model.**

## F10 — ⚠️ methodology: always assert the answer was retrievable

Two experiments were invalidated by skipping this. Mid-session DuckDuckGo began
serving CAPTCHAs, so the model was fed **empty** context and its correct "not found"
replies looked like over-refusal. Every retrieval test must first assert the answer
is present in the fed context, or you are measuring your own pipeline. Implemented
as `oracle()` / the `has` flag in `bench/harness.mjs`.

---

# Part 2 — retrieval architecture (offline, ZIM-only)

**Decision: all knowledge comes from local ZIM files served by `kiwix-serve`.** One
mechanism, offline, no API keys, no rate limits, no CAPTCHAs, no network latency.
Supersedes the earlier four-web-backend design (Wikipedia API, Stack Exchange API,
DevDocs HTTP, Grokipedia scraping), retained only as a possible online fallback for
staleness. Those APIs *were* verified working and are documented in git history if
needed.

## F11 — kiwix-serve API surface

| Need | Endpoint | Notes |
|---|---|---|
| list books | `/catalog/v2/entries?count=-1` | OPDS Atom: `<name>`, `<tags>`, `articleCount` |
| full-text search | `/search?books.name=<id>&pattern=<q>&format=xml&pageLength=N` | RSS `<item>` = title + link + **snippet** |
| fuzzy title lookup | `/suggest?content=<id>&term=<t>` | JSON, substring, returns `path` |
| fetch article | `/content/<id>/<path>` | article HTML |
| random article | `/random?content=<id>` | smoke tests |

**Book id is the filename stem** (`devdocs_en_rust_2026-07`), *not* the ZIM's
internal `<name>` (`devdocs_en_rust`) — the latter gives `400 No such book`. Param
names are strict: `/suggest` needs **`content=`** (`books.name=` and `book=` both
404); `/search` needs **`books.name=`**.

## F12 — [CORRECTED] `_ftindex:no` does NOT mean unsearchable

Catalog `<tags>` carries `_ftindex:yes|no`:

| ZIM | articles | ftindex |
|---|---|---|
| `archlinux_en_all_maxi` | 14,497 | **yes** |
| `devdocs_en_rust` | 2,983 | **no** |
| `devdocs_en_bash` | 132 | **no** |

**The original conclusion here was wrong.** I recorded that `/search` is unavailable on
DevDocs ZIMs and that retrieval strategy must therefore be chosen per book from its
tags. That was inferred from `HTTP 400 Invalid request`, which was really a
**name-format error**: `books.name` wants the *filename stem*, not the catalog `<name>`.

| Request | Result |
|---|---|
| `books.name=devdocs_en_bash` (catalog name) | **400 Invalid request** |
| `books.name=devdocs_en_bash_2026-04` (file stem) | **200, 3 hits** |
| no `books.name` at all — every mounted ZIM | **200, 3 hits** |

And it is genuine body-level full text, not title matching. `pattern=Bourne Again SHell
acronym` against the `_ftindex:no` bash ZIM returns §Basic Shell Features with the
snippet *"Bash is an acronym for 'Bourne-Again SHell'"* — prose that appears in no
title. `pattern=reallocating minimum capacity` likewise hits `std::ffi::OsString` body
text in the rust ZIM.

**Consequences for the design:**

1. **No per-book routing stage.** One `/search` with no `books.name` searches every
   mounted ZIM and returns each hit's book in its `<link>` — content, not configuration,
   picks the corpus. "swap file" → archlinux, "Vec with_capacity" → rust,
   "shell parameter expansion" → bash.
2. **`/suggest` is not needed for *search*,** only for exact title→path lookup (F13).
   Its parameter is `content=<file stem>`, not `book=` (that 404s).
3. `_ftindex` remains worth reading for *diagnostics*, but must not gate retrieval.


## F13 — DevDocs ZIMs: suggest → path → anchor

- Titles are fully qualified: `/random` returned `std::fmt::UpperHex`.
- `/suggest?term=Vec` → `std::vec::Vec` → `std/vec/struct.vec`; `term=HashMap` →
  `std::collections::HashMap`. Substring, not just prefix.
- **Method-level titles are absent**: `term=std::vec::Vec::with` → only a `pattern`
  row. Entries are page-level.
- **Anchors survive inside the page**: `std/vec/struct.vec` is 457 KB with **240**
  `id="method.*"` anchors including `method.with_capacity`.
- The devdocs ZIM root article is *not* an index (2.4 KB, 10 links to the Rust book),
  so entries cannot be enumerated for free.

So: **`/suggest` for the page, then slice by `#anchor` for the member** — no local
cache, no `index.json`. Match exact name → prefix → substring; a bare substring
match on "with_capacity" returns nightly-only `try_with_capacity`.

## F14 — `_maxi` ZIMs are polluted with localised duplicates

The *English* Arch ZIM contains `Netboot (Magyar)`, `Solid state drive (Magyar)`, …
At `pageLength=6` these consumed half the candidate list (only 2–4 of 6 survived).
**Request ~30 candidates, strip `(<Language>)` titles, dedupe by base title, keep
top 8.**

## F15 — query preparation is needed for Xapian recall

"why is my wifi not connecting" returned **zero** usable candidates raw. Stripping
question words and stopwords gave `"wifi connecting"` → 8 candidates including
NetworkManager and Netctl.

## F16 — ⚠️ the chat model is not worth its cost as a judge

The proposed design was "fuzzy-find candidates, let the model judge". Measured on 6
Arch queries with identical cached candidate lists:

| Selector | Score |
|---|---|
| Xapian rank-1 (free, deterministic) | **4/6** |
| 350M judge, forward order | 2/6 |
| 350M judge, reversed order | 4/6 |
| 350M judge, output title not index | 3/6 |
| **Qwen3.5-0.8B judge** | 3/6 · then 4, 4, 5 on re-runs |
| **RRF fusion (F17)** | **5/6** |

350M's picks were **#7, #7, #5, #7, #7, #8** forward and **#8, #7, #7, #7, #7, #7**
reversed — a near-constant index regardless of content. Reversing the list scores
4/6 only because it reshuffles which article lands in that fixed slot: luck, not
signal. Asking for the *title* removed the constant-index artifact (0 unmatched —
string copying works) but still lost to free rank-1. Qwen-0.8B's picks *did* vary
(3,1,6,1,5,1) so it attends to content, but it still lost to rank-1 and to RRF.

The information was present — correct articles sat at #4 (Udisks), #2 (USB flash
installation medium), #2 (NetworkManager). Selection is simply not a capability
these models have.

**Correction after three re-runs.** Qwen-0.8B's judge scored **4, 4, 5** against
rank-1's constant **4/6** — so at 0.8B the judge is *not* clearly worse than rank-1;
the original single 3/6 was inside sampling noise. The decision is unchanged but its
reason is narrower: the judge still loses to **RRF's 9/10** (F17), its picks are
heavily biased to index 1 (`4,1,6,1,1,1` / `4,1,1,1,1,1`) so it mostly *reproduces*
rank-1, and it costs a full model call and seconds of latency to do it. **Free and
deterministic beats paid and equal** — but "the model cannot judge" overstates the
evidence at 0.8B. The strong claim holds only for 350M, whose near-constant index is
a genuine pathology.

## F17 — hybrid retrieval + RRF fusion (9/10)

Arch titles do not lexically match natural questions — "Udisks" shares no term with
"mount usb drive automatically" — so no lexical or fuzzy re-rank can bridge it. The
gap is semantic, and a **33 M-param, 35 MB** model closes it. 10-query Arch
benchmark, top-8 candidates:

| Selector | Score |
|---|---|
| Xapian rank-1 | 8/10 |
| bge-small, query vs **title** | 8/10 |
| bge-small, query vs **title + snippet** | 8/10 |
| **RRF fusion of all three** | **9/10** |

Single signals tie but **fail on different queries** (rank-1 missed usb-mount and
bootable-usb; emb-title missed usb-mount and encrypt; emb+snippet missed timezone
and wifi) — exactly the condition where RRF wins. RRF is ~10 lines, deterministic,
no chat model: `score(d) = Σ 1/(k + rank_i(d))`, k=10.

Cost: one batched `/v1/embeddings` call (query + 8 titles + 8 title/snippet pairs =
17 texts) ≈ **2.3 s** on this weak box. bge-small requires the asymmetric query
prefix `"Represent this sentence for searching relevant passages: "`.

Remaining failure: "mount a usb drive automatically" → *Solid state drive*
(emb+snippet had Udisks right but was outvoted).

## F18 — answering is faithful only when the section is on-point

Oracle fixtures (section verified to contain the answer), 350M:

| Query | Section | Answer | Verdict |
|---|---|---|---|
| encrypt a partition | Encryption options for LUKS mode | "`cryptsetup luksFormat`" | **OK** |
| generate an ssh key | Generating an SSH key pair | `ssh-keygen`, correct `id_ed25519` path | **OK** |
| set the system timezone | UTC in Microsoft Windows | "set the system timezone to UTC." | ✗ |
| mount a usb drive automatically | Installation | **fabricated** `mount -t cdr usb-drive` | ✗ |

2/4 faithful. Both failures were fed *tangential* sections that merely contained the
keyword, and the model invented a plausible command rather than saying so. Section
selection is as load-bearing as article selection.

## F19 — ⚠️ Qwen3.5-0.8B thinking mode is unusable (verified from raw bytes)

Qwen3.5 is thinking-by-default. For the trivial prompt "Say OK." with
`max_tokens:60` it returned `content:""`, `finish_reason:"length"`, and
`reasoning_content:"Thinking Process:\n\n1. **Analyze the Input:** …"` — all 60
tokens spent deliberating.

Because empty `content` looks like a parsing bug, this was re-verified with
`--reasoning-format none`, which puts **raw** output (tags included) into `content`:

```
max_tokens=512: finish=length gen=512 in 95.5s
  content 1770ch | reasoning_content 0ch | closed </think>: false
  head: "<think>\nThinking Process:\n\n1.  **Analyze the Request:** …"
```

So the model opens `<think>` and **never closes it within 512 tokens**. The empty
answers were real, not a parse artifact. Earlier runs on real questions took 83–93 s
each and produced nothing. Whether 2048 tokens would eventually close is unmeasured
— 95.5 s for 512 tokens already disqualifies it for a terminal tool.

**Two hard requirements:**
1. Always send `chat_template_kwargs: {"enable_thinking": false}`. With it, the same
   prompt answers in one short turn.
2. **Treat empty `content` as an error**, never as an answer — llama.cpp routes
   reasoning to a separate field, so a naive client silently prints blanks.

## F20 — model comparison on identical contexts

Six terminal questions, top-3 embedding-selected Arch sections, answer verified
present in context **6/6**, `max_tokens:160`:

| Model | Correct | Latency |
|---|---|---|
| LFM2.5-350M Q8_0 | **2/6** | 7.2 s/answer |
| Qwen3.5-0.8B Q8_0 (thinking off) | **5/6 – 6/6** | 15.3–17.5 s/answer |
| Qwen3.5-0.8B Q8_0 (thinking on) | **0/3** before abort | 83–93 s/answer |

Throughput caveat: a fresh, uncached Qwen call was **372 prompt + 85 generated
tokens in 25.5 s**. An earlier "8.4 s/answer" figure was a prompt-cache artifact —
Qwen is genuinely ~3× slower than 350M, not equal.

Score variance: two runs of the identical benchmark gave **5/6 and 6/6** — the
"check what is using disk space" case flipped from "Which tool is being used to
check disk usage?" to "Use `du` or `ncdu`." At temp 0.1 a single benchmark run is
not a precise number; treat these as ranges and re-run before trusting a delta of 1.

Cosmetic note: with `--reasoning-format none` and `enable_thinking:false` the
template still emits an empty `<think>  </think>` into `content`. Harmless, but the
harness and `tny` should strip it.

## F21 — does a good corpus let a small model punch above its weight?

Same six questions, with and without reference material:

| Model | no corpus | with corpus |
|---|---|---|
| LFM2.5-350M | 1/6 | **2/6** |
| Qwen3.5-0.8B | 3/6 | **5/6** |

**Yes, but it multiplies capability rather than substituting for it.** Retrieval is
the single biggest lever — it converts confident, *dangerous* fabrication into
correct commands. Unretrieved, 350M proposed `swapfile.exe /dev/sdb` (a Windows
binary) and Python `cryptography.fernet` to encrypt a *partition*; Qwen proposed
`mkfs -f` with "`xfs -f -f`" — running that destroys data.

But the decisive comparison: **350M *with* the whole corpus (2/6) still loses to
Qwen *without* any of it (3/6).** Good data cannot close a capability gap.

## F22 — section selection: embeddings win, fusion does not

Six probes, "is the answer inside the chosen section?":

| Method | Score |
|---|---|
| heuristic heading scorer (`Σ|term| / √words`) | 2/6 |
| **bge-small embedding, argmax section** | **4/6** |
| RRF(embedding, heuristic) top-1 | 4/6 |
| RRF top-2 | 4/6 |
| RRF top-3 | 5/6 |
| **embedding, top-3 sections × 600 chars** | **6/6** |

Two lessons:
1. **Fuse at article level, embed-only at section level.** Fusion *hurt* here — the
   heuristic displaced embedding's correct §"du alternatives".
2. **Stop chasing a perfect argmax.** Three short sections (~1570 chars total) put
   the answer in context 6/6, which argmax never did.

Earlier heuristic failure worth remembering: matching bare tokens let "system" score
against "Set hardware clock from system clock" as strongly as "timezone" scored
against "Time zone"; normalising away spaces was necessary but insufficient.

## F23 — 350M's failure mode with perfect context is degeneration

Given the answer in context, 350M did not merely pick wrong facts — it stopped
answering:

| Query | 350M output |
|---|---|
| generate an ssh key | "generate an ssh key" (echoed the question) |
| encrypt a partition | "Encrypt a partition using LUKS mode." (no command) |
| check what is using disk space | "Check the command that lists disk usage." |

Instruction-following collapse, not missing knowledge — and unfixable by better
retrieval. This is why 350M is rejected as the answerer.

## F24 — a bigger, purpose-built embedder is *worse* here

Hypothesis: bge-small is a general-purpose 33 M model, so a retrieval-specialised
one should rank better. Tested `nomic-embed-text-v1.5` Q8_0 (137 M, 4× the size,
trained for retrieval, `search_query:` / `search_document:` prefixes) on the same
two benchmarks:

| Signal | bge-small (33 M) | nomic (137 M) |
|---|---|---|
| article RRF | **9/10** | 8/10 |
| emb·title | **8/10** | 6/10 |
| emb·title+snippet | 8/10 | 8/10 |
| section presence (top-3) | 6/6 | 6/6 |
| wall time | ~21 s (sections) | ~3–5× slower |

**bge-small stays.** The collapse is in *title* similarity (8→6): article ranking is
dominated by 1–3 word titles ("Udisks", "Systemd"), and nomic is tuned for
passage-length text. Sections tie on presence, though nomic's picks read better
qualitatively (§"Encrypting devices with LUKS mode" ranked first vs bge's
§"Resizing encrypted devices").

Corollary: embedder scaling is **not** the lever. `bench/harness.mjs` takes
`TNY_EMBED`, `TNY_QP`, `TNY_DP`, so re-testing another embedder
(embeddinggemma-300m, LFM2.5-Embedding-350M, Qwen3-Embedding-0.6B) is a one-line
experiment if it ever looks worthwhile.

## F25 — two operational hazards in llama.cpp's `-hf` downloader

Both cost real time this session and both must be handled by `tny`:

1. **Qwen3.5 GGUFs are multimodal.** `-hf unsloth/Qwen3.5-2B-GGUF:Q4_K_M` silently
   began downloading `mmproj-BF16.gguf` (671 MB) that a text-only tool never uses;
   the server sat in `starting` for 10 minutes. Fix: **always pass `--no-mmproj`**.
   With it, the same server was ready in 5.7 s.
2. **An interrupted download is left looking complete.** After killing the mmproj
   fetch, `models/…/blobs/` held a 945,661,018-byte file with no
   `.downloadInProgress` suffix. That size matches **no file** in the repo
   (Q4_K_M is 1,280,835,840 — it was 74% of one). llama.cpp loaded it without a
   single warning and the model emitted pure ASCII garbage:
   `DIO*=C,B5O%%NFH@OB->KB;M@R:4(1Q+…` for every prompt, 0/6, 60.2 s/answer.

**Never trust a model file's presence — verify its byte size** (HF exposes it via
`/api/models/<repo>/tree/main`) before serving, and treat non-text output as a
corrupt-model signal rather than a capability result. This nearly got recorded as
"2B is bad".

## F26 — 2B measured: no accuracy gain, 2.2× the cost

Qwen3.5-2B Q4_K_M (1.28 GB, byte-verified), served `--no-mmproj`, on the identical
fixtures:

| Benchmark | Qwen3.5-0.8B Q8_0 | Qwen3.5-2B Q4_K_M |
|---|---|---|
| answering, needle present 6/6 | **6/6** | **6/6** |
| latency (cold) | **12.7–17.5 s** | 33.9 s |
| no corpus → with corpus | 3/6 → 5–6/6 | 3/6 → 6/6 |
| refusal on mismatched context | 4–5/6 | 5/6 |

**The 6-case answering fixture is saturated** — both models score 6/6, so it can no
longer discriminate and any further model comparison needs harder cases.

2B fabricates just as dangerously without a corpus: it proposed **`mkfs.ext4` to
"mount" a USB drive** (that formats the disk) and an invented
`cryptsetup --keyring …`. Scale does not fix fabrication; the corpus does.

New benchmark `refuse` supplies the discrimination that was missing: each question
is paired with a *different* question's sections, so the answer is absent by
construction and declining is the only correct behaviour. Two distinct failure
modes appeared:

- **both models**: answered `ssh-keygen -t rsa -b 4096` from parametric memory —
  *correct but unfaithful*, which is the dangerous class, because the same reflex
  fires when memory is wrong.
- **0.8B only**: echoed the question back verbatim (`"create a swap file"`) — the
  F23 degeneration seen at 350M, surfacing at 0.8B under harder conditions.

## F27 — a model-free grounding check beats the model upgrade

Both F26 failure modes are detectable **without a model**, so 2B's single advantage
is purchasable for free. `ungrounded(answer, ref, question)` returns a reason string,
or `""` when the answer is grounded. Final rules, after three defects found by
sampling:

1. **Every command the answer proposes must appear in the reference.** Commands are
   collected from all three forms models actually emit — inline `` `cmd` ``, fenced
   blocks, and prompt lines (`# cmd`) — matched at word boundaries.
2. If the answer proposes **no** command, reject it when it merely **restates the
   question**, or when it is **only a question** (`/^[^.!]*\?\s*$/`).
3. Empty content is rejected (F19).

| Metric | model alone | model + F27 |
|---|---|---|
| refusal, mismatched context (0.8B) | 4–5/6 | **6/6**, three runs |
| refusal, mismatched context (2B) | 5/6 | **6/6** |
| false rejects on correct answers | — | **0/30** samples at temp 0.3 |
| wrong answers let through | — | **0/30**; the one wrong answer was caught |
| self-test | — | **17/17**, pure, no servers: `harness.mjs ground` |

### The three defects, all found by sampling rather than by one run

A single benchmark run reported "0 false rejects" three separate times while the
check was still broken. Only repeated sampling at temp 0.3 exposed each fault, so the
rules are only trustworthy to the extent they were *attacked*:

1. **Substring matching.** `ref.includes("du")` is satisfied by "pro**du**ce".
   Fixed with word-boundary regex.
2. **Minimum command length.** A `length > 2` filter silently dropped `du`, `df`,
   `ls`, `ip` — the commands users ask about most — so `` Use `du -h` … `` fell
   through to rule 2 and was rejected. The first live run passed only because `ncdu`
   happened to also be present.
3. **Word-overlap ratio.** The original rule 2 (">60 % of words come from the
   question") rejected the *correct* answers `# timedatectl set-timezone` (a prompt
   line, so no command was extracted) and `Use timedatectl set-timezone to set the
   timezone.` (unmarked command). It had caught exactly **one** real failure while
   producing two false-reject classes, so it was **deleted** in favour of exact
   containment. A false reject is strictly worse than a missed echo: it converts a
   correct answer into "not found" and destroys trust in the tool.

Two further live findings folded in: a quoted **path**
(`/home/username/.ssh/id_ed25519`) is not a command and must be skipped, and the
model sometimes **asks a question back** instead of answering — a wrong answer the
first version let through.

**Decision: 0.8B stays the answerer, and grounding is enforced in `tny`, not
delegated to the model.** A bigger model is the expensive way to buy what a regex
already provides — but the regex must be attacked by sampling before it is believed.

## Corpus catalogue (English, `library.kiwix.org`, 1,286 ZIMs)

| ZIM | Size | Articles |
|---|---|---|
| `devdocs_en_bash` | 0.6 MB | 132 |
| `devdocs_en_git` / `_go` | 1.6 MB | 206 / 192 |
| `devdocs_en_postgresql` | 2.6 MB | 683 |
| `devdocs_en_javascript` | 2.7 MB | 1,291 |
| `devdocs_en_python` | 4.4 MB | 497 |
| `devdocs_en_rust` | 6.2 MB | 2,983 |
| `devdocs_en_man` | 29.6 MB | 12,626 |
| `archlinux_en_all` | 35.6 MB | 14,497 |
| `www.mankier.com_en_all` | 190 MB | 73,481 |
| `wikipedia_en_100` | 4.6 / 15.3 / 52.7 / 332 MB | 5,032 |
| `security.stackexchange.com_en_all` | 440 MB | 132,039 |
| `softwareengineering.stackexchange.com` | 479 MB | 129,851 |
| `codereview.stackexchange.com_en_all` | 551 MB | 136,194 |
| `dba.stackexchange.com_en_all` | 702 MB | 177,961 |
| `unix.stackexchange.com_en_all` | 1.31 GB | 413,259 |
| `wikipedia_en_wp1-0.8` | 2.35 / 8.49 GB | 855,632 |
| `docs.python.org_en_all` | 2.93 GB | 20,068 |
| `wikipedia_en_all` | **12.53 / 52.69 / 123.98 GB** | ~19,000,000 |
| `stackoverflow.com_en_all` | **80.48 GB** | 30,138,063 |

- **The whole API-reference layer costs < 50 MB** (rust + python + bash + git + go +
  javascript + man ≈ 46 MB). `devdocs_en_python` (4.4 MB) does the same job as
  `docs.python.org_en_all` (2.93 GB) at 1/660 the size.
- **Arch Wiki at 35.6 MB is the best value per byte** for terminal questions.
- **Avoid `stackoverflow.com_en_all` (80 GB)**; topical dev sites are 0.44–1.31 GB,
  and `unix.stackexchange` is the highest-yield single addition.
- Wikipedia: `wikipedia_en_wp1-0.8` mini (2.35 GB, 855 k important articles) is the
  sweet spot; `wikipedia_en_100` (332 MB) the minimal option.
- Download URLs arrive as `.zim.meta4` — strip `.meta4`. Host `lb.download.kiwix.org`.

## Latency budget (Qwen-0.8B + bge, this weak box)

| Stage | Time |
|---|---|
| Xapian search (local) | ~10 ms |
| article fetch + section split (local) | ~20 ms |
| embedding re-rank, 17 texts batched | 2.3 s |
| answer, ~1570-char context, ≤160 tok | 17.5 s |
| **total, warm servers** | **~20 s** |

Prefill dominates. On any modern multi-core machine expect several× better; this box
is a 2-core 2015 Broadwell under browser load. Levers, best first: tighter sections;
fewer candidates to embed; smaller `max_tokens`; 350M for the *selection-only* paths.

## Sampling defaults

- Qwen3.5-0.8B: `temperature 0.1`, `top_k 50`, `repeat_penalty 1.05`,
  **`enable_thinking:false`**
- LFM2.5 (if reused): `temperature 0.1`, `top_k 50`, `repetition_penalty 1.05`

---

## Design decisions

1. **Qwen3.5-0.8B Q8_0 answers, thinking disabled** (F19, F20, F21) — and it stays
   the default: 2B scored identically at 2.2× the latency (F26).
2. **Treat empty `content` as an error** (F19).
3. **bge-small-en-v1.5 selects sections only** — at article level Xapian + lexical
   RRF matches it 9/10 for free (F17, F22, F31).
4. **The chat model never chooses** (F16).
5. **ZIM-only knowledge via supervised `kiwix-serve`** (F11).
6. **Retrieval strategy per book, from its `_ftindex` tag** (F12).
7. **Structural extraction**: `#anchor` for reference pages, top-3 embedded sections
   for wiki pages (F13, F22).
8. **Fuse at article level (Xapian + lexical, no embedder); embed-only at section
   level** (F17, F22, F31).
9. **Query prep before search** (F15); **filter localised dupes and dedupe** (F14).
10. **Denoise retrieved text** (F8).
11. **Terse contract, `max_tokens` ≈ 160, no refusal escape hatch** (F8, F9).
12. **Emptiness detected in the harness** (F9).
13. **Positive phrasing only** (F7).
14. **Print the source** — book · article · sections.
15. **Never answer factual questions from weights** (F5, F21).
16. **Autotune threads only after measuring it matters** for the shipping model (F3).
17. **Enforce grounding in code, not by asking the model** — reject any answer
    proposing a command absent from the reference, or merely restating the question
    (F27). This buys 2B's only measured advantage for 12 lines.
18. **Always `--no-mmproj`, and verify model byte size before serving** (F25).
19. **Split sections on h2–h5 and centre the window on query terms** — coarse h2-only
    chunks put the answer past the slice budget in a 12.9 KB section (F30, F31).
20. **Keep conversation history; never let the model rewrite the follow-up query.**
    History carries the antecedent for elliptical follow-ups (F28); model rewrites
    inverted the question's meaning (F29). Build the retrieval query as `q1 + " " + q2`.
21. **Three grounding rules, not one, and each needs its own reference** (F27/F32, F44,
    F45): commands against the source document, claims about a comparison's other side
    against the slice actually shown, numbers and identifiers against the document, and a
    how-to answer must name a command from the reference's vocabulary.
22. **Build the command vocabulary from `<code>` *and* `<a>` text** (F45). Wikis name
    tools as links; `<code>` alone gave one article a six-word vocabulary and false-rejected
    a correct answer.
23. **Re-run `refuse` on every model or quant swap, never just `answers`** (F43, F46).
    Accuracy parity has twice concealed a safety regression, because what a deterministic
    checker catches depends on the *shape* of the model's errors.

## F28 — follow-up turns need conversation history (24 samples per arm)

A follow-up is elliptical: "how do I unlock **it** at boot", "how do I remove **one**
instead". The antecedent is only in the previous turn, so the question is whether to
keep chat history or to rebuild a stateless prompt from the retrieved reference.

Both arms get the same re-retrieved reference; the stateless arm restates the pair as
`"<first question> — specifically: <follow-up>"`.

| Arm | Correct | Latency | Prompt tokens |
|---|---|---|---|
| **with history** | **20/24 (83 %)** | 29.6 s | 761 |
| stateless | 18/24 (75 %) | 24.1 s | 381 |

The failure *distribution* decides it, not the 2-point gap: the stateless arm failed
**4 out of 4** attempts at "how do I unlock it at boot" — it cannot know that "it" is
an encrypted partition — while history's failures were spread thin. History carries
the antecedent; a re-retrieved reference does not.

Single runs of this benchmark returned 4/6, 5/6 and 6/6 for the *same* arm, so the
per-arm scores here are the only trustworthy ones. Prompt-cache reuse also makes
history cheaper than its token count suggests: `cached_tokens` was 350–440 of a
700–870-token turn-2 prompt, because turn 1's prefix is still in the KV cache.

**Decision: keep the conversation, append turns, let the prompt cache absorb it.**

## F29 — [WARN] never let the model rewrite the follow-up query

To retrieve for turn 2, the elliptical follow-up must become a standalone query.
Three ways, scored by whether the right article comes back rank-1:

| Query construction | Right article rank-1 | Cost |
|---|---|---|
| raw follow-up alone | 2/6 | free |
| **concatenate both turns** | **5/6** | **free** |
| model rewrite (0.8B) | 4/6 | 3.5 s/turn |

The rewrites were not merely worse, they were **semantically inverted**:

- "how do I turn it off again" → `"How do I turn a swap file back on?"`
- "how do I remove one instead" → `"pacman -S <package-name>"` (that installs)
- "how do I see only the failed ones" → `"find service --type=systemd --failed | grep -v "Failed"` (invented command)

A query that means the opposite of the question retrieves confidently wrong material,
which is worse than retrieving nothing. Same lesson as F16: the model must not be put
in charge of *selection*, and query construction is selection.

**Decision: `q1 + " " + q2`. Free, cannot invert meaning.**

## F30/F31 — section granularity was the real retrieval ceiling

"disable root login over ssh" was unanswerable and it exposed the deepest retrieval
bug found so far. `PermitRootLogin` lives in OpenSSH's §Protection, and:

- embedding selection ranked §Protection **11 of 41** ("how do I stop root logging
  in": **36 of 41**) — outside top-3;
- raising to top-5 and top-8 did **not** fix it (F30);
- lexical scoring ranked it **1 of 41** — but the answer was *still* missing from the
  context, because §Protection is a single **12,939-char** h2 chunk and
  `PermitRootLogin` sits at offset **4,704**, past the 600-char-per-section slice.

Two fixes, both model-free, both in the slicing rather than the scoring:

1. **Split on h2–h5, not h2–h3.** OpenSSH becomes 77 sections of ≤3.4 KB instead of
   41 of ≤12.9 KB. The target section becomes §Restrict (1,047 chars).
2. **Centre the window on the query terms**, not on the section start.

Scored on all 14 cases (6 single-turn, 6 follow-up, 2 large-article), identical
splitting and windowing:

| Selector | Score | Avg context |
|---|---|---|
| **embedding, top-3** | **14/14** | **1,488 ch** |
| lexical, top-3 | 12/14 | 1,639 ch |
| lexical, top-5 | 14/14 | 2,683 ch |

**A correction I nearly recorded as fact.** An interactive probe scored lexical 14/14
and embedding 11/14, and I was one step from deleting bge-small from the design. The
probe scored the *untruncated* section text; the harness scores the 600-char slice the
model actually receives. Measuring anything other than the bytes that reach the model
is measuring nothing. At equal context the embedder is 2 cases better, and it reaches
14/14 in **44 % less context** than lexical needs — and since prefill dominates
latency here, those tokens cost more than the 35 MB embedder does.

**bge-small stays at section level. At *article* level it still buys nothing**:
Xapian + lexical RRF = **9/10**, identical to the 3-way embedding fusion of F17, so
the article stage drops its two embedding calls.

`pickSectionsLex` is kept as the no-server fallback (14/14 at top-5).

## F32 — ground against the source document, not the slice sent to the model

F27's reference was the ~1.5 KB windowed sections. That rejected a **correct** answer:
"unlock it at boot" cited `cryptsetup`, which the Dm-crypt article contains but the
slice did not. The false-reject rate was sampling-dependent — the same code produced
**0, 1 and 3** across three runs — which is the signature of every grounding defect
found so far.

Both arms, scored on both duties (18 follow-up samples, 6 mismatched contexts):

| Grounding reference | False rejects on correct answers | Fabrications caught |
|---|---|---|
| windowed slice | 1 | 6/6 |
| **full source article** | **0** | **6/6** |

Widening the reference strictly dominates: it loses no safety, because a fabricated
`ssh-keygen` or `mkfs.ext4` is absent from the whole article too. Self-test is now
**19/19**, including the pair that pins the distinction — the same answer is grounded
against the document and ungrounded against the slice alone.

## F33 — [WARN] a benchmark that contradicts itself is reporting a bug, not a result

2B scored **0/6 with 3 false rejects** on follow-ups. That is arithmetically
impossible: a false reject only counts when the answer was correct. An earlier edit had
clobbered the two `sc.hist += …` lines, so the counters never incremented. A raw probe
answered the same question correctly with `/etc/fstab` in the text.

Had the contradiction not been visible in the same output line, "2B collapses on
follow-ups" would have been recorded as a model finding. `benchFollowup` now throws
when `falseReject > correct`.

2B on the improved fixture, measured before the bug was found (these two are valid,
they use separate counters): **answering 6/6 at 13.2 s/answer** (0.8B: 6/6 at
**10.9 s**), **refusal 5/6 → 6/6 with F27** (0.8B: 4/6 → 6/6). Its follow-up score is
unmeasured; the decision does not rest on it.

## F34 — cross-book retrieval: routing is worthless even when perfect

Three ZIMs mounted (bash 132 articles, rust 2,983, arch 14,497), 15 queries with an
unambiguous home book — five per book, each target confirmed rank-1 *within* its own
book before the fixture was written.

| Arm | Right article rank-1 | Right book rank-1 | Requests | Latency |
|---|---|---|---|---|
| oracle — told the correct book | **12/15** | — | 1 | 102 ms |
| **all books, one query** | **12/15** | **15/15** | **1** | **149 ms** |
| all books + lexical RRF | 12/15 | 15/15 | 1 | 149 ms |
| per-book search, RRF across books | 11/15 | 14/15 | 3 | 174 ms |

**The oracle arm is the upper bound, and the all-books query ties it.** Being told the
correct book in advance buys *nothing*, so no routing stage can help — there is no
headroom for it to recover. Content picks the corpus: 15/15 right book, unrouted.

Fusing per-book result lists is **worse** (11/15) and costs 3 requests: each book's
list is scored independently, so a weak book's rank-1 gets promoted to compete with a
strong book's rank-1. Kiwix's own cross-book scoring already normalises this. Rejected.

Searching three ZIMs instead of one costs **+47 ms**. The 4 remaining misses are
within-book ranking failures, identical across every arm — the same class as F17's
"mount a usb drive automatically" → `Netboot`.

**Decision: one `/search` with no `books.name`, no routing stage, no per-book fusion.**

## F35 — kiwix ANDs every query term, so one stray word returns nothing

`string versus str slice` returned **0 hits**. Not a rare-word problem: `versus` has
hits on its own. Kiwix requires *one document containing every term*, so each extra
word multiplies the chance of an empty result — and an empty result is a dead tool.

Comparison phrasings are the common trigger, and they are exactly what people type in
a terminal. Three of six comparison queries returned zero hits.

| Fix | Relevant top-1 | Requests |
|---|---|---|
| F15 stopword list only | 0/6 | 6 |
| **+ comparison words stripped** | **3/6** | **6** |
| + bounded drop-a-term retry on top | 3/6 | 6 |

Adding `versus|vs|difference|between|tradeoffs|should|or|choose|compare|alternatives`
to `prep` eliminated **all three** zero-hit cases at **no extra request**:
`string versus str slice` → `str` first; `pacman versus yay aur helper` →
`AUR helpers`; `swapfile versus swap partition tradeoffs` → `Swap`.

Two things measured and **rejected**:

- **Xapian `OR` syntax** — unsupported. `/search?pattern=String OR str` treats `OR` as
  a literal term and returns 0 hits.
- **A retry loop** that drops trailing terms until hits appear. It never fires once the
  stopword list is wider, so it is code that cannot earn its keep.

The wider list also lifted F34's book selection from 14/15 to **15/15**, and left F17
at 9/10. The 3 remaining comparison misses return hits but rank a neighbouring article
first — the corpus has no single "X vs Y" article, which is a synthesis question this
design does not attempt.

## F36 — two-article synthesis is unreliable, and one-sided context fabricates

Comparison questions ("ext4 or btrfs") have no single article to answer them. Five
pairs, both articles' facts verified present in their own retrieved sections first:

| Arm | Result |
|---|---|
| both articles in context — mentions both sides' facts | **2/5, then 3/5 on re-run** |
| only side A in context — invented facts about side B | **2/5** |
| only side A — declined or was caught | 0/5 → **2/5 after F38** |

Verbatim, with the reference containing *only* systemd-timesyncd and iptables:

- *"chrony is the recommended alternative to systemd-timesyncd…"*
- *"iptables is the default table for most common use cases…, while nftables is
  recommended for complex configurations."*

Confident, plausible, and about a tool the model was never shown. **F27 caught
neither** — there is no command in either sentence, so the command rule cannot see it.

**Decision: do not build synthesis. Ask the user instead (F37).** At 2–3/5 it is not a
feature, and the one-sided case is a fabrication generator.

## F37 — the ask-the-user trigger, model-free

Split the question at its comparison word, carry the shared tail into **both** sides,
retrieve each side, and ask only if the two sides resolve to *different* articles.

| Metric | Result |
|---|---|
| fires on comparison questions | **6/6** |
| silent on normal questions | **26/26** |
| both sides retrieved correctly | 5/7 (2 others returned topical articles) |
| cost | 2 requests, no model |

The shared tail matters: `["bash", "zsh startup files"]` retrieved a bash-docs *index*
page for the bare left side, while the right side was correct — the tail was doing the
work. Splitting to `"bash startup files"` / `"zsh startup files"` fixed it.

Ordering constraint found here: **the split must run before `prep`**, because `prep`
strips the very comparison words the split needs (F35).

A first version matched a comparison word and then required two retrieved titles to
contain query terms. It fired on only 4/6, because retrieval on an *unsplit* comparison
query surfaces neither side. Splitting first is simpler and strictly better.

## F38 — a fabrication class with no command in it

F27 only inspects commands, echoes and questions. F36's fabrications were prose claims
about an entity that was never retrieved. Both sides' names are already known from the
question's grammar (F37), so the check stays model-free: **a side named in the answer
must appear in what was shown to the model.**

This exposed a direct tension with F32. Grounding commands against the *full article*
removed false rejects — but `chrony` appears in systemd-timesyncd's "See also", so the
wide reference licensed a claim about it. Resolution: **two references, one check.**

| Check | Reference | Why |
|---|---|---|
| command not in reference | full source article (F32) | a neighbouring section legitimately names `cryptsetup` |
| asserts about the other side | the slice actually shown | "See also: chrony" must not license a recommendation |

Self-test is now **23/23**, including the case where the same answer is grounded
against the document and ungrounded against the slice.

**Honest limit:** when the other side genuinely appears in the shown slice — the
Iptables article does mention nftables — a lexical check cannot verify the *claim*, only
the entity's presence. That case stays uncaught, which is why F37's ask path, not this
check, is the actual safety mechanism for comparisons.

## F39 — [WARN] article ranking cannot be improved by section evidence

F34's misses were identical across every arm, so the hypothesis was that title+snippet
is too thin and *section* evidence — the signal that fixed extraction (F31) — should be
promoted into article ranking. Top-5 candidates refetched and rescored, 25 queries:

| Ranker | Rank-1 correct |
|---|---|
| **base: RRF(xapian, lexical title+snippet)** | **21/25** |
| best single section | 16/25 |
| whole-article term density | 20/25 |
| best section − mean section (peakiness) | 15/25 |
| best section ÷ √sections | 13/25 |
| RRF(base, best-section) | 21/25 |

All four formulations are worse, and fusion only ties, for **+92 ms and 5 article
fetches per query**. Index pages win on max-section score: "Bash Documentation" beats
"Redirections" because many of its sections mention the query's terms.

**Rejected.** Title+snippet plus Xapian order is the ceiling at 84 %.

But the misses are not lost — **3 of 4 sit at rank 2 or 3**:

| Recall | @1 | @2 | @3 | @5 |
|---|---|---|---|---|
| of 25 | 21 | 22 | **24** | 24 |

So the fix is to widen, not to rerank — the same lesson as top-3 sections beating
argmax. Spreading the same budget over the top-3 articles costs almost nothing:
**3 articles × 1 section = 1,842 ch** versus **1 article × 3 sections = 1,603 ch**, and
the wide context contains the `udisksctl` answer the narrow one missed.

## F40 — the catalog is the index; suggest a download when the corpus can't answer

`library.kiwix.org` lists **1,286 English ZIMs, 2,773 GB total**, so no index needs
building — the OPDS catalog *is* the index. Cached as JSON it is **405 KB**, or
**192 KB** with only the fields the suggester uses.

Matching the question against catalog metadata, docs-category ZIMs only:

| Metric | Result |
|---|---|
| suggestions offered | 8/10 questions |
| **of those, correct ZIM in top-3** | **8/8** |
| wrong suggestions | **0** |
| silent when the local corpus answers | **5/5** |

Three matcher lessons, each measured:

1. **Metadata describes the corpus, not its contents.** "what is the capital of
   Mongolia" cannot match Wikipedia's blurb ("The free encyclopedia"), and "how do I
   unclog a drain" matches nothing at all. Lexical matching works only for questions
   naming a *technology*, which is exactly what a ZIM title carries.
2. **Restrict matching to docs-category ZIMs.** Unrestricted, "capital" matched
   `ted_mul_capitalism` — the "du in produce" substring bug again. Scoped to devdocs,
   precision went to 8/8 and the general-knowledge questions correctly fall through.
3. **Hybrid term rule.** Terms ≥4 chars match as substrings so `postgres` reaches
   `postgresql`; shorter terms need a word boundary so `git` reaches `devdocs_en_git`
   without matching "digit". Substring-only scored 8/10, word-boundary-only 7/10
   (it missed postgres→postgresql), hybrid 8/8 of matched.

For the two that fall through, suggest a fixed general tier rather than pretending to
match. **`wikipedia_en_top` is 0.3 GB for 875,265 articles** — versus 124 GB for
`wikipedia_en_all`. That is the sane default; the monolith should never be suggested.

The trigger is the existing failure signal: zero local hits, or a grounding rejection.
No new detection logic.

## F41 — [WARN] local file reading is not measurable yet, and not shippable

`tny "summarize src/main.rs"` sits in the CLI surface with nothing behind it. A 59 KB
source file is ~16k tokens against an 8192 context, so excerpt selection is mandatory.
Code has no `<h2>` headings, so F31's section split does not apply.

**Selection is solid and deterministic** — four ways to fit a file into 1,800 chars,
scored model-free on whether the answer is present at all:

| Excerpt strategy | Answer present |
|---|---|
| first 1,800 chars | 3/6 |
| one term-centred window | 5/6 |
| 3 term-scored fixed chunks | **6/6** |
| 3 term-scored structural chunks | **6/6** |

**Answering is not.** Three runs, and the numbers do not support any conclusion:

| Run | Arm | Excerpt | Score | Latency |
|---|---|---|---|---|
| 1 | 3 fixed chunks | 1.8 KB | 3/6 | 19.9 s |
| 2 | 3 structural chunks | 3.6 KB | 5/6 | 38.5 s |
| 3 | 3 fixed chunks | 1.8 KB | **5/6** | 19.6 s |

Runs 1 and 3 are the **same arm at the same budget** and differ by two cases. So the
"structural chunking fixed it" reading from run 2 is unsupported — variance dominates a
6-case fixture, exactly as it did in F27 (three runs of "0 false rejects" while broken)
and F33. Nothing here ranks the chunkers.

**A needle was satisfied by garbage.** For "what does the ungrounded function reject",
`/command|restat|question/i` matched:

> "The ungrounded function rejects command, restat, and question questions."

That is an echo of the needle's own alternatives, scored as correct. The fixture's
expectations are as broken as the ones F35 exposed for `/ntp/i`.

**Verdict: cut local file reading from the shipping surface** until it has a fixture
worth trusting — 15+ cases, expectations that a regurgitation cannot satisfy, and
repeated runs. It is the only capability in `PLAN.md` whose promise rests on no
evidence, and the honest options are to prove it or not to claim it.

Costs worth keeping: ~20 s per answer at a 1.8 KB excerpt, ~38 s at 3.6 KB — prefill
scales with the excerpt and dominates, so excerpt size is the latency dial.

## Benchmark hygiene — learned the hard way this session

- **One model server at a time.** Two `llama-server -t 4` processes on a 4-core box put
  load average over 8 and made the machine unusable. Threads must not exceed cores in
  aggregate.
- **Idle servers are free.** Measured over 10 s with no requests: **0 %** of one core for
  both the chat and embedding servers. A resident daemon costs RSS, not CPU.
- **`ps %CPU` is a lifetime average, not instantaneous.** It reported 79 % and 38 % for
  idle servers and nearly went into these notes as "llama-server busy-spins when idle".
  Use `/proc/<pid>/stat` deltas.
- **Spend model calls only where they are the measurement.** Selection is deterministic
  and free: score every arm model-free, then run the model on the winner. That turned
  F41 from 18 calls into 6.

## F42 — piped input works, and it is the best non-ZIM path measured

`tny "what is wrong" < paste.txt`. Structurally easier than F41: a paste averages
**288 chars**, so it is sent whole — no chunking, no boundaries, no selection — and the
paste itself is the grounding reference, so F27 applies unchanged.

Six real tool outputs (rustc E0502, `systemctl`+`journalctl`, ssh host-key warning,
vitest assertion, `df -h`, `docker ps` restart loop):

| Metric | Result |
|---|---|
| top-level cause correct | **5/6** |
| latency | **10.9 s per answer** |
| F27 false rejects | **0/5** |

Faster than every other path because prefill scales with input and a paste is tiny.

**The one miss is a knowledge gap, not a context gap.** For a container `Restarting
(137)` with `Killed` in the log, it blamed the `connection refused` line and missed that
**137 means the kernel OOM-killed it**. Both "137" and "Killed" were in the context, so
this is the case retrieval should fix — the natural next test is whether a docker/Arch
lookup rescues it.

**Caveat, and it is the recurring one:** 2 of the 5 passes contain invented specifics.
The borrow-checker answer claimed `v` was "declared as a mutable reference (`&mut v`) in
the loop" — there is no loop and no `&mut v` — and the `df` answer said "consuming 220GB
out of its 209GB available space", inverting the columns. Both satisfied their needle by
naming the right cause. This is the same shape as the earliest finding in these notes:
**the headline fact is right and the elaboration is fabricated.** The terse contract
(≤160 tokens, two sentences) is already in place and does not stop it.

Untested mitigation: one sentence for pastes instead of two.

## F43 — Q4_K_M rejected: quantisation changes the *shape* of the errors

Halving the weights was meant to be a free RAM win. Same fixtures, Q8_0 versus Q4_K_M:

| Metric | Q8_0 | Q4_K_M |
|---|---|---|
| answering, context present | 6/6 | **6/6** |
| refusal, model alone | 4/6 | 4/6 |
| **refusal + F27 grounding check** | **6/6** | **4/6** |
| corpus lift | 3/6 → 6/6 | 2/6 → 6/6 |
| on disk | ~800 MB | **508 MB** |

Accuracy with context is identical, and disk drops 37 % (not the half I assumed). The
reason to reject it is the **safety net degrading**: F27 recovers Q8_0 from 4/6 to 6/6
but Q4_K_M only from 4/6 to 4/6.

Q4_K_M's fabrications take an *uncatchable form*. Asked about disk space with a
mismatched reference, it answered:

> "To check disk space usage, open your file explanations or command prompt and navigate
> to the C…"

Windows-flavoured prose with **no command token in it**, so the command rule cannot fire
— the same blind spot F38 found for comparative claims. Q8_0's failures were wrong
*commands*, which the check catches every time.

**A deterministic checker is only as good as the failure shape it was designed against,
and quantisation changes that shape.** Any future quant swap must re-run `refuse`, not
just `answers` — accuracy parity hid a real safety regression.

**Nor is it faster.** The harness numbers (17.4 s vs 10.9–20 s) were useless — different
machine load — so both quants were benchmarked back-to-back on an idle box:

| Model | Size | pp512 | tg64 |
|---|---|---|---|
| **Q8_0** | 784 MiB | 40.55 ± 1.29 t/s | **7.82 ± 0.05 t/s** |
| Q4_K_M | 497 MiB | 41.54 ± 0.53 t/s | 7.71 ± 0.11 t/s |

Identical inside the error bars: +2 % prefill, −1 % decode. **37 % fewer weight bytes
buys no speed, so decode on this CPU is compute-bound in dequantisation and the BLAS
path, not memory-bandwidth-bound.** That kills the whole "smaller quant for speed" idea
on this class of machine, not just Q4_K_M.

So the trade is 287 MB of RAM against the grounding check's refusal recovery. Stay Q8_0.

## F44 — detail-level grounding: free, safe, and less powerful than hoped

Targets the most persistent failure in these notes: *headline right, elaboration
invented*. Rule: **every multi-digit number and code-shaped identifier in the answer must
appear in the reference.** Numbers compare as digit strings, because the model reformats
units ("220GB" against a reference saying "220G"). Single digits are exempt — they are
usually enumeration ("three files").

| Measurement | Result |
|---|---|
| recorded fabrications caught | **3/3** — invented version+date, `--keyring` flag, `/dev/mapper/x` |
| false rejects, ZIM answers | **0/11** |
| false rejects, pastes | **0/5** |
| wrong answers caught *live* | **0** |

So it is **free** — it never fires on a correct answer across 16 samples — but its value
is demonstrated only on recorded fabrications. The two live failures in that run happened
to contain no invented numbers or identifiers. Keep it (zero cost, catches a class F27
cannot see), but do not claim it as a live catcher yet.

**What it cannot do.** Two recorded fabrications are invisible to it:

- *"220GB out of its 209GB available space"* — both numbers are in the reference; the
  fabrication is the *relationship* between them.
- *"`&mut v` in the loop"* — no loop exists, but "loop" is prose, not an identifier.
  Catching that needs claim verification, not token matching.

**One defect, same class as always.** The flag pattern `--?[\w-]{2,}` matched
`-contained` inside "self-**contained**" and rejected a correct answer. Fixed with a left
boundary. That is the third time a pattern here has been bitten by ordinary prose, after
`du` matching inside "produce" and the word-overlap ratio of F27.

## F45 — the commandless-prose rule, and two defects that only a weak model exposed

F27 inspects commands, so an answer containing none is invisible to it. That is exactly
how Q4_K_M evaded it (F43). Rule: **if the question asks how to do something and the
answer names no command, it is not an answer.** Refusals are exempt — declining is the
behaviour the check exists to encourage.

Unit cases pass 9/9, including four that must *not* fire: "why did this fail" (diagnosis),
"what does this say about disk usage" (reading output), "what is a mutex" (conceptual),
and "how many neurons…" ("how many" is not "how to").

**Defect 1: it measured formatting, not grounding.** The first version looked for
*marked-up* commands in the answer — backticks, fences, `$`/`#` lines. Qwen-0.8B always
uses backticks, so it looked perfect. LFM2.5-350M writes bare prose, "Use timedatectl
set-timezone", which F27 explicitly allows — and F45 **rejected 3/3 of 350M's correct
answers**. Only re-testing a weaker model exposed it.

Fixed by matching against the reference's own command vocabulary instead of markup.

**Defect 2: `<code>` is not where wikis keep tool names.** Core_utilities yields a
six-word code vocabulary — `rm mv cp arch kill ln` — and names `ncdu`, `gdu`, `dust`,
`dua-cli` only as wiki **links**. So a correct "du alternatives include ncdu, gdu…"
answer was still rejected. Vocabulary now includes single-token `<a>` text (112 entries
for that article instead of 6).

After both fixes, verified in both directions: **0/6 false rejects on 0.8B** and the
catch on 350M is undiminished at **6/6 safe**.

## F46 — the safety net holds as the model degrades; the answers do not

Re-tested the small models now that retrieval and the rules had improved. Retrieval
improvements could not help them — their failures were measured with the needle *already
present* — so this was a test of the **rules**, not the corpus.

| Model | Size | Correct | Refuses alone | +F27 | +F44/F45 | Decode |
|---|---|---|---|---|---|---|
| **Qwen3.5-0.8B Q8_0** | 784 MiB | **6/6** | 4/6 | 6/6 | **6/6** | 7.8 t/s |
| LFM2.5-350M Q8_0 | 359 MiB | 3–4/6 | 1/6 | 1–2/6 | **6/6** | 20.5 t/s |
| LFM2.5-230M Q8_0 | 233 MiB | 1/6 | **0/6** | 4/6 | **6/6** | 29.5 t/s |

**230M refuses nothing on its own and still ends up 6/6 safe.** Model-side judgement
collapses as size falls; the deterministic rules do not. The rules, not the model, are
what make this safe to point at a shell.

Note the crossover: F27 alone rescues 230M better (4/6) than 350M (1–2/6), because 230M
fabricates *commands* — catchable — while 350M emits content-free prose that needs F45.

**Output quality, verbatim, same question and context:**

| Question | 0.8B | 350M |
|---|---|---|
| mount a usb automatically | "Use `udiskie` … via the `udiskie-dmenu` interface." | "Use `bashmount` to mount a removable USB drive." ✗ |
| encrypt a partition | "use `cryptsetup luksFormat` with the `-s` flag…" | "encrypt a partition" ✗ (echo) |
| generate an ssh key | "…`ssh-keygen` with the `-t ed25519-sk` option…" | "generate an ssh key with the ssh-keygen(1) command." |
| check disk space | "du alternatives include cdu, dua-cli, dust, gdu, ncdu…" | "Check the disk usage with `gdu`." |

350M's signature: question-echo prefixes, man-page artifacts (`ssh-keygen(1)`,
`mkswap(8)`), and plausible-but-wrong tools. **Decision: 0.8B stays.** 2.6× faster decode
does not pay for 6/6 → 3–4/6, but 350M is now a *safe* fallback for hardware that cannot
host 0.8B — it declines rather than fabricates.

**0.8B's own quality is not clean either.** Two of its six correct answers carry wrong
details: `-s` is cryptsetup's *key-size* flag, not a partition size, and `ed25519-sk`
requires FIDO2 hardware. F44 cannot catch either, because both flags appear in the source
article. The needle-based fixture scores these as correct, which is a further argument for
the stricter fixture F41 demands.

## F47 — the answering system prompt was never reviewed; one sentence was worth 2/6

Prompts were tuned once (F15), for *tool routing* — a stage the deterministic pipeline
deleted. The answering prompt had never been A/B tested. Five variants, both live fixtures,
scoring correctness, grounding friction, output tokens, and unaided refusal:

| variant | correct | false rej | tok | refuses alone | safe |
|---|---|---|---|---|---|
| current | 6/6 | 0/6 | 31 | 4/6 | 6/6 |
| **strict** | **6/6** | **0/6** | **29** | **6/6** | 6/6 |
| cmdfirst | 5/6 | 1/5 | 41 | 6/6 | 6/6 |
| bare | 6/6 | 1/6 | 35 | 6/6 | 6/6 |
| strictcmd | 5/6 | 0/5 | 29 | 6/6 | 6/6 |

**Adopted `strict`** — the added sentence is *"Use only facts written in the reference.
Never add a flag, option, version, or path that does not appear there."* It takes unaided
refusal from **4/6 to 6/6**, doing what F27's regex does, while cutting output by 2 tokens.
Confirmed on a second independent run: answers 6/6, **0/6 false rejects**, refuse 6/6.

Two variants that *lost* an answer are as informative: both command-only phrasings
("reply with that command and nothing else") scored 5/6, because some correct answers are
prose. And `bare` produced a false reject, so the verbose contract earns its keep.

## Speculative decoding — rejected twice, measured, not argued

Pairing 0.8B as a draft model for 2B looks attractive because reranking-style workloads are
prefill-bound, which is this CPU's fast direction (40.55 t/s prefill vs 7.82 t/s decode).
It fails on the ratio: 0.8B decodes at 7.82 t/s against 2B's 5.08 t/s, a **1.54× draft
advantage** where speculation wants 10×.

| arm | per answer |
|---|---|
| 2B alone | 38.8 s |
| 2B + 0.8B draft (`--spec-type draft-simple`, n-max 6) | **69.1 s (78 % slower)** |
| 0.8B alone | 15.0 s |
| 0.8B + n-gram (`--spec-type ngram-simple`) | 15.4 s (no change) |

n-gram speculation was the better bet — it drafts from the prompt, costs no RAM, and our
answers are extractive by construction — and it still bought nothing. Both rejected.

## Gemma 4 E2B — rejected on arithmetic, without downloading it

`google/gemma-4-E2B-it-qat-q4_0-gguf` is **3.35 GB** (E2B is *effective* 2B compute via
per-layer embeddings; the file still carries all 4.63 B raw params). Against ~4 GB available
on a 7.7 GB box with 2.3 GB of swap already in use, it cannot run *consistently*. And PLE
reduces effective compute, not the dequantisation work F43 proved is this CPU's bottleneck,
so decode ≤ 2B's 5.08 t/s → ≥39 s per answer before any paging. Not tested, and the
decision is recorded rather than the download.

## Qwen3-Reranker-0.6B — rejected: 25× the cost of the fix that already works

Reranking generates exactly one token, so it is pure prefill — the objection is not "another
model". Scoring 8 article candidates costs ~920 tokens ≈ **18 s** at ~50 t/s. F39b already
measured the alternative: *send the top 3 articles* scored 5/6 vs 4/6 for +140 characters
≈ 0.7 s. Section-level reranking is worse still: top-8 sections ≈ 6,400 tokens ≈ 2 minutes.
F39 also tried four hand-built rerank formulations and **every one lost** to plain
title+snippet (base 21/25, best variant 16/25). The one thing a cross-encoder uniquely
offers is a *calibrated absolute score*, and both uses for that — corpus-miss detection and
early refusal — are already solved model-free (F40 at 8/8, F27 for free after generation).

## F48 — corpus growth inverted F34: one global query is swamped

Mounting `unix.stackexchange.com_en_all` (413,259 articles) beside the Arch Wiki (14,497)
broke retrieval, and the recorded decision "query all books, unrouted" (F34) went with it:

| 15-case cross-book fixture | 3 books | 9 books |
|---|---|---|
| right **book** rank-1, one global query | 15/15 | **8/15** |
| right **article** rank-1, one global query | 12/15 | **7/15** |

Three real how-to queries through the shipped CLI all routed to Stack Exchange discussions
instead of wiki instructions, and one produced a **factually wrong answer** — "a symmetric
key pair" for SSH keys, from a thread about AES-256-CBC. **The grounding rules cannot catch
this class**: the answer is faithfully grounded in the page it was given. The page is wrong.
That makes retrieval the only unguarded stage in the pipeline.

**Per-book RRF is not the fix, and knowing why matters.** RRF fuses *rankings of the same
items*. Per-book result lists are disjoint, so every book's rank-1 ties at `1/(k+1)` and
insertion order silently decides the winner: **3/15**, worse than doing nothing. F34's
"per-book fused 11/15" was the same accident, disguised by a book list that happened to
contain only relevant books.

Two structural signals were being discarded, both free:

* kiwix **bolds the terms it matched** in each snippet — its own match evidence, and the
  only cross-book comparable signal in an API whose search XML carries **no score at all**.
* the **path** identifies page kind: `questions/<id>/<slug>` is an answer page, while
  `questions/tagged/…` and titles like "Highest Voted 'pacman' Questions" are navigation.
  That index page was ranking #1 for a how-to query — a page that lists questions and
  answers none. Filtering page kind is pure gain.

## F49 — it was candidate *generation*, not scoring

Built a fixture worth trusting first, since F41 established that 6 cases cannot rank two
options: **32 verified cases** across four intents, each pinned to an expected book, article
title, and a needle proven present in that article (`bench/fixture-instructional.mjs` 18,
`bench/fixture-qa.mjs` 14). Two agents built them in parallel; 26 of 58 candidate cases were
dropped for failing verification rather than weakening a regex.

The measurement that reframed everything:

| | value |
|---|---|
| candidate recall, one global query | **10/32** |
| candidate recall, per-book union (5 each) | **31/32** |

Every case is findable at rank ≤8 *within its book*; the global query simply never surfaces
it. **No scorer can rank a candidate that was never retrieved** — which is why six scoring
formulations all sat between 2/18 and 7/18. Ask each book separately and merge: 9 requests,
~57 ms each, against a 15-22 s query.

| formula (32 cases) | article@1 | book@1 | article@3 |
|---|---|---|---|
| `lex` — what shipped | 3/32 | 5/32 | 7/32 |
| xapian raw | 4/32 | 7/32 | 7/32 |
| union + bm25t | 7/32 | 8/32 | 12/32 |
| union + kind prior | 11/32 | 12/32 | 15/32 |
| **union + kind + title coverage** | **19/32** | **27/32** | **25/32** |
| union + kind + rare-term weight | 18/32 | 26/32 | 24/32 |

Three signals earned their place, all model-free:

1. **Length-normalised title scoring.** `lexScore` weighted every title hit +3 with no
   normalisation, so a 74-character Stack Exchange question title beat the Arch Wiki's
   "Swap" on a how-to query by matching more terms through sheer length.
2. **Title coverage as an entity match.** Reference questions name their own article — "what
   does the `--rm` option do in *docker run*". Term coverage cannot see that; "every term of
   a short title appears in the query" can, and it took `reference` from 3/6 to 4/6.
3. **Intent × page-kind prior.** A how-to question wants instructions, and a Q&A title that
   is itself a question is not evidence that it answers *this* one. Intent is inferred from
   the query by regex (12/18 correct), never labelled.

**Where it is still weak, honestly:** `diagnose` sits at 4-5/10 and no formulation improved
it. The QA fixture's own construction explains why — 6 of its 15 dropped cases were dropped
because *the wiki answered them better* (sudo password timeout, `.bashrc` vs
`.bash_profile`, `sh` vs `bash`, permission denied on an owned directory). So "diagnosis
prefers Q&A" is not true as a prior, and the remaining 13/32 rank-1 misses are unfixed.

## F50 — the Rust CLI exists and answers end-to-end

`src/main.rs`, `src/retrieve.rs`, `src/ground.rs`, `src/corpus.rs` — three deps (`ureq`,
`serde_json`, `regex`), no async runtime, no clap, no HTML parser. 13 unit tests carry the
31-case grounding self-test, and they are pure: no servers, no network.

First full-query measurement, the number PLAN.md had never had:

| stage | time |
|---|---|
| search | 467 ms |
| fetch article | 15 ms |
| generate | 21,226 ms |
| **total** | **22.6 s** |

Retrieval is **2 %** of a query. That kills any objection to spending more requests on
better candidates (F49), and it means every latency lever is prefill: at ~40 t/s, the
top-5×600-char lexical context costs ~18 s of the 22.6 s. The embedding selector's 39 %
smaller prompt (F31) is therefore worth ~8 s per query — now measured end-to-end rather
than derived, and worth revisiting.

Deliberately not built: `--online` fallback, tool calling (F6, unused by a deterministic
pipeline), local file reading (F41, cut for lack of a trustworthy fixture), and an embedder
(one fewer supervised process; `pickSectionsLex` is 14/14 at top-5).

## F51 — the catalog's own length is wrong, and my check trusted it

`tny --corpus add wikipedia_en_top_nopic` refused a *complete* 2.2 GB download and retried
forever. The catalog advertises `length="2239865856"`; the file is really **2,239,864,871**
bytes. 2239865856 / 1024 = 2187369 exactly — the catalog rounds every size **up to a KiB
boundary**. The byte-verification from F21 compared against that figure, so any download
whose true size is not KiB-aligned could never be declared complete.

Fix, and the general rule: **the transferring party is the authority on the transfer.** A
`HEAD` against the mirrors gives the exact `Content-Length`; the catalog figure is now used
only for display and as a fallback. A `416 Range Not Satisfiable` reply carries
`Content-Range: bytes */<total>`, which is the server stating the size outright — treated as
proof of completion when local bytes match, not as an error.

## F52 — one hard-coded host is a design flaw; measured, not theorised

Mid-session **every `kiwix.org` host went down** — `library.kiwix.org` (the catalog),
`download.kiwix.org`, and `mirror.download.kiwix.org` all timed out. Three independent
mirrors stayed up and all reported the identical length:

| host | status | length |
|------|--------|--------|
| library.kiwix.org | timeout | — |
| lb.download.kiwix.org | timeout | — |
| mirrors.dotsrc.org | 200 | 2,239,864,871 |
| ftp.fau.de | 200 | 2,239,864,871 |
| saimei.ftp.acc.umu.se | 200 | 2,239,864,871 |

`corpus::add` now tries the catalog URL then those mirrors in order, per pass, and a pass
that advances zero bytes aborts instead of hammering. Proven live: the second corpus failed
over from the dead `download.kiwix.org` to `mirrors.dotsrc.org` and completed. The cached
catalog (1.5 MB) meant the library outage could not block downloads at all.

## F53 — full article text loses to kiwix's snippet, twice measured

Union recall reached 30/32 while the best scorer got 17/32 right at rank 1, so 13 cases had
the right article in the list and ranked it below something else. The obvious next lever:
stop scoring a ~200-character snippet and score the article. Fetches cost 15–41 ms and
retrieval is 2 % of a query, so 6 fetches per query is affordable.

**Result: 5/32 — a third of the snippet scorer.** Every intent got worse (howto 7→2,
reference 3→0, diagnose 4→1). Same direction as F39's section-evidence rerank (21/25 →
16/25), and now the reason is clear: **kiwix's snippet is the query-matched passage**.
Xapian has already localised the evidence. Full text replaces a focused signal with a
diluted one, and no length normalisation recovers it — the information was in *which*
passage matched, which the body no longer tells you.

Closed: content-based reranking. The remaining ranking headroom is in title/entity signals
and page kind, not in reading more text.

## F54 — the entity bonus was promoting the most generic article

`title_covered` gave a flat +3 when every word of a title appeared in the query. A one-word
title satisfies that trivially, so **`Docker`, `PostgreSQL`, `Ocean`, `Plant`, `Memory`,
`Chemical compound`** collected the full bonus for covering a quarter of the question and
beat the article that answers it. Scaling the bonus by the share of the query the title
accounts for (`title terms / query terms`) recovered **+1 case of 58** — real, but it proves
the bonus was not the whole story: those generic pages also win on snippet terms.

## F55 — measuring the binary, not a JS re-implementation

`bench/rank-cli.mjs` runs `tny --rank`, which stops after ranking and prints the shortlist.
Retrieval is 2 % of a query's wall time, so measuring through generation cost 80 s per case
and hid the thing under test; `--rank` measures all 58 cases in 110 s. Four fixtures,
58 verified cases (18 instructional, 14 Q&A, 16 general knowledge, 10 ambiguous-term):

| metric | result |
|--------|--------|
| right book @1 | **48/58** |
| right article in top-8 shortlist | **47/58** |
| right article @1 | **33/58** |
| **answer present in the fetched article** | **36/58** |

The last row is the only one a user experiences: the answering stage reads the article, not
its title, and three cases answer correctly from an article the fixture calls wrong
(`Burmese python` for reticulated-python prey, a Q&A thread for `useradd`). Conversely
`Penicillin` at rank 1 does *not* satisfy the Fleming needle.

**The 25 misses split three ways, and they need opposite fixes:**

- **14 scoring** — right article at rank 2–5, retrieved and then out-ranked. 8 of them at
  rank 2. This is where the headroom is: 47/58 recall against 33/58 at rank 1.
- **9 absent** from the shortlist — candidate generation, at 5 per book. Not a scorer problem.
- **2 by design** — "difference between symbolic and hard links" and "SIGTERM vs SIGKILL"
  route to F37's clarify prompt, which retrieves nothing. F37 measured 0/26 false fires, but
  that fixture had no comparison question a *single* article answers. Both of these are
  answerable from one page, so the trigger is now over-firing on a class it never saw.

Book routing (48/58) is not the wall, and the model is not the wall. **Ranking is**, and
the two named signals — generic-parent titles and near-duplicate Q&A titles — are lexical,
model-free, and measurable in 110 s per iteration.

## F56 — an offline scorer sweep, and the weights were wrong

Every live ranking measurement costs ~100 s (58 queries × 16 books of kiwix search), which is
too slow to search a weight space — and slow enough that one bad idea (IDF weighting, 23/58
against 32/58) cost 15 minutes to disprove. `bench/sweep.mjs` dumps the candidates once via
`tny --dump` and caches every candidate article's text, then scores a variant in **~1 s**.

It reproduces the shipping scorer exactly (32/58 article@1, 43/58 answer@3), so it is a
faithful stand-in, not an approximation. Two metrics: `article@1`, and `answer@3` — is the
verified answer text inside one of the three articles tny actually sends the model. The second
is the one that predicts quality; the first is a proxy for it.

A grid over title weight × entity weight × rank divisor put **every one of the top 14
configurations at title × 2**, not the shipping × 3, and at rank ÷ 5, not ÷ 100:

| | article@1 | answer@1 | answer@3 | answer@5 |
|---|---|---|---|---|
| shipping | 32/58 | 35 | 43 | 49 |
| title×2, rank÷5 | **36/58** | **37** | **47** | 49 |

The win is a **plateau**, not a spike — title 2 × cover 3–4 × rank 4–5 all score alike — it
gains in all four fixtures rather than one, and the configuration chosen on a 13-book corpus
still won after three more books were mounted. That is the only cross-validation available at
58 cases, and it is why this is shippable where a single-point maximum would not be.

## F57 — one truncated ZIM took down every query

A dead mirror left a 376 KB fragment of a 237 MB book under its final name. `kiwix-serve`
refuses to mount **its whole library** when one file is unreadable, so every query failed with
`did not come up within 120s` while 15 good books sat on disk. Resume needs those bytes to
survive, so they now live at `.zim.part`, which the mount glob cannot see, and are renamed
only on a verified byte count. A short file already under the final name is *moved back* to
`.part`, not deleted — migration and de-poisoning are the same operation.

## F58 — the navigation filter was deleting the answer

`NAV_PATH` was unanchored: `/tags?/` matched devdocs'
`engine/reference/commandline/tag/index`, so **`docker tag` was classified as a Stack Exchange
tag-index page and dropped from every candidate list** — the exact article the question named.
Verified against the mounted ZIMs, these pages only ever appear at the path root
(`questions/tagged/bash`), so the patterns are now anchored. Worth +1 on every metric, and a
unit test pins both directions because this was invisible for the project's whole life.

## F59 — the remaining ranking failures are synonym gaps, and lexical cannot reach them

Once the weights were right, the failure list stopped being a list of bugs and became one
coherent class. The right article's title shares **zero terms** with the question:

| question | right article | wrong winner |
|---|---|---|
| chemical formula of table salt | `Sodium chloride` | `Why do all particles have the same influence on osmosis?` |
| why is the sky blue | `Rayleigh scattering` | `Sky Blue Sky` (Wilco album) |
| perform CPR on an adult | `Cardiopulmonary resuscitation` | `Is hands-only CPR as effective as traditional CPR?` |
| propagate a plant from cuttings | `Vegetative reproduction` | `Why some plants can be propagated from a leaf cutting…` |
| kernel in corn | `Maize` | `Are corn kernels considered a grain` |
| shell in biology | `Mollusca` | `Bioperl` |
| cookie made of | `Biscuit` | `Dillo` |

The decomposition is unambiguous. `Rayleigh scattering` scores **0.80** with **zero title
hits**; `Sky Blue Sky` scores 7.81 because all three of its title words appear in the query.
Body hits cannot break the tie: kiwix only *returns* pages containing the terms, so "the
snippet contains the query terms" is nearly constant across candidates and carries almost no
information. A grid over the snippet weight confirmed it — answer@3 never moved off 46.

Three model-free repairs were measured and **all failed**:

- **snippet weight** (1–8 against title 0–3): answer@3 capped at 46.
- **`opensearch:totalResults` per book** as a specificity prior: the 875k-article general
  Wikipedia reports `total=5` for "sky blue" while the physics book reports 300. Not usable.
- **`<b>` match markers** as term frequency: they *reward* the failure — `Sky Blue Sky` has 12
  bold markers to `Rayleigh scattering`'s 2.

This is the textbook limit of lexical retrieval, not a defect to fix. The four surviving cases
of this class sit at ranks 14–19, while the other four sit at ranks 3–4 and are reachable by
widening the article count. **The semantic step is the model's job** — it reads the articles —
or an embedder's, which F39 measured as no better than lexical for sections and which would
add a second supervised process.

## F60 — deeper per-book retrieval is free

`PER_BOOK` 5 → 8 lifts the recall ceiling 54/58 → 55/58 and costs **nothing**: it is one
request per book either way, only a longer response. `Hippocampus` answers "how does the brain
consolidate long term memory" from its book's 6th hit. 12 adds nothing further, and neither 8
nor 12 moves article@1 or answer@3 — the extra candidates are inert, not noise, which is the
evidence that the scorer is stable rather than luckily tuned.

## F61 — the proxy is not the product

Every benchmark up to here measured whether retrieval *put* the answer in the context.
`bench/answer-cli.mjs` measures whether `tny "question"` **prints a correct answer**, over all
58 cases, and the grader is derived mechanically from the fixture's own verified needle rather
than from hand-written expected answers: the largest number in the needle when it has one,
its rarest words otherwise. Validated 11/11 against my own reading of real output, and it
catches exactly the failure that matters — answers that read as grounded but are not:

```
article says 115 known moons        ->  "Jupiter has 95 known moons."
article says fell on 9 November 1989 -> "The Berlin Wall fell on November 9, 2009."
```

Both are fluent, confident, and wrong, and no retrieval metric can see either. Three outcomes
are reported, because the difference between the last two is the whole purpose of the
grounding rules: **correct**, **refused** (safe), **wrong** (not).

A methodological failure worth recording: the first run of this benchmark was invalid because
I rebuilt the binary while it was in flight, so early cases used a different configuration
from later ones. An end-to-end run is a measurement of one build, and nothing may touch that
build while it runs.

## F62 — co-occurrence is not comparison evidence, so the clarify prompt stays

Two questions in the QA fixture route to F37's clarify prompt instead of an answer:
"difference between symbolic and hard links" and "SIGTERM vs SIGKILL". The notes flagged this
as a suspected over-fire, since both are answerable from a single page, so I tested the obvious
repair: if one article's text mentions **both** sides, answer from it and skip the clarify.

It fails, and the failure is instructive. Of 24 candidates checked for the link question, the
two that mention both are `pg_combinebackup` and `pg_upgrade` — PostgreSQL tooling. For the
signal question, **all 24 of 24** mention both, led by `git fast-import`, `Atrium`, and
`Keyboard shortcuts (Русский)`. Every one is incidental co-occurrence, not an explanation.

Same shape as the snippet-weight result in F59: presence of a term is not evidence about the
term. Answering from `pg_combinebackup` would be a confident wrong answer; asking which side
the user means is safe. **The clarify prompt is correct behaviour and stays**, and these two
cases are scored as refusals rather than errors.

## F63 — a wrong section is indistinguishable from a hallucination

"when did the berlin wall fall" answered **"November 9, 2009"**. The obvious reading is
fabrication, and it was not. `tny -v` shows the retrieval was *correct* — the right article,
`Fall of the Berlin Wall` — and names the sections it sent:

```
§References, §20th anniversary celebrations, §References, §Fall, §1 Answers 1
```

`§20th anniversary celebrations` describes an event **held in 2009**. The model answered
faithfully from the section it was shown. Two model-free defects produced it:

1. **Apparatus sections were eligible.** `References` is a wall of article titles, so it
   matches query terms by sheer density and outscored the prose. It won two of five slots.
2. **No dedupe by head.** Wikipedia repeats `References` under several parents, so one head
   can take several slots and spend the budget twice on identical junk.

Filtering apparatus heads (`references`, `see also`, `external links`, `bibliography`, …) and
deduping by head gives `§20th anniversary celebrations, §Official demolition, §Fall, §Start of
the construction (1961)` and the answer **"November 9, 1989"**.

The lesson generalises past this bug: **no retrieval metric can see this failure.** article@1
was correct, answer@3 was correct, the answer was wrong. Generation got slower (9.8 s → 28.8 s)
precisely because the sections now hold prose instead of citation lists — the speed was a
symptom of sending junk.

## F64 — the grader had to be measured too, and it was broken

Two bugs in `bench/answer-cli.mjs`, both of which produced *plausible* numbers:

- **Tuple misalignment.** `bench/answer-cli.mjs` prepends the fixture name, making each case 7
  long, but the loop destructured 6. So `expectRe` silently received `needleRe` and every score
  was computed by the article-prose regex the fact grader was written to replace. The reported
  32/58 measured nothing that was intended.
- **`only` matched `argv[0]`**, filtering every case out, and reported `0/0` rather than failing.

With both fixed, the ambiguous fixture — the hardest one, ten word-sense pairs — scores **10/10**
where the broken grader claimed 5/10. Answers that were always correct were being counted wrong:

```
"A corn kernel is a small one-seeded dry indehiscent fruit"        graded WRONG, is right
"A kernel is the core component of an operating system"            graded WRONG, is right
"a shell ... uses alphanumeric characters typed on a keyboard"     graded WRONG, is right
```

Generation is also now cached (`bench/.answers.json`, `--regrade`), so a grader change costs
**0.3 s instead of 14 minutes** — the same split that made the retrieval sweep usable in F56.
A benchmark is code, and an unmeasured benchmark is worth as little as unmeasured code:
every number in this session's earlier answer runs was wrong for a reason that had nothing to
do with the system under test.

## Exploration backlog — speed

The cost model, measured (0.8B Q8_0, 4 threads, this CPU):

```
prefill  40.55 t/s  ->  25 ms per input token
decode    7.82 t/s  -> 128 ms per output token
search   149 ms (all ZIMs)   fetch ~90 ms   embed ~1 call/query
```

So a query costs roughly `25ms × context_tokens + 128ms × answer_tokens`. A 1,500-char
context (~400 tok) plus a 40-token answer is **~15 s**, and both terms are worth attacking.

Ordered by expected payoff, each with the measurement that would settle it:

1. **Skip the model entirely when the answer is one command.** The boldest lever: if the
   selected sections contain exactly one command matching the question's verb, print it
   with its citation and generate nothing. **Saves the whole 10–20 s** for a large class
   of queries. F27's extractor already finds commands in text, so the machinery exists.
   Measure: on the answer fixture, how often does the extracted command equal the model's
   answer? If it is high, the model becomes the *fallback*, not the default path.
2. **Cap answer length harder.** At 128 ms/token, every sentence costs ~2 s. Answers ran
   23–51 tokens against a 160 cap. Measure one-sentence prompting against accuracy — it
   also targets the detail-fabrication failure (F42), so it may buy both.
3. **Thread count is probably mistuned.** The very first sweep (1.2B) peaked at **3
   threads** for decode — 11.09 t/s at `-t 3` versus 9.72 at `-t 4` — because the 4th
   thread contends with the OS. Never re-measured for 0.8B. Free if it holds: sweep
   `-t 2,3,4` × `--poll 0,50`, plus `-fa on`.
4. **Speculative decoding.** llama.cpp takes `--model-draft`; LFM2.5-230M Q8_0 (233 MiB)
   is already on disk and shares nothing architecturally, so a Qwen3.5 draft would be
   needed. Typical 1.5–2× on decode. Measure acceptance rate before believing it.
5. **Trim context, not just answers.** 14/14 held at 1,488 chars (F31); nobody has tried
   800. Each 400 chars removed saves ~2.5 s of prefill. Measure presence and answers at
   400/600/800 per section.
6. **Sentence-level windows instead of 600-char blocks.** Smaller context and less
   irrelevant text to fabricate from. Doubles as an accuracy lever.
7. **Prompt-cache the system prefix across queries.** Follow-ups already reuse 350–440
   tokens (F28). A fresh query only shares the ~40-token system prompt. Ordering is
   already correct (stable text first); measure whether a longer stable preamble that
   ends up cached is net cheaper than a short uncached one.
8. **Smaller quants are a dead end here** — F43 proved decode is compute-bound, so
   Q4/Q3 buy RAM, never speed.

## Exploration backlog — accuracy

Current ceilings, measured: article recall@1 **21/25**, recall@3 **24/25**, end-to-end
**5/6**, refusal **6/6** with the check, comparison detection **6/6**.

1. **A fixture worth trusting comes first.** Six cases cannot rank two options — `file`
   scored 3/6 and 5/6 on the *same arm at the same budget*, and two needles were
   satisfied by garbage (`/command|restat|question/` matched "command, restat, and
   question questions"; `/ntp/i` matched a dump of `ntp.org`). Needed: 25–40 cases,
   exact-answer expectations, absent-answer cases, and repeated runs. **Every accuracy
   claim below is unmeasurable until this exists.**
2. **Detail-level grounding — the highest-value idea here.** The recurring failure across
   the whole session is *headline right, elaboration invented*: "`&mut v` in the loop"
   (no loop exists), "220GB out of 209GB available", fabricated Rust release dates,
   invented Yaccarino tenure. A deterministic rule catches this class: **every number and
   identifier in the answer must appear in the reference.** Cheap, model-free, and it
   targets the one failure mode nothing has yet touched. Measure false-reject rate first,
   by sampling — that is how F27's three defects were found.
3. **Commandless-prose rule.** F38 and F43 share a blind spot: an answer with no command
   cannot be checked against commands. If the question is imperative ("how do I X") and
   the reference contains commands but the answer contains none, that is suspicious.
   Would have caught Q4_K_M's "open your file explorer" and closed the F43 regression.
4. **Retry once on rejection instead of surrendering.** Grounding failure currently means
   "not found". Re-asking with the next-best sections costs one call (~15 s) and may
   convert a refusal into an answer. Measure the conversion rate against the false-answer
   rate it introduces.
5. **Widen to top-3 articles at fixed budget.** Recall@3 (24/25) far exceeds recall@1
   (21/25), and end-to-end went 4/6 → 5/6 for +140 chars. Confirm on the bigger fixture;
   it is the cheapest accuracy win identified.
6. **Anchor route for API questions.** The one total retrieval miss ("Box dyn Error trait
   object") had the right article absent from all 8 candidates. F13 proved
   `suggest → path → #anchor` works (240 anchors in `std/vec/struct.vec`) but it is only
   used for pages already known to be reference docs. Measure it as a *router*: for
   API-shaped questions, does the anchor route beat FTS outright?
7. **Retrieval for error codes.** F42's only miss was not knowing exit **137** means
   OOM-killed, with both "137" and "Killed" in context. Test whether a corpus lookup
   keyed on the error token rescues it — the general question being whether pastes should
   also trigger retrieval.
8. **Section-level lexical fusion, reconsidered.** F31 chose embeddings on a 14-case
   fixture where lexical needed top-5 for the same score. On the OpenSSH cases lexical
   ranked the target 1st while embeddings ranked it 11th and 36th. On a larger fixture the
   ordering may flip — and lexical needs no server.
9. **Better comparison-side retrieval.** Split retrieval got both sides 5/7; the misses
   returned topical-but-not-target articles. Worth revisiting once the ask-the-user flow
   is real, since it decides what the user is offered.

## Open questions

- [x] ~~**Is 0.8B the floor?**~~ **Yes, on this fixture.** The Qwen3.5 dense ladder is
      **0.8B → 2B → 4B → 9B → 27B** (MoE 35B-A3B needs all 35B resident — dead on
      7.7 GB RAM). 2B measured at 6/6 vs 0.8B's 6/6 for 2.2× the cost, and its one
      refusal advantage is replaced by F27's regex. **0.8B stays.** Not retested:
      4B/9B — pointless until a fixture exists that 0.8B actually fails.
- [ ] **Build a harder fixture.** The 6-case answering benchmark is saturated at 6/6
      for both models, so it cannot guide further model choice. Needed: multi-fact
      synthesis, exact-flag questions, and cases whose answer is genuinely absent
      from the corpus. Until then any "bigger model is better" claim is unmeasurable.
- [x] ~~Qwen3.5-0.8B at **Q4_K_M** (halves RAM — does 6/6 survive?)~~ **Rejected**
      (F43). Answering survives at 6/6 and disk drops to 508 MB, but F27's refusal
      recovery collapses from 6/6 to 4/6 because Q4_K_M fabricates commandless prose
      the check cannot see. Accuracy parity hid a safety regression.
- [ ] LFM2.5-350M *fine-tuned* for this one extraction task — still the only path back
      to a ~360 MB answerer.
- [x] ~~Would a purpose-built embedder beat bge-small?~~ **No** — nomic-embed-text
      v1.5 (137 M) scored *worse* on articles (F24). Embedder scaling is not the
      lever; `TNY_EMBED`/`TNY_QP`/`TNY_DP` make re-testing cheap if that changes.
- [ ] **Section selection**: 6/6 *presence* and now 6/6 *answers*, so it is no longer
      the measured weak link — but presence is a weak bar. Untried: h4-aware
      splitting, sentence-window selection, including the article lead with the top-3.
- [ ] Staleness is inherent to ZIM snapshots — "current stable Rust version" is
      unanswerable offline. Ship `--online` fallback (Wikipedia + Stack Exchange
      APIs, keyless, verified working) or accept the limit?
- [ ] **Stack Exchange ZIM article structure is unverified** (question + answers in
      one page?). Needed before the code-question path is built.
- [x] ~~Multi-book search: query all books, or route to one book first?~~ **Query all
      books, unrouted** (F34). One `/search` ties an oracle told the right book
      (12/15), picks the right book 15/15, and costs +47 ms. Per-book RRF fusion is
      worse (11/15) at 3× the requests.
- [ ] Does `tny` need tool calling at all (F6) if the harness routes deterministically?
- [ ] Three supervised processes (~1.3 GB RSS) — acceptable, or fold embeddings into
      the chat server's spare capacity?

## Reproducing

`bun bench/harness.mjs all` with the three servers up. Individual benchmarks:
`rank` (F17/F31), `judge` (F16), `sections` (F22), `answers` (F20), `corpus` (F21),
`refuse` (F26/F27/F44/F45), `select` (F31), `cross` (F34), `rerank`/`widen` (F39),
`file` (F41), `stdin` (F42), `detail` (F44), `followup` (F28), `rewrite` (F29),
`depth` (F30), `synth` (F36), `clarify` (F37), `thinking` (F19), and `ground` — the
F27/F32/F38/F44 self-test, which is pure: no servers, no network, exits non-zero on
failure.

`bun bench/quality.mjs "<label>"` dumps verbatim answers for whatever model is on port
8080; that is how F46's quality comparison was made, and it is the only way to see the
wrong-detail failures a needle scores as correct.

Every helper is exported and the CLI only dispatches under `import.meta.main`, so
`await import("./bench/harness.mjs")` gives `ungrounded`, `ungroundedDetail`,
`ungroundedShape`, `commandsIn`, `commandVocab`, `pickSections`, `pickSectionsLex`,
`rankArticles`, `search`, `searchAll`, `article`, `ask`, `embed`, `prep`, `terms`,
`lexScore`, `window`, `splitCompare`, `needsClarify`, `fileWindows` and `structChunks`
for ad-hoc probing without rebuilding anything.

Model swap: point `llama-server` at another GGUF on port 8080 and re-run — the harness
is model-agnostic. **Re-run `refuse`, not just `answers`**: Q4_K_M matched Q8_0 on
answering while silently halving what the grounding check could catch (F43).
Embedder swap: `TNY_EMBED`, `TNY_QP`, `TNY_DP`. Always pass `--no-mmproj` (F25).

One model server at a time — two at `-t 4` on a 4-core box makes the machine unusable
(see Benchmark hygiene).

### Runtime cost of each benchmark, measured

Model calls dominate everything. At ~7.8 tok/s decode and ~40 tok/s prefill, one answer
costs 10–35 s depending on context size, so a benchmark's runtime is just its call count
times that. Budget accordingly — and prefer the free ones when iterating.

| Command | Runtime | Model calls |
|---|---|---|
| `ground` | **0.4 s** | 0 — pure, run this constantly |
| `clarify` | 2 s | 0 |
| `cross` | 7 s | 0 |
| `rerank` | 13 s | 0 |
| `rank` | 30 s | 0 |
| `stdin` | 65 s | 6 |
| `refuse` | 90 s | 6 |
| `answers` | 110 s | 6 |
| `select` | 165 s | 0 (embedder only) |
| `file` | 120–230 s | 6 |
| `synth` | 220–340 s | 10 |
| `followup` | 275 s | 18 |
| `widen` | **610 s** | 12 |
| `all` | **~15 min** | ~60 |

Two rules that came out of this:

1. **Score deterministic arms model-free, then spend calls on the winner.** This turned
   `file` from 18 calls into 6 with no loss of information.
2. **A 6-case fixture cannot rank two options.** `file` scored 3/6 and 5/6 on the *same
   arm at the same budget*. Either sample repeatedly or do not claim a difference.
