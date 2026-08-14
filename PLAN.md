# tny — build plan

Findings and evidence live in `NOTES.md`; every decision here cites one. This file
is the build sequence.

## What it is

A fully offline terminal search engine with a small model as the front end.
`tny "question"` → grounded answer on stdout, source on stderr.

```
tny "how do I set the system timezone"      # Arch Wiki ZIM
tny "what does Vec::with_capacity do"       # DevDocs ZIM, anchor-sliced
tny "how many neurons does C. elegans have" # Wikipedia ZIM
tny "summarize src/main.rs"                 # local file
tny -d "explain what a mutex is"            # no lookup
tny "ext4 or btrfs"                         # comparison -> asks which side you mean
tny --corpus list|add <name>|rm <name>      # manage ZIMs
tny -v "..."                                # per-stage timings
tny --stop                                  # stop supervised servers
```

Answer → **stdout**. Source (book · article · sections), timings, errors →
**stderr**, so `tny "..." | pbcopy` does the obvious thing.

## The pipeline (every stage measured)

```
query  (+ previous turns, if any)
  ├─ 0. follow-up? search query = "<prev question> <this question>"  F29  5/6 vs 2/6 raw
  │        NEVER a model rewrite: it inverted "turn it off" → "turn it back on"
  ├─ 0b. comparison? split BEFORE prep — prep strips the very words the
  │        split needs (F35). "versus|vs|or|difference between … and", F37  6/6 detected,
  │        carry the shared tail into both sides, retrieve each.            0/26 false fires
  │        Two different articles → ASK the user which side they mean.
  │        The user selects; the model never does (F16, F29).
  ├─ 1. prep: strip question words, stopwords, and     F15/F35  raw query → 0 hits;
  │        comparison words — kiwix ANDs every term            "string versus str
  │        so one stray word returns nothing at all            slice" → 0 hits
  ├─ 2. search ALL mounted ZIMs, no routing              F12  content picks the book
  │      /search?pattern=…&format=xml  (no books.name)  F11  title+link+snippet
  │      each <link> names its book; `_ftindex:no` ZIMs F12  real body-level FTS
  │      are searched too — the tag must not gate this
  ├─ 3. filter: drop "(Magyar)"-style dupes, dedupe      F14  half the list was noise
  ├─ 4. rank: RRF(xapian order, lexical title+snippet)   F31  9/10, no embedder
  ├─ 5. fetch article by path                            F11
  ├─ 6. extract
  │      wiki pages → split h2–h5, embed-rank sections,  F31  14/14 in 1.5 KB
  │                   take top 3, window each on terms        (h2-only: 12/14)
  │      ref  pages → slice by #anchor                   F13  240 anchors survive
  ├─ 7. denoise: citation markers, [edit], link refs      F8
  ├─ 8. answer: Qwen3.5-0.8B, thinking OFF, ≤160 tok      F19/F20  6/6
  │        keep prior turns in the message list           F28  83% vs 75% stateless
  └─ 9. verify grounding against the FULL article,        F27/F32  refusal 4/6 → 6/6,
         else say "not found"                                      0 false rejects
```

Stage 6 is the only embedder user — 35 MB of bge-small, which reaches 14/14 in **44 %
less context** than the model-free lexical fallback needs, and prefill dominates
latency here (F31). Stage 4 dropped its embedding calls for a free lexical fusion at
equal score. Stage 8 never selects anything — the chat model only adapts and formats
(F16, F29) — and stage 9 is a regex, not a model: it rejects answers proposing
commands absent from the source article, which is the entire measured advantage of a
2.5× larger model (F26, F27).

Grounding reads the **whole fetched article**, not the 1.5 KB slice sent to the model:
the slice rejected a correct answer for citing `cryptsetup` from a neighbouring section
(F32).

## Shape

```
main.rs       arg parse, orchestration, grounding check, output ~230 loc
supervise.rs  spawn / health / reuse of three servers         ~130 loc
retrieve.rs   search, suggest, filter, RRF, sections, anchors ~330 loc
              + comparison split / clarify prompt
corpus.rs     catalog parse, ZIM download + verify + resume   ~140 loc
```

Four files, ~830 lines. Three deps: `ureq` (blocking HTTP + TLS + gzip),
`serde_json`, `regex`.

**No async runtime** — the pipeline is sequential except one batched embeddings
call; tokio/hyper/reqwest would add ~200 crates to serialize the same round trips.
**No `clap`** (few flags), **no HTML parser** (regex strip + `indexOf` anchor
slicing is exactly what was verified), **no `serde` derive** (`serde_json::Value`
indexing over read-once responses), **no `dirs`** (3 lines of XDG fallback).

## Load-bearing decisions

**1. Answering model: Qwen3.5-0.8B Q8_0 with thinking OFF.**
Identical contexts, answer verified present 6/6: 350M **2/6**, Qwen-0.8B **6/6**,
Qwen3.5-**2B** also **6/6** at 2.2× the latency — so the size ladder was climbed and
stopped on evidence, not taste (F26).
350M is rejected because with perfect context it *degenerates* — echoed the question
for ssh-keygen, emitted "Encrypt a partition using LUKS mode." with no command
(F23). `chat_template_kwargs:{"enable_thinking":false}` is **mandatory**: raw output
shows Qwen opens `<think>` and never closes it inside 512 tokens, burning 95.5 s for
zero answer (F19).

Upward pressure was tested and answered: 2B's only edge was refusing an unanswerable
context 5/6 vs 4/6, and stage 10's regex takes both models to 6/6 (F27). Downward,
350M cannot be rescued by better retrieval.

**2. Empty `content` is an error, never an answer.** llama.cpp routes reasoning to
`reasoning_content`, so a naive client prints blanks silently (F19).

**3. Selection model: bge-small-en-v1.5 Q8_0 (33 M params, 35 MB).** Beats the chat
model at choosing (F16 vs F17) at 1/25 the size. Query prefix `"Represent this
sentence for searching relevant passages: "` is required.

**4. The chat model never chooses.** 350M emitted a near-constant index (#7,#7,#5,
#7,#7,#8); Qwen judged 3/6 versus free rank-1's 4/6 and RRF's 5/6 (F16).

**5. ZIM-only knowledge via supervised `kiwix-serve`.** One mechanism, offline, no
keys, no rate limits. Book id is the **filename stem**, not the ZIM's internal name.

**6. Retrieval strategy per book from its `_ftindex` tag** (F12). DevDocs ZIMs have
no full-text index — `/search` 400s there and `/suggest` is the only route.

**7. Fuse at article level, embed-only at section level.** RRF gave +1 on articles
but *hurt* sections, displacing the correct §"du alternatives" (F17, F22).

**8. Top-3 sections, not the best one.** Argmax was 4/6; three 600-char sections put
the answer in context **6/6** at ~1570 chars (F22). Cheaper than a perfect argmax.

**9. Terse contract, `max_tokens` 160, no refusal escape hatch** (F8, F9). Emptiness
is detected in the harness: if no candidate survives filtering or the extracted
context is under ~120 chars, print `no results` to stderr, exit 1, and **never call
the model**.

**10. Positive phrasing only** — "Do NOT" measurably backfires (F7).

**11. Print the source.** Book · article · section headings, so a wrong answer is
checkable rather than silent.

**12. Three supervised processes** (`kiwix-serve`, chat, embeddings), ~1.3 GB RSS.
Dropping bge saves a process and 2.3 s/query but costs −1/10 on articles *and* halves
section accuracy (F22) — the section win is what earns it.

**13. llama.cpp downloads both models** via `-hf ggml-org/Qwen3.5-0.8B-GGUF:Q8_0`
and `-hf ggml-org/bge-small-en-v1.5-Q8_0-GGUF`. Zero model-download code in `tny`.

## Corpus profiles (real sizes, `NOTES.md` catalogue)

Disk is the one genuine tradeoff, so ship tiers rather than one default download.

| Profile | Contents | Size |
|---|---|---|
| **min** | devdocs (rust, python, bash, git, go, javascript) + archlinux | ~82 MB |
| **standard** | + devdocs_en_man + mankier + wikipedia_en_100 | ~630 MB |
| **wide** | + wikipedia_en_wp1-0.8 mini (855 k articles) | ~3 GB |
| **deep** | + unix.stackexchange + topical SE sites | ~5 GB |
| **max** | + wikipedia_en_all mini (12.5 GB) | ~18 GB |

Never `stackoverflow.com_en_all` (**80.5 GB**) by default. Catalog:
`library.kiwix.org/catalog/v2/entries?lang=eng&count=-1`; download links arrive as
`.zim.meta4` — strip `.meta4`.

## Caches — `${XDG_CACHE_HOME:-~/.cache}/tny/`

```
zim/          ZIM files (managed by `tny --corpus`)
models/       LLAMA_CACHE for both GGUFs
books.json    book id, _ftindex flag, article count (from the local catalog)
```

No answer cache, no config file — env vars only (`TNY_ZIM_DIR`, `TNY_PORT`,
`TNY_THREADS`, `TNY_LLAMA_SERVER`, `TNY_KIWIX_SERVE`).

## Phases

Each ends runnable; none merged on a claim.

**P0 — spine.** `supervise.rs`: spawn/reuse/health for three servers, arg parse,
streaming, stdout/stderr split, empty-`content` guard (F19).
Check: `tny -d "explain what a mutex is"` streams an answer; a second invocation
skips model load; `tny --stop` stops all three.

**P1 — retrieval.** Query prep, book discovery via `/catalog/v2/entries` including
`_ftindex`, `/search`, localisation filter, dedupe, batched embeddings, RRF.
Check: `tny -v "mount a usb drive automatically"` ranks **Udisks** first and prints
the ranking; `bun bench/harness.mjs rank` still scores ≥9/10.

**P2 — extraction + answer.** Section split, embed-rank top-3, denoise, terse
contract, source line.
Check: `bun bench/harness.mjs sections` → answer present 6/6; `answers` → 6/6 with
`timedatectl set-timezone`, `mkswap`, `ssh-keygen`, `cryptsetup luksFormat` appearing,
and `refuse` → 6/6 safe with 0 false rejects (F27's check is part of this phase, not
a later hardening pass — it is what keeps 0.8B honest).

**P3 — reference books.** `/suggest?content=` → path → `#anchor` slice; exact →
prefix → substring name matching.
Check: `tny "what does Vec::with_capacity do"` returns the real method, not
nightly-only `try_with_capacity` (the F13 trap).

**P4 — corpus management.** `tny --corpus list|add|rm`, catalog parse, profile
install, `.meta4` strip, free-space preflight.
Check: `tny --corpus add archlinux` fetches and serves with no manual `kiwix-serve`.

**P5 — local files.** Path-token detection → read → same contract. Fixes the one V2
routing failure (F7) deterministically instead of by prompt.
Check: `tny "summarize src/main.rs"` reads the file, no search.

## Verification

`#[cfg(test)]` asserts on the pure functions — `prep`, localisation filter, dedupe,
RRF ordering, section split, anchor slice, **grounding check** — against fixed
strings. The grounding tests are the **17 cases already passing** in
`bun bench/harness.mjs ground` (pure, no servers), which is the port target: allow a
short `du -h`, a fenced block, a bare `# cmd` prompt line, an unmarked command, a
quoted path and a refusal; catch `ssh-keygen` absent from a swap reference,
`mkfs.ext4` for a mount question, a question echo, a question asked back, and empty
content. No network, no
framework. Network paths are covered by the per-phase commands and
`bench/harness.mjs`.

Three regression tests earn their place:
- **F9**: the recorded Stack Exchange answer through the real system prompt must not
  produce a refusal — guards the escape hatch from creeping back.
- **F19**: a mocked response with empty `content` and populated `reasoning_content`
  must surface as an error, never as a blank answer.
- **F27**: the 17 grounding cases, which are the only thing standing between a
  fabricated command and the user's shell.

## Deliberately not built

- **Model-as-judge** — at 350M it emits a near-constant index; at 0.8B it ties free
  rank-1 (4,4,5 vs 4/6) while mostly reproducing it, loses to RRF's 9/10, and costs a
  full model call to do so (F16). Not worth its latency, whatever its score.
- **Online fallback** — ZIM snapshots cannot answer "current stable Rust version".
  Deferred behind `--online`; the Wikipedia and Stack Exchange APIs are keyless and
  were verified working, so this is a small addition when the limit bites.
- **Thread autotune** — F3's +25% hyperthread win was on 1.2B Q4_0, untested for
  Qwen-0.8B Q8_0. `-t` = physical cores, `TNY_THREADS` overrides.
  `ponytail: fixed thread count; autotune+cache only if measured to matter.`
- Multi-hop retrieval, conversation history, TUI, tool-call loop — no measurement
  demands them yet.
