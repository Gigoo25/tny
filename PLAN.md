# tny — build plan

Findings and evidence live in `NOTES.md`; every decision here cites one. This file
is the build sequence.

## Status

**All of P0-P4 is built and measured** (P5 was cut — see *Deliberately not built*). This file
is the design and the reasoning; `README.md` is how to run it and where it currently stands;
`NOTES.md` is the 107 findings with the numbers. When they disagree, `NOTES.md` wins — it is
the only one written at the time of measurement.

## What it is

A fully offline terminal search engine with a small model as the front end.

```
tny "how do I set the system timezone"      # Arch Wiki ZIM
tny "what does Vec::with_capacity do"       # DevDocs ZIM, anchor-sliced
tny "how many neurons does C. elegans have" # Wikipedia ZIM
tny "ext4 or btrfs"                         # comparison -> both sides retrieved
tny                                         # the interface: transcript, sources, vim keys
tny --ultrafast|--fast|--slow|--molasses    # how much to read (F94)
tny --low|--max                             # how much to write (F102)
tny --model 0.8b|2b|4b                      # who answers (F101)
tny --corpus list|search|add|pack|update    # manage ZIMs
tny -v "..."                                # per-stage timings
```

Answer → **stdout**. Sources, timings, errors → **stderr**, so `tny "..." | pbcopy` does the
obvious thing. A terminal with no redirection gets the full interface instead (F96).

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
  ├─ 2. search each mounted ZIM separately, 8 hits each   F49  one shared query dilutes
  │      /search?books.name=…&pattern=…&format=xml       F63  8 hits lifts recall 54→55
  │      four books in flight; `_ftindex:no` ZIMs are    F12  the tag lies both ways
  │      searched too, and probed on arrival (F89)
  ├─ 3. filter: drop "(Magyar)"-style dupes, dedupe      F14  half the list was noise
  ├─ 4. rank: lexical title+body+kind prior, no model    F49/F59/F91  RRF lost, 17/32
  ├─ 5. fetch the top 1-3 articles by path               F58  answer in top-3 for 45/58
  ├─ 6. extract
  │      wiki pages → split h2–h5, lexical section rank, F31  14/14 in 1.5 KB
  │                   take top N, window each on terms         (h2-only: 12/14)
  │      flag questions → window on the flag's own entry F93  48/58 in context, +2
  │      ref  pages → slice by #anchor                   F13  240 anchors survive
  ├─ 7. denoise: citation markers, [edit], link refs      F8
  ├─ 8. answer: Qwen3.5-0.8B, thinking OFF, 80–512 tok    F19/F20  6/6
  │        keep up to five prior turns in the message list F84  "it" resolves 3 deep
  └─ 9. verify grounding against the FULL article,        F27/F32  refusal 4/6 → 6/6,
         else say "not found"                                      0 false rejects
```

**No embedder.** Stage 6 used one for six sessions — 35 MB of bge-small, 14/14 in 44 % less
context than lexical needed (F31) — and F79 removed it: against the 58-case fixture a
bi-encoder scored 21/58 and two cross-encoders 17–19/58, all at or below one line of lexical
scoring, and far below the model reading three articles (42/58). Stage 4 dropped its
embedding calls earlier for the same reason. Stage 8 never selects anything — the chat model
only adapts and formats (F16, F29, F79) — and stage 9 is a regex, not a model: it rejects
answers proposing commands absent from the source article, which is the entire measured
advantage of a 2.5× larger model (F26, F27), and of a 5× larger one (F107).

Grounding reads the **whole fetched article**, not the 1.5 KB slice sent to the model:
the slice rejected a correct answer for citing `cryptsetup` from a neighbouring section
(F32).

## Shape

```
main.rs       arg parse, orchestration, grounding, dials, cache   1,773 loc
retrieve.rs   search, filter, rank, sections, windowing, extract    861 loc
corpus.rs     catalog parse, packs, ZIM download + verify + resume  615 loc
tui.rs        transcript, shortlist, steering, vim keys             600 loc
ground.rs     the grounding rules and html→text                    417 loc
```

Five files, 4,266 lines. Four deps: `ureq` (blocking HTTP + TLS + gzip), `serde_json`,
`regex`, `ratatui` (+`crossterm`).

**No async runtime** — the pipeline is sequential apart from four book searches in flight
(F78: `std::thread`, 1.07 s → 0.67 s) and the TUI's one worker thread; tokio/hyper/reqwest
would add ~200 crates to serialize the same round trips. **No `clap`** (the flags are a
`match` on `&str`), **no HTML parser** (regex strip + `indexOf` anchor slicing is exactly what
was verified), **no `serde` derive** (`serde_json::Value` indexing over read-once responses),
**no `dirs`** (3 lines of XDG fallback).

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

**3. Selection is lexical, and there is no selection model.** bge-small held this slot for
six sessions on a 14-case fixture. On the 58-case one it lost: bi-encoder 21/58, cross-encoder
(jina-tiny 33 M) 17/58, cross-encoder (bge-reranker-base 278 M) 19/58, lexical 21/58, and the
model reading three articles 42/58 — while an oracle over every sentence reaches 43/58, so the
answer *is* one sentence in the text and no reranker finds it (F79). One server fewer, one
model fewer, 2.3 s per question saved.

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

**12. Two supervised processes** (`kiwix-serve`, chat), ~1.2 GB RSS. The embedder was the
third for six sessions; F79 removed it — the section win it was earning on a 14-case fixture
did not survive the 58-case one. `serve_kiwix` re-checks on every question that the running
server's mounted set still matches the ZIMs on disk, because a downloaded book is not a
mounted book (F89).

**13. llama.cpp downloads the model** via `-hf ggml-org/Qwen3.5-0.8B-GGUF:Q8_0`, and any
model named by `--model` the same way. Zero model-download code in `tny`; ZIMs are a different
matter — those `tny --corpus` fetches itself, resumable and byte-verified.

## Corpus profiles (real sizes, `NOTES.md` catalogue)

Disk is the one genuine tradeoff, so ship tiers rather than one default download.

Shipped as `tny --corpus pack <name>`, sizes from the live catalogue (F87):

| Pack | Contents | Size |
|---|---|---|
| **mini** | devdocs man, bash, git, python + a general floor | ~28 MB |
| **small** | + devdocs docker, postgresql, rust, cpp, archlinux | ~313 MB |
| **medium** | + wikipedia topical (computer, physics, chemistry, maths) | ~3.1 GB |
| **large** | + Stack Exchange, ifixit, wikibooks | ~18.9 GB |
| **huge** | + all of Wikipedia, all of Stack Overflow, Wiktionary | ~151 GB |

The two genuinely surprising catalogue finds: **12,626 man pages for 28 MB**, the best
value in the whole library for a terminal tool, and `docs.python.org` at 92 MB against
DevDocs' 4 MB for the same language (F87).

Never `stackoverflow.com_en_all` (**80.5 GB**) by default. Catalog:
`library.kiwix.org/catalog/v2/entries?lang=eng&count=-1`; download links arrive as
`.zim.meta4` — strip `.meta4`.

## Storage

`${XDG_DATA_HOME:-~/.local/share}/tny/` — one fixed location, never the working directory
(F77: preferring `./zim` when it existed made the same question answer differently depending
on where the user stood).

```
zim/          ZIM files (managed by `tny --corpus`)
models/       LLAMA_CACHE for both GGUFs
```

`${XDG_CACHE_HOME:-~/.cache}/tny/` — only what is regenerable.

```
books.json    book id, _ftindex flag, article count (from the local catalog)
```

No answer cache, no config file — env vars only (`TNY_ZIM_DIR`, `TNY_PORT`,
`TNY_THREADS`, `TNY_LLAMA_SERVER`, `TNY_KIWIX_SERVE`).

## Phases — all shipped

Each ended runnable; none was merged on a claim. Kept as a record of what each one had to
prove, because the checks are still the fastest way to tell whether something has rotted.

**P0 — spine.** Spawn/reuse/health for two servers (the embedder was dropped — F79: every
embedding-based selector lost to lexical scoring), arg parse, stdout/stderr split,
empty-`content` guard (F19), mount-drift detection (F89).
Check: `tny "what is a swap file"` answers with both servers cold; a second invocation is a
cache read (F85).

**P1 — retrieval.** Query prep, per-book search (F49), localisation filter, dedupe, lexical
ranking. RRF exists behind `TNY_RANK=rrf` and lost (F91).
Check: `bun bench/rank-cli.mjs` → `article@1 34/58 · in shortlist 48/58 · book@1 44/58`.

**P2 — extraction + answer.** Section split, lexical section selection, density and
flag-entry windowing (F31/F93/F100), denoise, terse contract, grounding check, source line.
Check: `bun bench/ctx-cli.mjs` → `context has the fact 48/58`; `cargo test` → 16 passing,
which are the grounding cases that keep a 0.8B honest.

**P3 — reference books.** `/search?books.name=` → path → section slice; exact → prefix →
substring name matching; searchability probe on every new book (F89).
Check: `tny "what does the -p flag do in mkdir"` answers from `mkdir(3p)`, not from a page
that merely mentions it (F93).

**P4 — corpus management.** `tny --corpus list|search|add|pack|update`, catalog parse, packs
by shelf, resumable byte-verified downloads, 30-day staleness check offered *after* an
answer (F88).
Check: `tny --corpus pack mini` fetches and serves with no manual `kiwix-serve`.

**P6 — the interface.** ratatui TUI: scrolling transcript, shortlist, steering, three
persisted dials, vim keys (F94-F103).
Check: `tny`, type a question, `1`-`9` then `⏎` to read a different source, `:q`.

**P5 — local files.** Cut, not built (F41). See *Deliberately not built*.

## Verification

`#[cfg(test)]` asserts on the pure functions — `prep`, term selection, localisation filter,
dedupe, section split, windowing, **grounding check** — against fixed strings: **16 tests,
`cargo test`, no network and no framework.** They are the port of the grounding cases from
`bun bench/harness.mjs ground`: allow a short `du -h`, a fenced block, a bare `# cmd` prompt
line, an unmarked command, a quoted path and a refusal; catch `ssh-keygen` absent from a swap
reference, `mkfs.ext4` for a mount question, a question echo, a question asked back, and
empty content.

Everything that touches the network or the corpus is a `bench/*-cli.mjs` guard against the
58-case fixture, and each pins its own configuration (F105). Current numbers are in
`README.md`; how each was arrived at is in `NOTES.md`.

Three regression tests earn their place:
- **F9**: the recorded Stack Exchange answer through the real system prompt must not
  produce a refusal — guards the escape hatch from creeping back.
- **F19**: a mocked response with empty `content` and populated `reasoning_content`
  must surface as an error, never as a blank answer.
- **F27**: the 17 grounding cases, which are the only thing standing between a
  fabricated command and the user's shell.

## Deliberately not built

- **Local file reading** — was in the CLI surface; **cut** (F41). Excerpt selection works
  (6/6 answer present), but answering scored 3/6 then 5/6 on the *same arm at the same
  budget*, so nothing is established, and one "pass" was an answer echoing the needle's
  own words. Needs 15+ cases with expectations a regurgitation cannot satisfy before it
  is claimed again.
- **Two-article synthesis** — 2–3/5, and given one side only it invents the other
  (F36). Comparisons take the ask-the-user path instead (F37).
- **Model-as-judge** — at 350M it emits a near-constant index; at 0.8B it ties free
  rank-1 (4,4,5 vs 4/6) while mostly reproducing it, loses to RRF's 9/10, and costs a
  full model call to do so (F16). Not worth its latency, whatever its score.
- **Online fallback** — ZIM snapshots cannot answer "current stable Rust version".
  Deferred behind `--online`; the Wikipedia and Stack Exchange APIs are keyless and
  were verified working, so this is a small addition when the limit bites.
- **Thread autotune** — F3's +25% hyperthread win was on 1.2B Q4_0, untested for
  Qwen-0.8B Q8_0. `-t` = physical cores, `TNY_THREADS` overrides.
  `ponytail: fixed thread count; autotune+cache only if measured to matter.`
- **Multi-hop retrieval, tool-call loop, streaming** — no measurement demands them. Streaming
  in particular buys ~nothing here: prefill is 85-90 % of an answer and nothing can be shown
  until it finishes, and the grounding rules need the whole answer before printing any of it
  (F80).
- **Conversation history and a TUI were on this list and are now built** — a five-turn rolling
  window (F84) and the interface in P6. The measurement that moved them: three turns deep,
  "how big should it be" could not resolve "it" without the first turn (F84), and a 20-90 s
  answer in a line-based prompt leaves the screen frozen with no way to change your mind
  (F96).
