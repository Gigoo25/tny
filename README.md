# tny

A fully offline terminal search engine with a small model as the front end. No network, no
API key, no telemetry: every answer comes from ZIM corpora on your own disk, and every answer
cites the articles it was built from.

```
$ tny "how do I create a swap file"
Use mkswap(8) to create a swap file of the size you want, activate it with `swapon`, and add
it to /etc/fstab to keep it across reboots.

  1 Arch Wiki · Swap   38.7s
```

Run it with no question for the full interface — a scrolling transcript, the shortlist of
everything retrieved, and vim keys to drive it.

## Install

Needs `llama-server` (llama.cpp) and `kiwix-serve` (kiwix-tools) on `PATH`; tny starts and
supervises both itself.

```sh
nix-shell -p llama-cpp kiwix-tools     # or: pacman -S llama.cpp kiwix-tools
cargo build --release
./target/release/tny --corpus pack small     # ~10 books, downloads to ~/.local/share/tny
```

Model weights and ZIMs live in one fixed place, `${XDG_DATA_HOME:-~/.local/share}/tny/`, so a
question answers the same from any directory (F77). The 0.8B model downloads on first use.

## Using it

```
tny "question"          one answer, then the interface
tny --corpus search bash        find ZIMs in the kiwix library
tny --corpus add devdocs_en_bash    download one
tny --corpus pack mini|small|medium|large|huge     download a shelf
tny --corpus update     check for newer editions
```

Piped output is plain text, so `tny q > file` and scripts behave like any other command.

### Three dials

Speed, length and model are independent, and what you pick in the interface is what you get
next time (`~/.cache/tny/prefs`).

| speed | reads | typical |
|---|---|---|
| `--ultrafast` | best passage from the page, **no model** | 0.25 s |
| `--fast` | one article | ~14 s |
| `--medium` | three articles (default) | ~39 s |
| `--slow` | three articles, twice as deep | ~50 s |
| `--molasses` | three articles, as deep as it gets | ~90 s |

`--low` / `--max` set answer length (one sentence / up to three paragraphs).
`--model 0.8b|2b|4b|<hf repo:quant>` picks who answers. Bigger is not automatically better
here — see `NOTES.md`.

### Keys

Modal, like vim. `i` types a question, `Esc` drives.

```
i / a       ask                      j k ^D ^U gg G   scroll
1-9         pick a source            ⏎                read it (skips retrieval)
r           ask that again           + -              speed
o           new topic                < >              length
:model 4b   switch model             :q               quit
```

Picking a source is the important one: the answer is built from the top articles, and when
it lands on the wrong one, the right page is usually already on screen a keypress away.

## How it works

```mermaid
graph LR
  Q[question] --> S[search 18 ZIMs<br/>per-book, parallel]
  S --> R[rank articles<br/>lexical, model-free]
  R --> C[pick sections<br/>+ window on the answer]
  C --> M[0.8B paraphrases]
  M --> G{grounded?}
  G -->|yes| A[answer + citations]
  G -->|no| N[not found]
```

The model never *selects* anything — ranking, query building and section choice are
deterministic rules, all of which beat the model when measured. It only paraphrases what it
was handed, and a regex grounding check rejects any answer containing a flag, path or version
that is not in the source text.

## Where it stands

4,266 lines of Rust, 36 commits, 107 findings. Every number below comes from `bench/`, and
each guard pins its own configuration (F105) so a run is reproducible.

| what it measures | command | current |
|---|---|---|
| grounding rules (no servers, no network) | `cargo test` | **16/16** |
| is the right article retrieved | `bun bench/rank-cli.mjs` | rank-1 **34/58**, in shortlist **48/58** |
| is the fact in the context handed to the model | `bun bench/ctx-cli.mjs` | **48/58** |
| is the answer right, end to end | `bun bench/answer-cli.mjs inst` | **11/18** correct, 2 refused, 5 wrong |
| questions nobody here wrote | `bun bench/holdout-cli.mjs` | title **55/57**, body **54/57**, ref **18/18** |

Corpus behind those numbers: 18 ZIMs, 5.5 GB — Wikipedia, Stack Exchange, Arch Wiki, DevDocs,
man pages. Hardware: 4-core 2015 laptop, no GPU, 8 GB RAM.

### What is honestly weak

- **It is slow.** 39 s for a default answer here, 85-90 % of it prefill. `--ultrafast`
  (0.3 s, no model) exists because sometimes the passage is all you wanted.
- **Refusals are common.** 2 of the 18 above, and nearer a quarter across the full 58. A
  refusal is the *designed* failure — the grounding check rejects any answer whose commands
  or paths are not in the source — but it is still a non-answer.
- **11/18 is the honest end-to-end number**, and the gap is mostly *selection*: in 43 of 58
  cases the answer is already a single sentence in the retrieved text (F79), and no selector
  tried — lexical, bi-encoder, two cross-encoders — picked it better than the model does.
- **Bigger models do not fix it.** 4B: half the wrong answers, 3.7× the wall clock, two fewer
  correct, does not fit in 8 GB (F107). 2B: no better (F90). 350M: twice as wrong (F81).
- **A 58-case fixture cannot resolve small differences**, and it was authored here against
  this corpus. `holdout-cli` counters that with 93 questions taken from the pages themselves,
  which nothing was tuned against.

## The measurements

- **`NOTES.md`** — 107 findings, each with the numbers behind it. Read this first; where the
  three documents disagree, it wins.
- **`PLAN.md`** — the design and the reasoning. Every decision cites a finding.
- **`bench/`** — the guards in the table above, plus `harness.mjs` for the pre-Rust work.

Headlines:

- **The corpus is the lever, not the model.** Same model, same questions: 3/6 from weights,
  6/6 with retrieval — and it converts confident fabrications (`mkfs -xfs -f` "to encrypt a
  partition") into correct commands.
- **Grounding is a regex, not a bigger model.** It buys 2B's entire measured advantage.
- **More context is not more accuracy.** A wider window around a flag pulls in the
  *neighbouring* flag and the model reports its meaning instead (F100).
- **The model must never select.** Ranking, query building and section choice are all done
  better by deterministic rules; every arm that let the model choose scored worse (F16, F79).
- **Retrieval is 2 % of the wall clock.** Prefill is 85-90 % of it, which is why the speed
  dial is a context-size dial.
