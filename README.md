# tny

A fully offline terminal search engine with a small model as the front end.
`tny "question"` → grounded answer on stdout, source on stderr.

Not built yet. This repo currently holds the measurements that decide how to build it.

- **`NOTES.md`** — 41 findings, each with the numbers behind it. Read this first.
- **`PLAN.md`** — the build sequence. Every decision cites a finding.
- **`bench/harness.mjs`** — the measurement harness. Reproduces every number in `NOTES.md`.

## Reproducing

Needs three servers (see `NOTES.md` → Environment). Model weights and ZIM corpora live
outside the repo, under `${XDG_DATA_HOME:-~/.local/share}/tny/` — one fixed location, so a
question answers the same from any directory (F77). `llama-server -hf` and `tny --corpus`
fetch them there.

```sh
nix-shell -p llama-cpp kiwix-tools
export TNY=${XDG_DATA_HOME:-~/.local/share}/tny
export LLAMA_CACHE=$TNY/models

kiwix-serve --port 8082 --address 127.0.0.1 $TNY/zim/*.zim
llama-server -hf ggml-org/Qwen3.5-0.8B-GGUF:Q8_0 --no-mmproj -t 4 -c 8192 --jinja --port 8080
llama-server -hf ggml-org/bge-small-en-v1.5-Q8_0-GGUF --embeddings --pooling cls -c 512 -t 4 --port 8084

bun bench/harness.mjs all      # everything
bun bench/harness.mjs ground   # pure self-test, no servers, no network
```

`bun bench/harness.mjs ground` is the only one that needs nothing running, and it is the
test that matters: 23 cases pinning the grounding rules that stop the model serving a
fabricated shell command as fact.

Override endpoints with `TNY_CHAT`, `TNY_EMBED`, `TNY_KIWIX`, `TNY_BOOK`.
Always pass `--no-mmproj` — Qwen3.5 GGUFs are vision models and llama.cpp otherwise
downloads a 671 MB projector for nothing.

## The short version of what was measured

- **Qwen3.5-0.8B Q8_0, thinking OFF.** 2B scored identically at 2.2× the latency; 350M
  degenerates into echoing the question; 1.2B-Thinking never stops thinking.
- **The corpus is the lever, not the model.** Same model, same questions: 3/6 from
  weights, 6/6 with retrieval — and it converts confident fabrications
  (`mkfs -xfs -f` to "encrypt a partition") into correct commands.
- **The model never selects.** Ranking, query rewriting and judging are all done better
  and cheaper by deterministic rules; the model only paraphrases what it was handed.
- **Grounding is a regex, not a bigger model.** It buys 2B's entire measured advantage.
