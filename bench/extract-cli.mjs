#!/usr/bin/env bun
// F76 control: how many of the 58 cases does the model actually win?
//
// `tny` spends 12-39 s per question generating, which is 60-95 % of its wall clock, to read
// ~3 KB of retrieved text and repeat one fact from it. That is extraction, not reasoning —
// so the honest control is extraction with no model at all: take the same context the model
// is handed, print the best-scoring sentence, and grade it with the same grader.
//
// The comparison is only meaningful because both arms see identical context: this calls
// `tny --context`, which stops after retrieval and prints the exact slice the model receives.
//
// Contexts cost ~3 s each and never change between selection variants, so they are cached and
// every variant after the first is free (the F56 split, again).
//   bun bench/extract-cli.mjs            score the default variant
//   bun bench/extract-cli.mjs --sweep    every variant, offline, from the cache
//   bun bench/extract-cli.mjs --rebuild  re-fetch contexts
import { CASES, verdictOf, tallyOf } from "./grade.mjs";

const CACHE = "bench/.contexts.json";
const rebuild = process.argv.includes("--rebuild");
const sweep = process.argv.includes("--sweep");
const show = process.argv.includes("--show");
const cache = !rebuild && await Bun.file(CACHE).exists() ? JSON.parse(await Bun.file(CACHE).text()) : {};

async function contextOf(query) {
  if (cache[query] != null) return cache[query];
  const p = Bun.spawn(["./target/release/tny", "--context", query], { stdout: "pipe", stderr: "pipe" });
  const out = (await new Response(p.stdout).text()).trim();
  await p.exited;
  cache[query] = out;
  return out;
}

// Sentence splitting over encyclopedia prose. Abbreviations ("e.g.", "U.S.", "Dr.") and
// decimals ("42.195 km") are the two things that shatter a naive split, and both appear in
// the fixtures, so the boundary needs a capital letter after it and a non-abbreviation before.
const ABBR = /\b(?:[A-Z]|e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Ms|St|Inc|Ltd|Fig|approx|ca|cf|al)$/;
function sentences(text) {
  const out = [];
  for (const block of text.split("\n")) {
    // `## Heading` lines are structure, not prose: they carry query terms without carrying a
    // fact, so scoring them would hand every headline question its own title back.
    const line = block.trim();
    if (!line || line.startsWith("##")) continue;
    let start = 0;
    for (let i = 0; i < line.length - 1; i++) {
      if (!".!?".includes(line[i])) continue;
      if (!/\s/.test(line[i + 1])) continue;           // 42.195, e.g.foo
      if (ABBR.test(line.slice(Math.max(0, i - 6), i))) continue;
      if (!/[A-Z"'(]/.test(line.slice(i + 1).trimStart()[0] ?? "")) continue;
      out.push(line.slice(start, i + 1).trim());
      start = i + 1;
    }
    const tail = line.slice(start).trim();
    if (tail) out.push(tail);
  }
  // A fragment cannot carry an answer, and a 400-character wall is not an answer either.
  return out.filter(s => s.length >= 40 && s.length <= 400);
}

const STOP = /^(a|an|the|and|or|of|to|in|on|at|for|is|are|was|were|be|do|does|did|how|what|why|when|where|which|who|i|my|me|it|its|you|your|we|can|with|that|this|from|by|as|if|so|not|no|but|about|into|over|use|used|using)$/i;
const terms = q => q.toLowerCase().match(/[a-z0-9][a-z0-9'-]*/g)?.filter(w => !STOP.test(w)) ?? [];

// Variants. Each takes the context and the question and returns an answer string; the point
// of the sweep is that the cheapest one that scores well is the one to ship.
const VARIANTS = {
  // The first real sentence: on Wikipedia the lead's opening sentence is the definition, and
  // F67 already showed how much of the answer lives there.
  lead: (ctx) => sentences(ctx)[0] ?? "",

  // Distinct query terms present, length-normalised so a 300-character sentence does not win
  // by covering the query with sheer surface area.
  overlap: (ctx, q) => best(ctx, q, (s, t) => cover(s, t) / Math.sqrt(s.length)),

  // Rarity-weighted: a term appearing in one sentence of the context is worth more than one
  // appearing in twenty. This is IDF computed over the context itself — no corpus statistics
  // needed, which is what F71 failed to get from kiwix.
  idf: (ctx, q) => {
    const sents = sentences(ctx);
    const df = new Map();
    for (const s of sents) for (const w of new Set(terms(s))) df.set(w, (df.get(w) ?? 0) + 1);
    const w = t => Math.log(1 + sents.length / (1 + (df.get(t) ?? 0)));
    return best(ctx, q, (s, ts) => {
      const have = new Set(terms(s));
      return ts.filter(t => have.has(t)).reduce((a, t) => a + w(t), 0) / Math.sqrt(s.length);
    });
  },

  // The best sentence plus the one after it: facts split across a sentence boundary ("The
  // adult hermaphrodite has 959 somatic cells. The male has 1033.") are common in prose.
  window: (ctx, q) => {
    const sents = sentences(ctx);
    const i = bestIndex(sents, q, (s, t) => cover(s, t) / Math.sqrt(s.length));
    return sents.slice(i, i + 2).join(" ");
  },

  // The whole best-scoring section, not a sentence. Not shippable — it is a wall of text,
  // not an answer — but it bounds what any extractor can reach: if the fact is not in here,
  // no selection rule can find it.
  section: (ctx, q) => {
    const secs = ctx.split(/\n(?=## )/).filter(s => s.trim());
    const ts = [...new Set(terms(q))];
    let top = "", bs = -1;
    for (const s of secs) {
      const v = cover(s, ts) / Math.sqrt(s.length);
      if (v > bs) { bs = v; top = s; }
    }
    return top;
  },

  // The lead sentence and the best-scoring one: the definition plus whatever the question
  // actually asked about. Two sentences is still an answer a person would read.
  leadbest: (ctx, q) => {
    const sents = sentences(ctx);
    const i = bestIndex(sents, q, (s, t) => cover(s, t) / Math.sqrt(s.length));
    return i <= 0 ? (sents[0] ?? "not found") : `${sents[0]} ${sents[i]}`;
  },

  // A bi-encoder scores the question and each sentence separately and compares vectors. It
  // is the cheapest model that can see past vocabulary — the lexical arms' failure is that
  // an answer sentence rarely repeats the question's words.
  embed: async (ctx, q) => {
    const sents = sentences(ctx);
    if (!sents.length) return "not found";
    // bge-small is asymmetric: queries carry an instruction prefix, passages do not.
    const [qv, ...svs] = await embed([`Represent this sentence for searching relevant passages: ${q}`, ...sents]);
    let bi = 0, bs = -Infinity;
    for (const [i, v] of svs.entries()) {
      const s = cos(qv, v);
      if (s > bs) { bs = s; bi = i; }
    }
    return sents[bi];
  },

  // A cross-encoder reads the question and the sentence together and scores the pair. This is
  // the shape the task actually has — "does this sentence answer this question" — and it is
  // what a purpose-built 33M reranker is trained to do, against a 0.8B general model doing it
  // as a side effect of generating prose.
  rerank: async (ctx, q) => {
    const sents = sentences(ctx);
    if (!sents.length) return "not found";
    const scores = await rerank(q, sents);
    let bi = 0, bs = -Infinity;
    for (const [i, s] of scores.entries()) {
      if (s > bs) { bs = s; bi = i; }
    }
    return sents[bi];
  },
};

// Both servers are llama.cpp. Embeddings are cached on disk because a sweep re-scores the
// same sentences repeatedly and each pass would otherwise cost a full CPU embed of the corpus.
const EMBED = process.env.TNY_EMBED ?? "http://127.0.0.1:8084";
const RERANK = process.env.TNY_RERANK ?? "http://127.0.0.1:8085";
const EV = "bench/.embeds.json";
const evCache = await Bun.file(EV).exists() ? JSON.parse(await Bun.file(EV).text()) : {};
let evDirty = false;

async function embed(texts) {
  const miss = [...new Set(texts.filter(t => !evCache[t]))];
  // Batching buys nothing — llama.cpp bills per token, and a batch of 8 costs 8x one — but
  // the server has four slots, so four requests in flight is a straight 4x. Measured: 77 ms
  // per sentence serial, 23 sentences per question.
  const post = async (batch) => {
    const r = await fetch(`${EMBED}/v1/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: batch.map(t => t.slice(0, 1200)) }),
    });
    const j = await r.json();
    if (!j.data) throw new Error(`embed failed: ${JSON.stringify(j).slice(0, 200)}`);
    for (const [k, d] of j.data.entries()) evCache[batch[k]] = d.embedding;
    evDirty = true;
  };
  const batches = [];
  for (let i = 0; i < miss.length; i += 4) batches.push(miss.slice(i, i + 4));
  for (let i = 0; i < batches.length; i += 4) {
    await Promise.all(batches.slice(i, i + 4).map(post));
  }
  return texts.map(t => evCache[t]);
}

async function rerank(query, docs) {
  const r = await fetch(`${RERANK}/v1/rerank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, documents: docs.map(d => d.slice(0, 1200)), top_n: docs.length }),
  });
  const j = await r.json();
  if (!j.results) throw new Error(`rerank failed: ${JSON.stringify(j).slice(0, 200)}`);
  const out = new Array(docs.length).fill(-Infinity);
  for (const x of j.results) out[x.index] = x.relevance_score;
  return out;
}

const cos = (a, b) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb) || 1);
};

const cover = (s, ts) => {
  const have = new Set(terms(s));
  return ts.filter(t => have.has(t)).length;
};
function bestIndex(sents, q, score) {
  const ts = [...new Set(terms(q))];
  let bi = 0, bs = -1;
  for (const [i, s] of sents.entries()) {
    const v = score(s, ts);
    if (v > bs) { bs = v; bi = i; }
  }
  return bs <= 0 ? -1 : bi;
}
function best(ctx, q, score) {
  const sents = sentences(ctx);
  const i = bestIndex(sents, q, score);
  // No term overlap at all is a refusal, not a guess: the honest outcome when the context
  // does not visibly contain the answer, and the same choice the model's grounding makes.
  return i < 0 ? "not found" : sents[i];
}

const t0 = Date.now();
for (const [, query] of CASES) await contextOf(query);
// Always: a sweep that fetched 58 contexts and threw them away made the next run pay again.
await Bun.write(CACHE, JSON.stringify(cache, null, 1));
const fetched = ((Date.now() - t0) / 1000).toFixed(0);

// Named arms run only when asked: the model-backed ones need their server up.
const pick = process.argv.slice(2).filter(a => !a.startsWith("-"));
const arms = pick.length ? pick : sweep ? Object.keys(VARIANTS) : ["leadbest"];
for (const name of arms.filter(a => a !== "oracle")) {
  if (!VARIANTS[name]) throw new Error(`no such arm: ${name} (have: ${Object.keys(VARIANTS).join(", ")}, oracle)`);
  const tally = tallyOf();
  for (const [set, query, , , , needleRe, expectRe] of CASES) {
    const t = Date.now();
    const ans = await VARIANTS[name](cache[query] ?? "", query);
    const verdict = verdictOf(ans, "", needleRe, expectRe);
    tally.add(set, verdict, Date.now() - t);
    if (show) {
      console.log(`${verdict.padEnd(8)} ${set} ${query.slice(0, 44).padEnd(46)} ${ans.replace(/\s+/g, " ").slice(0, 76)}`);
    }
  }
  tally.report(name.padEnd(8));
  if (evDirty) await Bun.write(EV, JSON.stringify(evCache));
}

// The ceiling for extraction, not a shippable arm: grade *every* sentence and every adjacent
// pair, and count the case correct if any of them passes. No selector can beat this, so it
// separates "our selection is weak" from "the answer is not a sentence in the text at all".
if (arms.includes("oracle")) {
  const tally = tallyOf();
  for (const [set, query, , , , needleRe, expectRe] of CASES) {
    const t = Date.now();
    const sents = sentences(cache[query] ?? "");
    const spans = [...sents, ...sents.slice(0, -1).map((s, i) => `${s} ${sents[i + 1]}`)];
    const hit = spans.find(s => verdictOf(s, "", needleRe, expectRe) === "ok");
    tally.add(set, hit ? "ok" : "WRONG", Date.now() - t);
    if (show && !hit) console.log(`MISS     ${set} ${query.slice(0, 46)}`);
  }
  tally.report("oracle  ");
}
console.log(`\n(contexts: ${Object.keys(cache).length} cached, ${fetched}s fetching this run)`);
