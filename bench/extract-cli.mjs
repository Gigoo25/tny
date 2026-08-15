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

for (const name of sweep ? Object.keys(VARIANTS) : ["idf"]) {
  const tally = tallyOf();
  for (const [set, query, , , , needleRe, expectRe] of CASES) {
    const t = Date.now();
    const ans = VARIANTS[name](cache[query] ?? "", query);
    const verdict = verdictOf(ans, "", needleRe, expectRe);
    tally.add(set, verdict, Date.now() - t);
    if (show && !sweep) {
      console.log(`${verdict.padEnd(8)} ${set} ${query.slice(0, 44).padEnd(46)} ${ans.replace(/\s+/g, " ").slice(0, 76)}`);
    }
  }
  tally.report(name.padEnd(8));
}
console.log(`\n(contexts: ${Object.keys(cache).length} cached, ${fetched}s this run — selection itself is sub-millisecond)`);
