#!/usr/bin/env bun
// Offline scorer sweep.
//
// Each live measurement costs ~100 s (58 queries x 13 books of kiwix search), which is too
// slow to search a weight space — and it was slow enough that a bad idea (IDF weighting,
// 23/58 against 32/58) cost 15 minutes to disprove. So dump the candidates ONCE, cache every
// candidate article's text ONCE, then score variants in milliseconds.
//
// Two metrics, both from the fixtures' own verified data:
//   article@1  the pinned article ranks first
//   answer@3   the verified answer text appears in one of the top 3 articles — this is what
//              tny actually sends the model (F58), so it is the number that predicts quality
import { CASES as INST } from "./fixture-instructional.mjs";
import { TNY_ENV } from "./env.mjs";
import { CASES as QA } from "./fixture-qa.mjs";
import { CASES as GEN } from "./fixture-general.mjs";
import { CASES as AMB } from "./fixture-ambiguous.mjs";

const KIWIX = process.env.TNY_KIWIX ?? "http://127.0.0.1:8082";
const CACHE = "bench/.sweep-cache.json";
const CASES = [...INST.map(c => ["inst", ...c]), ...QA.map(c => ["qa", ...c]),
               ...GEN.map(c => ["gen", ...c]), ...AMB.map(c => ["amb", ...c])];

// ---------------------------------------------------------------- cache build
const sh = async (args) => {
  const p = Bun.spawn(args, { env: TNY_ENV, stdout: "pipe", stderr: "pipe" });
  const out = await new Response(p.stdout).text();
  await new Response(p.stderr).text();
  await p.exited;
  return out;
};

async function build() {
  const cache = { cands: {}, text: {} };
  for (let i = 0; i < CASES.length; i += 4) {
    await Promise.all(CASES.slice(i, i + 4).map(async ([, q]) => {
      const out = await sh(["./target/release/tny", "--dump", q]);
      try { cache.cands[q] = JSON.parse(out); } catch { cache.cands[q] = []; }
    }));
    process.stderr.write(`\rdumped ${Math.min(i + 4, CASES.length)}/${CASES.length}`);
  }
  // Article text for every candidate any variant could plausibly rank top-3. Fetch all of
  // them: 58 x ~40 is 2,300 fetches at ~20 ms, once, and then no variant needs the network.
  const want = new Set();
  for (const rows of Object.values(cache.cands)) for (const c of rows) want.add(`${c.book}\u0000${c.path}`);
  const keys = [...want];
  for (let i = 0; i < keys.length; i += 12) {
    await Promise.all(keys.slice(i, i + 12).map(async k => {
      const [book, path] = k.split("\u0000");
      const r = await fetch(`${KIWIX}/content/${book}/${path}`).catch(() => null);
      cache.text[k] = r?.ok
        ? (await r.text()).replace(/<[^>]+>/g, " ").replace(/&[a-z]+;/g, " ").replace(/\s+/g, " ")
        : "";
    }));
    process.stderr.write(`\rfetched ${Math.min(i + 12, keys.length)}/${keys.length}`);
  }
  await Bun.write(CACHE, JSON.stringify(cache));
  process.stderr.write("\n");
  return cache;
}

const cache = await Bun.file(CACHE).exists() && !process.argv.includes("--rebuild")
  ? JSON.parse(await Bun.file(CACHE).text())
  : await build();

// ---------------------------------------------------------------- primitives
// Ported from src/retrieve.rs so a winning variant transfers verbatim.
const STOP = /^(how|do|i|a|an|the|is|are|does|to|in|of|for|and|or|my|me|what|why|when|which|that|this|it|its|on|with|from|be|can|you|your)$/;
const terms = q => (q.toLowerCase().match(/[a-z0-9_.:+-]{2,}/g) ?? []).filter(w => !STOP.test(w));
const denoise = s => s.replace(/\[\s*\d+\s*\]/g, " ").replace(/\s+/g, " ");
const SE_INDEX = /^(highest voted|newest|active|unanswered|top|recent)\b|\bquestions$/i;
const TITLE_STOP = /^(the|a|an|of|in|to|and|or|for)$/;

const titleWords = t => (t.toLowerCase().match(/[a-z0-9_.:-]{2,}/g) ?? []).filter(w => !TITLE_STOP.test(w));

// Document frequency over the retrieved set: measured as a *loss* when used to weight the
// whole score (F-IDF, 23/58), kept available because gating is a different question.
const dfWeights = (t, cands) => {
  const hay = cands.map(c => `${c.title} ${denoise(c.snip)}`.toLowerCase());
  return t.map(w => Math.log((hay.length + 1) / (hay.filter(h => h.includes(w)).length + 1)));
};

// ---------------------------------------------------------------- variants
// Every variant returns a score; higher ranks first.
const V = {
  // The shipping scorer (32/58 article@1, 43/58 answer@3 measured live).
  ship: (c, t, q) => base(c, t) + cover(c, t) * 3 + prior(c, t, q) - c.rank / 100,
  // The grid winner: title hits at *2 not *3, and Xapian's within-book rank at /5 not /100.
  // Sits on a plateau (rank 4-5, cover 3-4 all score the same) and gains in every fixture.
  tuned: (c, t, q) => baseW(c, t, 2) + cover(c, t) * 3 + prior(c, t, q) - c.rank / 5,
  // F62: the two intent defects, measured apart and together.
  strictWhy: (c, t, q) => baseW(c, t, 2) + cover(c, t) * 3 + priorWith(DIAG_STRICT, false)(c, t, q) - c.rank / 5,
  concept: (c, t, q) => baseW(c, t, 2) + cover(c, t) * 3 + priorWith(DIAG_SHIP, true)(c, t, q) - c.rank / 5,
  both: (c, t, q) => baseW(c, t, 2) + cover(c, t) * 3 + priorWith(DIAG_STRICT, true)(c, t, q) - c.rank / 5,
  // and with the title weight the snippet grid liked
  bothT1: (c, t, q) => baseW(c, t, 1) + cover(c, t) * 3 + priorWith(DIAG_STRICT, true)(c, t, q) - c.rank / 5,
  // Is the entity bonus earning its keep at all?
  noCover: (c, t, q) => base(c, t) + prior(c, t, q) - c.rank / 100,
  // Trust Xapian's within-book confidence more than a hundredth of a point.
  rank10: (c, t, q) => base(c, t) + cover(c, t) * 3 + prior(c, t, q) - c.rank / 10,
  rank3: (c, t, q) => base(c, t) + cover(c, t) * 3 + prior(c, t, q) - c.rank / 3,
  // Only the book's top hit competes; deeper hits are a fallback.
  rankHard: (c, t, q) => base(c, t) + cover(c, t) * 3 + prior(c, t, q) - (c.rank > 0 ? 1 : 0),
  // The title as a phrase inside the question ("docker tag" in "docker image tag for a
  // registry") is a much stronger claim than its words appearing scattered.
  phrase: (c, t, q) => base(c, t) + cover(c, t) * 3 + prior(c, t, q) + phraseHit(c, q) * 2 - c.rank / 100,
  // Gate the entity bonus on the query's heaviest term (rejected live at 23/58 when it also
  // reweighted the whole score — retested here in isolation).
  gate: (c, t, q, w) => base(c, t) + coverGated(c, t, w) * 3 + prior(c, t, q) - c.rank / 100,
  // Both of the above.
  phraseGate: (c, t, q, w) => base(c, t) + coverGated(c, t, w) * 3 + prior(c, t, q) + phraseHit(c, q) * 2 - c.rank / 100,
  // A long Q&A title matches many terms by sheer length; normalise harder than sqrt.
  linNorm: (c, t, q) => baseLin(c, t) + cover(c, t) * 3 + prior(c, t, q) - c.rank / 100,
  linPhrase: (c, t, q) => baseLin(c, t) + cover(c, t) * 3 + prior(c, t, q) + phraseHit(c, q) * 2 - c.rank / 100,
};

// title+snippet term overlap, sqrt-normalised by title length (the shipping form)
const base = (c, t) => baseW(c, t, 3);
function baseW(c, t, tiWeight) {
  const title = c.title.toLowerCase();
  const tw = (title.match(/[a-z0-9_.:+-]{2,}/g) ?? []).length || 1;
  const th = t.filter(w => title.includes(w)).length;
  const body = denoise(c.snip).toLowerCase().slice(0, 400);
  const bh = t.filter(w => body.includes(w)).length;
  return (th * tiWeight) / Math.sqrt(tw) + bh / Math.max(t.length, 1);
}

// the same, normalised by title length outright
function baseLin(c, t) {
  const title = c.title.toLowerCase();
  const tw = (title.match(/[a-z0-9_.:+-]{2,}/g) ?? []).length || 1;
  const th = t.filter(w => title.includes(w)).length;
  const body = denoise(c.snip).toLowerCase().slice(0, 400);
  const bh = t.filter(w => body.includes(w)).length;
  return (th * 6) / tw + bh / Math.max(t.length, 1);
}

// Entity coverage: is the title an entity the query names, scaled by the share of the
// question it accounts for (F49 + F54). Ported from retrieve.rs::title_covered.
function cover(c, t) {
  const w = titleWords(c.title);
  if (!w.length || w.length > 5) return 0;
  const joined = t.join(" ");
  if (!w.every(x => joined.includes(x))) return 0;
  return w.length / Math.max(t.length, 1);
}

// Does the whole title appear in the question as a phrase?
function phraseHit(c, q) {
  const title = c.title.toLowerCase().replace(/\s*\([^)]*\)\s*$/, "").trim();
  if (title.length < 4 || titleWords(title).length > 4) return 0;
  return q.toLowerCase().includes(title) ? 1 : 0;
}

function coverGated(c, t, w) {
  const s = cover(c, t);
  if (!s) return 0;
  let hi = 0;
  for (let i = 1; i < w.length; i++) if (w[i] > w[hi]) hi = i;
  return c.title.toLowerCase().includes(t[hi] ?? "") ? s : 0;
}

// F62: a *diagnostic* "why" carries an error token — "why is ssh connection refused". A bare
// "why is the sky blue" is curiosity, and classifying it as diagnosis boosts Q&A threads over
// the encyclopedia article that answers it. `DIAG_STRICT` drops the bare-why alternative;
// the rest of the vocabulary still catches every real diagnosis.
const ERRW = String.raw`error|errors|fail|failed|failing|refused|denied|cannot|can't|won't|does ?n't|broken|no such|not found|timed? ?out|exit code|permission`;
const DIAG_SHIP = new RegExp(String.raw`\b(${ERRW}|why (is|are|does|do|did|would|am|can't))\b`, "i");
const DIAG_STRICT = new RegExp(String.raw`\b(${ERRW})\b`, "i");
// A concept question wants the article about the thing, not a thread discussing it.
const CONCEPT = /^(what|who|when|where) (is|are|was|were|does|do|did)\b|^what.*\bmade of\b/i;
const HOWTO = /^(how (do|can|would) i|how to|what command)\b|^(create|set|mount|encrypt|generate|check|list|install|enable|configure|disable|remove|start|stop|make)\b/i;
const kindOf = c => c.kind === "Qa" ? "qa" : c.kind === "Index" ? "index" : "article";

// intent priors, parameterised so the two defects can be measured apart
const priorWith = (diag, concept) => (c, t, q = "") => {
  const k = kindOf(c);
  if (k !== "qa") return 0;
  if (diag.test(q)) return 1;
  if (HOWTO.test(q.trim())) return -2;
  if (concept && CONCEPT.test(q.trim())) return -2;
  return 0;
};
const prior = priorWith(DIAG_SHIP, false);

// ---------------------------------------------------------------- scoring
function evaluate(score) {
  let art = 0, ans1 = 0, ans3 = 0, ans5 = 0, any = 0, n = 0;
  const misses = [];
  const per = {};
  for (const [set, q, , wantBook, titleRe, needleRe] of CASES) {
    const cands = cache.cands[q] ?? [];
    n++;
    per[set] ??= [0, 0];
    per[set][1]++;
    if (!cands.length) { misses.push([set, q, "none", ""]); continue; }
    const t = terms(q);
    const w = dfWeights(t, cands);
    const ranked = cands
      .map(c => ({ c, s: score(c, t, q, w, cands) }))
      .sort((a, b) => b.s - a.s)
      .map(o => o.c);
    if (titleRe.test(ranked[0].title)) art++;
    const body = i => cache.text[`${ranked[i]?.book}\u0000${ranked[i]?.path}`] ?? "";
    const hit = i => needleRe.test(body(i));
    // Where the answer sits decides which lever applies: a needle at rank 3-7 is a scoring
    // problem, one nowhere in the list is a candidate-generation problem.
    const at = ranked.findIndex((_, i) => hit(i));
    if (at === 0) ans1++;
    if (at >= 0 && at < 3) { ans3++; per[set][0]++; }
    if (at >= 0 && at < 5) ans5++;
    if (at >= 0) any++;
    if (at < 0 || at > 2) misses.push([set, q, at < 0 ? "ABSENT" : `rank${at}`, ranked[0].title]);
  }
  const shape = Object.entries(per).map(([k, [a, b]]) => `${k} ${a}/${b}`).join(" ");
  return { art, ans1, ans3, ans5, any, n, misses, per: shape };
}
const only = process.argv.find(a => a.startsWith("--only="))?.slice(7);
for (const [name, fn] of Object.entries(V)) {
  if (only && name !== only) continue;
  const r = evaluate(fn);
  console.log(
    `${name.padEnd(12)} article@1 ${String(r.art).padStart(2)}/${r.n}  ` +
    `answer@1 ${String(r.ans1).padStart(2)}  @3 ${String(r.ans3).padStart(2)}  ` +
    `@5 ${String(r.ans5).padStart(2)}  ceiling ${String(r.any).padStart(2)}/${r.n}`
  );
  if (process.argv.includes("-v")) {
    for (const [set, q, where, got] of r.misses) {
      console.log(`   ${set} ${q.slice(0, 44).padEnd(46)} ${String(where).padEnd(7)} got ${got.slice(0, 34)}`);
    }
  }
}

// ---------------------------------------------------------------- grid
// 58 cases is small enough that a grid search can overfit, so this prints the whole
// neighbourhood: a weight worth shipping sits on a plateau, not a spike, and its gain is
// spread across fixtures rather than concentrated in one.
if (process.argv.includes("--grid")) {
  const rows = [];
  for (const rk of [100, 10, 5, 4, 3, 2, 1.5, 1]) {
    for (const cv of [0, 1, 2, 3, 4]) {
      for (const ti of [2, 3, 4]) {
        const fn = (c, t, q) => {
          const title = c.title.toLowerCase();
          const tw = (title.match(/[a-z0-9_.:+-]{2,}/g) ?? []).length || 1;
          const th = t.filter(w => title.includes(w)).length;
          const body = denoise(c.snip).toLowerCase().slice(0, 400);
          const bh = t.filter(w => body.includes(w)).length;
          return (th * ti) / Math.sqrt(tw) + bh / Math.max(t.length, 1)
            + cover(c, t) * cv + prior(c, t, q) - c.rank / rk;
        };
        const r = evaluate(fn);
        rows.push({ rk, cv, ti, ...r });
      }
    }
  }
  rows.sort((a, b) => b.ans3 - a.ans3 || b.art - a.art);
  for (const r of rows.slice(0, 14)) {
    console.log(`rank/${String(r.rk).padEnd(4)} cover*${r.cv} title*${r.ti}  article@1 ${String(r.art).padStart(2)}  answer@1 ${String(r.ans1).padStart(2)}  @3 ${String(r.ans3).padStart(2)}  @5 ${String(r.ans5).padStart(2)}  ${r.per}`);
  }
}

// ---------------------------------------------------------------- snippet grid
// F62: every remaining failure is a synonym gap — `Sodium chloride` for "table salt",
// `Rayleigh scattering` for "why is the sky blue", `Mollusca` for "shell in biology". The
// title shares no term with the question, so no title scorer can reach them. But kiwix's
// snippet is the passage that matched, and the synonym evidence lives there. This sweeps how
// much the snippet is worth against the title.
if (process.argv.includes("--snip")) {
  const rows = [];
  for (const ti of [0, 1, 2, 3]) {
    for (const bo of [1, 2, 3, 4, 6, 8]) {
      for (const cv of [0, 1, 3]) {
        const fn = (c, t, q) => {
          const title = c.title.toLowerCase();
          const tw = (title.match(/[a-z0-9_.:+-]{2,}/g) ?? []).length || 1;
          const th = t.filter(w => title.includes(w)).length;
          const body = denoise(c.snip).toLowerCase().slice(0, 400);
          const bh = t.filter(w => body.includes(w)).length;
          return (th * ti) / Math.sqrt(tw) + (bh * bo) / Math.max(t.length, 1)
            + cover(c, t) * cv + prior(c, t, q) - c.rank / 5;
        };
        rows.push({ ti, bo, cv, ...evaluate(fn) });
      }
    }
  }
  rows.sort((a, b) => b.ans3 - a.ans3 || b.art - a.art);
  for (const r of rows.slice(0, 12)) {
    console.log(`title*${r.ti} snip*${r.bo} cover*${r.cv}  article@1 ${String(r.art).padStart(2)}  answer@1 ${String(r.ans1).padStart(2)}  @3 ${String(r.ans3).padStart(2)}  @5 ${String(r.ans5).padStart(2)}  ${r.per}`);
  }
}
