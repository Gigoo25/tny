#!/usr/bin/env bun
// End-to-end answer correctness, over every verified case.
//
// Everything else in bench/ measures a proxy: did retrieval put the answer in the context.
// This measures the product — did `tny "question"` print a correct answer.
//
// The grader lives in grade.mjs, shared with the no-model arm (bench/extract-cli.mjs), so the
// two are directly comparable: same cases, same regexes, same verdicts, different answerer.
import { CASES, verdictOf, tallyOf } from "./grade.mjs";
import { TNY_ENV } from "./env.mjs";

const only = process.argv.slice(2).find(a => !a.startsWith("-"));
// Generation costs ~20 s per case, so a grader change must never re-pay it: answers are cached
// verbatim and `--regrade` scores the cache offline, the same split that made the retrieval
// sweep usable (F56).
// One cache per model: a variant run must never overwrite the baseline it is compared against.
const CACHE = process.env.TNY_ANSWERS ?? "bench/.answers.json";
const regrade = process.argv.includes("--regrade");
const cache = await Bun.file(CACHE).exists() ? JSON.parse(await Bun.file(CACHE).text()) : {};
const tally = tallyOf();

// F107: `--resume` reuses answers already on disk and only pays for the missing ones. A 4B
// pass over 18 cases is 55 minutes; without this, one timeout means paying all of it again.
const resume = process.argv.includes("--resume");

async function answerOf(query) {
  if (regrade) return cache[query] ?? { ans: "", err: "" };
  if (resume && (cache[query]?.ans ?? "").trim() !== "") return cache[query];
  // --fresh: F85 caches answers on disk, and a benchmark that reads its own cache measures
  // the cache. Every case here must pay the model.
  const p = Bun.spawn(["./target/release/tny", "--fresh", query], { env: TNY_ENV, stdout: "pipe", stderr: "pipe" });
  const ans = (await new Response(p.stdout).text()).trim();
  const err = await new Response(p.stderr).text();
  await p.exited;
  cache[query] = { ans, err };
  // F106: write through, not at the end. A 4B answer costs 4-5 minutes here, and a run
  // killed at case 13 of 18 used to leave nothing on disk — an hour of measurement thrown
  // away by a timeout. Rewriting the whole file per case costs milliseconds against minutes.
  await Bun.write(CACHE, JSON.stringify(cache, null, 1));
  return cache[query];
}

// [set, query, intent, book, titleRe, needleRe, expectRe] — `set` is prepended above, so the
// answer-grading regexes are at 5 and 6, not 4 and 5.
for (const [set, query, , , , needleRe, expectRe] of CASES) {
  if (only && set !== only) continue;
  const t = Date.now();
  // tny prints the answer on stdout and its diagnostics on stderr; the grader needs both,
  // since a rejection note is the difference between a refusal and a wrong answer.
  const { ans, err } = await answerOf(query);
  const verdict = verdictOf(ans, err, needleRe, expectRe);
  tally.add(set, verdict, Date.now() - t);
  console.log(`${verdict.padEnd(8)} ${((Date.now() - t) / 1000).toFixed(0).padStart(3)}s ${set} ${query.slice(0, 46).padEnd(48)} ${ans.replace(/\s+/g, " ").slice(0, 70)}`);
}


tally.report("model   ");
