#!/usr/bin/env bun
// End-to-end answer correctness, over every verified case.
//
// Everything else in bench/ measures a proxy: did retrieval put the answer in the context.
// This measures the product — did `tny "question"` print a correct answer. The grader is the
// fixture's own `needleRe`, the fact regex that was verified to appear in the source article
// before the case was allowed in, applied to the model's output instead of the article's text.
//
// Three outcomes, and the difference between the last two is the whole point of the grounding
// rules (F27/F44/F45): a refusal is safe, a confident wrong answer is not.
//   correct   the fact is in the answer
//   refused   tny declined ("not found" / rejected)
//   wrong     an answer that does not carry the fact
import { CASES as INST } from "./fixture-instructional.mjs";
import { CASES as QA } from "./fixture-qa.mjs";
import { CASES as GEN } from "./fixture-general.mjs";
import { CASES as AMB } from "./fixture-ambiguous.mjs";
import { EXPECT } from "./expect.mjs";

// A case carries [query, intent, book, titleRe, needleRe, expectRe?]. `fixture-ambiguous.mjs`
// authors its `expectRe` inline; the other three take theirs from `expect.mjs` by position, so
// the retrieval fixtures stay about articles and answer grading lives in one file.
const CASES = [["inst", INST, EXPECT.instructional], ["qa", QA, EXPECT.qa],
               ["gen", GEN, EXPECT.general], ["amb", AMB, []]]
  .flatMap(([set, cs, exp]) => cs.map((c, i) => [set, ...c.slice(0, 5), c[5] ?? exp[i]]));


// `needleRe` matches the *article's* prose, so it cannot be applied to an answer directly:
// the article says "115 known moons", a correct answer says "Jupiter has 115 moons". What
// transfers is the fact itself, and it is mechanically extractable — no hand-written expected
// answers, which would only encode my own idea of the right wording.
//
// The largest number in the needle is the fact whenever there is one (1989, 5895, 42.195,
// 104): a wrong answer changes it, which is exactly the failure to catch — this rule flags
// "the Berlin Wall fell on November 9, 2009" and "Jupiter has 95 known moons" against
// articles that say 1989 and 115, both of which read as confidently grounded. With no number,
// the needle's rarest words carry it (`NaCl`, `Leonardo`, `Calvin`, `scatter`).
const COMMON = /^(the|and|for|are|was|were|with|that|this|from|have|has|been|into|than|then|when|what|which|about|their|there|these|those|being|other|also|more|most|some|such|only|over|after|before|between|called|named|later|primarily|known|used|using|use|its|his|her)$/i;

// An escaped `\.` in a needle is a decimal point ("42\.195" is one number); a bare `.` is
// regex-any, so "100.120 compressions" is really the range 100-120 and must be read as two.
function factOf(needleRe) {
  const raw = String(needleRe).replace(/^\//, "").replace(/\/[a-z]*$/, "");
  const src = raw.replace(/\\\./g, "\u0001").replace(/\\/g, "");
  const nums = (src.match(/\d[\d\u0001,]*\d|\d+/g) ?? []).map(s => s.replace(/,/g, "").replace(/\u0001/g, "."));
  const words = (src.match(/[A-Za-z][A-Za-z-]{2,}/g) ?? []).filter(w => !COMMON.test(w));
  // The largest value is the fact: a wrong answer changes it, while small incidentals ("9"
  // in "fell on 9 November 1989") appear in almost any answer by chance.
  const num = nums.length ? nums.reduce((a, b) => (parseFloat(b) > parseFloat(a) ? b : a)) : null;
  return { num, words };
}

// The number must stand alone: "195" inside "42.195" is not a match for 195, but "42.195" is.
const hasNum = (ans, n) => new RegExp(String.raw`(?<![\d.,])${n.replace(".", "\\.")}(?![\d])`).test(ans.replace(/,/g, ""));
const hasWord = (ans, words) => words.some(w => new RegExp(String.raw`\b${w}`, "i").test(ans));
const only = process.argv.slice(2).find(a => !a.startsWith("-"));
// Generation costs ~20 s per case, so a grader change must never re-pay it: answers are cached
// verbatim and `--regrade` scores the cache offline, the same split that made the retrieval
// sweep usable (F56).
const CACHE = "bench/.answers.json";
const regrade = process.argv.includes("--regrade");
const cache = await Bun.file(CACHE).exists() ? JSON.parse(await Bun.file(CACHE).text()) : {};
const tally = {};
const t0 = Date.now();

async function answerOf(query) {
  if (regrade) return cache[query] ?? { ans: "", err: "" };
  const p = Bun.spawn(["./target/release/tny", query], { stdout: "pipe", stderr: "pipe" });
  const ans = (await new Response(p.stdout).text()).trim();
  const err = await new Response(p.stderr).text();
  await p.exited;
  cache[query] = { ans, err };
  return cache[query];
}

// [set, query, intent, book, titleRe, needleRe, expectRe] — `set` is prepended above, so the
// answer-grading regexes are at 5 and 6, not 4 and 5.
for (const [set, query, , , , needleRe, expectRe] of CASES) {
  if (only && set !== only) continue;
  const t = Date.now();
  const { ans, err } = await answerOf(query);
  // tny prints the answer on stdout and its diagnostics on stderr; a refusal is either the
  // "not found" sentinel or an empty answer with a rejection note.
  const refused = /^not found$/im.test(ans) || ans === "" || /rejected/.test(err);
  // An authored `expectRe` grades the answer directly: it was written against the source
  // article to accept any paraphrase of the fact and reject the plausible wrong answer, which
  // the needle-derived rule cannot do. Cases without one fall back to that rule.
  const { num, words } = factOf(needleRe);
  const correct = !refused && (expectRe ? expectRe.test(ans) : num === null ? hasWord(ans, words) : hasNum(ans, num));
  const verdict = refused ? "REFUSED" : correct ? "ok" : "WRONG";
  tally[set] ??= { correct: 0, refused: 0, wrong: 0, n: 0, ms: 0 };
  tally[set].n++;
  tally[set].ms += Date.now() - t;
  tally[set][refused ? "refused" : correct ? "correct" : "wrong"]++;
  console.log(`${verdict.padEnd(8)} ${((Date.now() - t) / 1000).toFixed(0).padStart(3)}s ${set} ${query.slice(0, 46).padEnd(48)} ${ans.replace(/\s+/g, " ").slice(0, 70)}`);
}

if (!regrade) await Bun.write(CACHE, JSON.stringify(cache, null, 1));

console.log("");
let C = 0, R = 0, W = 0, N = 0;
for (const [set, r] of Object.entries(tally)) {
  console.log(`${set.padEnd(6)} correct ${r.correct}/${r.n}  refused ${r.refused}  wrong ${r.wrong}  ${(r.ms / r.n / 1000).toFixed(1)}s per answer`);
  C += r.correct; R += r.refused; W += r.wrong; N += r.n;
}
console.log(`TOTAL  correct ${C}/${N} (${((C / N) * 100).toFixed(0)}%)  refused ${R}  wrong ${W}  ${((Date.now() - t0) / 1000 / 60).toFixed(1)} min`);
