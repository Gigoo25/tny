#!/usr/bin/env bun
// Does the shipping binary retrieve the right article? The harness measures JS
// re-implementations of the scorer; this measures `tny` itself, which is the only thing a
// user runs. Verbose mode prints `book · title · §sections` to stderr, so the CLI's own
// choice is observable without a second code path.
import { CASES as INST } from "./fixture-instructional.mjs";
import { CASES as QA } from "./fixture-qa.mjs";
import { CASES as GEN } from "./fixture-general.mjs";
import { CASES as AMB } from "./fixture-ambiguous.mjs";

// All four fixtures share one tuple shape: [query, intent, book, titleRe, needleRe].
const ALL = [
  ...INST.map(c => ["inst", ...c]),
  ...QA.map(c => ["qa", ...c]),
  ...GEN.map(c => ["gen", ...c]),
  ...AMB.map(c => ["amb", ...c]),
];

const only = process.argv[2];
const cases = only ? ALL.filter(c => c[0] === only) : ALL;
let art = 0, bk = 0, n = 0;
const misses = [];
for (const [set, query, , wantBook, titleRe] of cases) {
  const p = Bun.spawn(["./target/release/tny", "--rank", query], { stdout: "pipe", stderr: "pipe" });
  const out = await new Response(p.stdout).text();
  await new Response(p.stderr).text();
  await p.exited;
  const [book = "", title = ""] = out.trim().split("\t");
  const okArt = titleRe.test(title);
  const okBk = book.includes(wantBook.replace(/_nopic.*|_all.*|_maxi.*/, ""));
  n++; art += okArt ? 1 : 0; bk += okBk ? 1 : 0;
  if (!okArt) misses.push(`${set} ${query.slice(0, 46).padEnd(48)} -> ${title.slice(0, 44) || "(nothing)"}`);
}
console.log(`\ncli right article@1 ${art}/${n} | right book@1 ${bk}/${n}`);
for (const m of misses) console.log("  MISS", m);
