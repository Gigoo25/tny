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
// The product question is not "is the title the one I pinned" but "does the article we
// fetch contain the answer" — `Penicillin` names Fleming, and the answering stage reads the
// article, not its title. `needleRe` is the fixture's verified answer text.
const KIWIX = process.env.TNY_KIWIX ?? "http://127.0.0.1:8082";
const text = async (book, path) => {
  const r = await fetch(`${KIWIX}/content/${book}/${path}`).catch(() => null);
  if (!r?.ok) return "";
  return (await r.text()).replace(/<[^>]+>/g, " ").replace(/&[a-z]+;/g, " ").replace(/\s+/g, " ");
};

let art = 0, bk = 0, rec = 0, ans = 0, n = 0;
const misses = [], unanswered = [];
for (const [set, query, , wantBook, titleRe, needleRe] of cases) {
  const p = Bun.spawn(["./target/release/tny", "--rank", query], { stdout: "pipe", stderr: "pipe" });
  const out = await new Response(p.stdout).text();
  await new Response(p.stderr).text();
  await p.exited;
  const rows = out.trim().split("\n").filter(Boolean).map(l => l.split("\t"));
  const [book = "", title = "", path = ""] = rows[0] ?? [];
  const okArt = titleRe.test(title);
  // recall: is the right article anywhere in the shortlist? Separates a scoring miss from
  // a candidate that was never retrieved — the two need opposite fixes.
  const inList = rows.findIndex(r => titleRe.test(r[1] ?? ""));
  const okBk = book.includes(wantBook.replace(/_nopic.*|_all.*|_maxi.*/, ""));
  n++; art += okArt ? 1 : 0; bk += okBk ? 1 : 0; rec += inList >= 0 ? 1 : 0;
  // Does the article we actually fetched carry the answer? This is what the answering
  // stage sees, and it is the only number the user experiences.
  const body = path ? await text(book, path) : "";
  const okAns = needleRe ? needleRe.test(body) : false;
  ans += okAns ? 1 : 0;
  if (!okArt) {
    const where = inList >= 0 ? `rank ${inList + 1}` : rows.length ? "ABSENT" : "no candidates";
    misses.push(`${set} ${query.slice(0, 42).padEnd(44)} ${where.padEnd(8)} ${okAns ? "ANSWER OK" : "no answer "} -> ${(title || "(nothing)").slice(0, 32)}`);
  } else if (!okAns) {
    unanswered.push(`${set} ${query.slice(0, 42).padEnd(44)} right article, needle absent`);
  }
}
console.log(`\ncli article@1 ${art}/${n} | in shortlist ${rec}/${n} | book@1 ${bk}/${n} | ANSWER in fetched article ${ans}/${n}`);
for (const m of misses) console.log("  MISS", m);
