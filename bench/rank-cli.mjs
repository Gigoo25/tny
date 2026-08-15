#!/usr/bin/env bun
// Does the shipping binary retrieve the right article? The harness measures JS
// re-implementations of the scorer; this measures `tny` itself, which is the only thing a
// user runs. Verbose mode prints `book · title · §sections` to stderr, so the CLI's own
// choice is observable without a second code path.
import { CASES as INST } from "./fixture-instructional.mjs";
import { TNY_ENV } from "./env.mjs";
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

// 13 corpora x 58 queries is 754 kiwix searches; serial that is 15 minutes and the loop
// stops being usable. Four at a time on a 4-core box, which is what tny itself would face
// under any real concurrent use.
const one = async ([set, query, , wantBook, titleRe, needleRe]) => {
  const p = Bun.spawn(["./target/release/tny", "--rank", query], { env: TNY_ENV, stdout: "pipe", stderr: "pipe" });
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
  // Rank-1 is not the product's constraint: tny now sends the top 3 articles (F58), so the
  // ceiling that matters is whether the answer is in any of them.
  const bodies = [];
  for (const [b, , pth] of rows.slice(0, 3)) bodies.push(pth ? await text(b, pth) : "");
  const okAns = needleRe ? needleRe.test(bodies[0] ?? "") : false;
  const okAns3 = needleRe ? bodies.some(x => needleRe.test(x)) : false;
  const where = inList >= 0 ? `rank ${inList + 1}` : rows.length ? "ABSENT" : "no candidates";
  return { set, query, okArt, okBk, okAns, okAns3, where, title, book };
};

const results = [];
for (let i = 0; i < cases.length; i += 4) {
  results.push(...(await Promise.all(cases.slice(i, i + 4).map(one))));
}
const n = results.length;
const sum = k => results.filter(r => r[k]).length;
console.log(`\ncli article@1 ${sum("okArt")}/${n} | in shortlist ${results.filter(r => r.where.startsWith("rank") || r.okArt).length}/${n} | book@1 ${sum("okBk")}/${n}`);
console.log(`answer present: top-1 article ${sum("okAns")}/${n} | top-3 articles ${sum("okAns3")}/${n}`);
for (const r of results.filter(r => !r.okArt)) {
  console.log(`  MISS ${r.set} ${r.query.slice(0, 42).padEnd(44)} ${r.where.padEnd(8)} ${r.okAns3 ? "ANSWER OK" : "no answer "} -> ${(r.title || "(nothing)").slice(0, 32)}`);
}
