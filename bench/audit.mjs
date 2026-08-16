#!/usr/bin/env bun
// One-shot audit view: every cached answer beside the fact it must carry and the verdict the
// grader gives it. F65's hand-audit is the only test a grader has, and it was done once by
// eye against a terminal scroll; this prints the same thing reproducibly.
//
// Not a guard — it asserts nothing. `answer-cli.mjs --regrade` is the number.
import { CASES, verdictOf } from "./grade.mjs";

const CACHE = process.env.TNY_ANSWERS ?? "bench/.answers.json";
const cache = JSON.parse(await Bun.file(CACHE).text());
const only = process.argv.slice(2).find(a => !a.startsWith("-"));
const wrap = (s, n) => (s.match(new RegExp(`.{1,${n}}(\\s|$)`, "g")) ?? [s]).map(l => l.trim());
// [set, query, intent, book, titleRe, needleRe, expectRe] — five holes, not four. F64 lost a
// whole session to this exact destructure, and it fails silently: needleRe lands in expectRe
// and every case is graded by the article-prose regex the fact grader was written to replace.
for (const [set, query, , , , needleRe, expectRe] of CASES) {
  if (only && set !== only) continue;
  const { ans = "", err = "" } = cache[query] ?? {};
  const v = verdictOf(ans, err, needleRe, expectRe);
  console.log(`\n${v} · ${set} · ${query}`);
  console.log(`  needle ${needleRe}`);
  console.log(`  expect ${expectRe}`);
  for (const l of wrap(ans.replace(/\s+/g, " ") || "(empty)", 96)) console.log(`  > ${l}`);
  if (err.trim()) for (const l of wrap(err.replace(/\s+/g, " "), 96).slice(0, 2)) console.log(`  ! ${l}`);
}
