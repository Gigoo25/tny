#!/usr/bin/env bun
// Is the fact in the context the model was handed?
//
// This is the measurement that separates the two failure classes, and nothing else in bench/
// answers it: `rank-cli.mjs` grades the *article* at rank 1, `answer-cli.mjs` grades the
// *answer*. Between them sits section selection, which is where F63 hid a wrong answer behind
// correct retrieval. `tny --context` prints the exact slice without loading the model, so a
// full pass costs ~20 s instead of ~20 minutes.
//
// A case counts as IN-CTX if either the article-prose needle or the authored answer regex
// matches the slice: the needle can be phrased for prose the slice cut, and vice versa.
import { EXPECT } from "./expect.mjs";
import { TNY_ENV } from "./env.mjs";

const SETS = [["instructional", "instructional"], ["qa", "qa"], ["general", "general"], ["ambiguous", null]];
const only = process.argv.slice(2).find(a => !a.startsWith("-"));
const failing = process.argv.includes("--failing");
const cache = await Bun.file("bench/.answers.json").exists()
  ? JSON.parse(await Bun.file("bench/.answers.json").text()) : {};

let inCtx = 0, absent = 0;
for (const [f, k] of SETS) {
  if (only && !f.startsWith(only)) continue;
  const { CASES } = await import(`./fixture-${f}.mjs`);
  for (let i = 0; i < CASES.length; i++) {
    const c = CASES[i];
    const exp = c[5] ?? (k ? EXPECT[k][i] : null);
    // `--failing` narrows to the cases that got the answer wrong, which is the working set
    // while a section-selection change is under test.
    if (failing && exp && exp.test((cache[c[0]]?.ans ?? "").replace(/\s+/g, " "))) continue;
    const p = Bun.spawn(["./target/release/tny", "--context", c[0]], { env: TNY_ENV, stdout: "pipe", stderr: "pipe" });
    const ctx = await new Response(p.stdout).text();
    await p.exited;
    const hit = c[4].test(ctx) || (exp && exp.test(ctx));
    hit ? inCtx++ : absent++;
    const heads = (ctx.match(/^## .*/gm) ?? []).join(" | ").replace(/## /g, "");
    console.log(`${hit ? "IN-CTX " : "ABSENT "} ${c[0].slice(0, 44).padEnd(46)} ${String(ctx.length).padStart(5)}ch  ${heads.slice(0, 62)}`);
  }
}
console.log(`\ncontext has the fact: ${inCtx}  absent: ${absent}  (${inCtx}/${inCtx + absent})`);
