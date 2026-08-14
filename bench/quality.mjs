#!/usr/bin/env bun
// Verbatim answer dump for whatever model is on port 8080, so output *quality* can be
// compared, not just scores. Contexts are identical across models (same ZIM, same
// deterministic selection), so differences are the model alone.
import { buildContexts, ask, ungrounded, ungroundedDetail, ungroundedShape } from "./harness.mjs";

const SYS = "Answer the question using the reference material. Be concise: at most two sentences plus the exact command if one applies.";
const label = process.argv[2] ?? "model";
const ctx = await buildContexts();
let ok = 0, t = 0, rejected = 0;
console.log(`\n######## ${label} ########`);
for (const c of ctx) {
  const t0 = Date.now();
  const { content } = await ask([
    { role: "system", content: SYS },
    { role: "user", content: `Reference:\n${c.text}\n\nQuestion: ${c.q}` },
  ]);
  const ms = Date.now() - t0;
  t += ms;
  const hit = c.needle.test(content);
  ok += hit ? 1 : 0;
  const why = ungrounded(content, c.full ?? c.text, c.q)
    || ungroundedDetail(content, c.full ?? c.text)
    || ungroundedShape(content, c.text, c.q, c.vocab);
  if (hit && why) rejected++;
  console.log(`\nQ: ${c.q}   [${hit ? "correct" : "WRONG"}, ${(ms / 1000).toFixed(1)}s${why ? `, rule: ${why}` : ""}]`);
  console.log(`A: ${content.trim().replace(/\n+/g, " ")}`);
}
console.log(`\n${label}: ${ok}/${ctx.length} correct | ${(t / ctx.length / 1000).toFixed(1)}s per answer | false rejects ${rejected}/${ok}`);
