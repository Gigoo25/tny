#!/usr/bin/env bun
// F28 arm comparison under sampling: does conversation history beat a stateless
// concatenated turn? One run of 6 cases is noise (measured 4/6, 5/6 and 6/6 for the
// same arm), so this repeats each case and reports both arms plus latency.
import { buildFollowups, ask, ungrounded } from "./harness.mjs";

const SYS = "Answer the question using the reference material. Be concise: at most two sentences plus the exact command if one applies.";
const REPS = +(process.argv[2] ?? 4);

const fu = await buildFollowups();
const first = {};
for (const c of fu) {
  first[c.q2] = (await ask([
    { role: "system", content: SYS },
    { role: "user", content: `Reference:\n${c.ref1}\n\nQuestion: ${c.q1}` },
  ])).content;
}

const sc = { hist: 0, plain: 0 }, ms = { hist: 0, plain: 0 }, tok = { hist: 0, plain: 0 };
const fail = { hist: [], plain: [] };
let n = 0, falseReject = 0;
for (let r = 0; r < REPS; r++) {
  for (const c of fu) {
    n++;
    const t0 = Date.now();
    const H = await ask([
      { role: "system", content: SYS },
      { role: "user", content: `Reference:\n${c.ref1}\n\nQuestion: ${c.q1}` },
      { role: "assistant", content: first[c.q2] },
      { role: "user", content: `Reference:\n${c.ref2}\n\nQuestion: ${c.q2}` },
    ], { temperature: 0.3 });
    ms.hist += Date.now() - t0;
    tok.hist += H.usage?.prompt_tokens ?? 0;
    const t1 = Date.now();
    const P = await ask([
      { role: "system", content: SYS },
      { role: "user", content: `Reference:\n${c.ref2}\n\nQuestion: ${c.q1} — specifically: ${c.q2}` },
    ], { temperature: 0.3 });
    ms.plain += Date.now() - t1;
    tok.plain += P.usage?.prompt_tokens ?? 0;
    const okH = c.needle.test(H.content), okP = c.needle.test(P.content);
    sc.hist += okH ? 1 : 0;
    sc.plain += okP ? 1 : 0;
    if (!okH) fail.hist.push(c.q2);
    if (!okP) fail.plain.push(c.q2);
    // F27 must never reject a correct follow-up; reference is both turns' material
    for (const [arm, ok, a] of [["hist", okH, H.content], ["plain", okP, P.content]]) {
      const why = ok && ungrounded(a, `${c.ref1}\n${c.ref2}`, c.q2);
      if (why) {
        falseReject++;
        console.log(`\nFALSE REJECT [${arm}] [${why}] ${c.q2}\n  ${a.slice(0, 220).replace(/\n/g, " ")}`);
      }
    }
  }
  console.log(`after ${r + 1} reps (${n} samples): hist ${sc.hist}/${n} | plain ${sc.plain}/${n}`);
}
const pct = (a, b) => `${a}/${b} (${((100 * a) / b).toFixed(0)}%)`;
console.log(`\n=== ${n} samples per arm ===`);
console.log(`history:   ${pct(sc.hist, n)}  ${(ms.hist / n / 1000).toFixed(1)}s  ${Math.round(tok.hist / n)} prompt tok`);
console.log(`stateless: ${pct(sc.plain, n)}  ${(ms.plain / n / 1000).toFixed(1)}s  ${Math.round(tok.plain / n)} prompt tok`);
console.log(`F27 false rejects on correct answers: ${falseReject}`);
const tally = a => [...a.reduce((m, q) => m.set(q, (m.get(q) ?? 0) + 1), new Map())].map(([q, k]) => `${k}x ${q}`).join(" | ");
console.log(`history failures:   ${tally(fail.hist) || "none"}`);
console.log(`stateless failures: ${tally(fail.plain) || "none"}`);
