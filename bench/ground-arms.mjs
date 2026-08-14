#!/usr/bin/env bun
// F32: what should the grounding check treat as "the reference"?
//
// The windowed sections are what the model sees, but they are only ~1.5 KB. A correct
// answer that cites a neighbouring command ("swapoff /swapfile, then rm -f /swapfile")
// gets rejected because `rm` is not in the slice. That false-reject rate is sampling
// dependent: the same code produced 0, 1 and 3 across three runs, which is exactly the
// pattern that has hidden every earlier grounding defect.
//
// Arm A: reference = the windowed sections sent to the model (current)
// Arm B: reference = the full source article text (F32 candidate)
//
// Both must be scored on BOTH duties: never reject a correct answer, always catch a
// fabricated command. Widening the reference trades one against the other, so the F26
// mismatched-context cases run here too.
import { buildFollowups, buildContexts, article, html2txt, ask, ungrounded } from "./harness.mjs";

const SYS = "Answer the question using the reference material. Be concise: at most two sentences plus the exact command if one applies.";
const SYS_REFUSE = SYS + " If the reference material does not contain the answer, reply exactly: not found.";
const REPS = +(process.argv[2] ?? 3);

const fu = await buildFollowups();
const first = {};
for (const c of fu) {
  first[c.q2] = (await ask([
    { role: "system", content: SYS },
    { role: "user", content: `Reference:\n${c.ref1}\n\nQuestion: ${c.q1}` },
  ])).content;
}

// ---- duty 1: never reject a correct answer
const rej = { slice: 0, full: 0 }, seen = { slice: new Set(), full: new Set() };
let correct = 0, n = 0;
for (let r = 0; r < REPS; r++) {
  for (const c of fu) {
    const { content } = await ask([
      { role: "system", content: SYS },
      { role: "user", content: `Reference:\n${c.ref1}\n\nQuestion: ${c.q1}` },
      { role: "assistant", content: first[c.q2] },
      { role: "user", content: `Reference:\n${c.ref2}\n\nQuestion: ${c.q2}` },
    ], { temperature: 0.3 });
    n++;
    if (!c.needle.test(content)) continue; // only correct answers can be false-rejected
    correct++;
    for (const [arm, ref] of [["slice", `${c.ref1}\n${c.ref2}`], ["full", c.full]]) {
      const why = ungrounded(content, ref, c.q2);
      if (why) {
        rej[arm]++;
        seen[arm].add(`${c.q2} -> ${why}`);
        if (arm === "slice") console.log(`  slice rejects: ${why}\n      ${content.slice(0, 160).replace(/\n/g, " ")}`);
      }
    }
  }
}

// ---- duty 2: still catch fabrication when the context cannot answer
const ctx = await buildContexts();
const fullOf = {};
for (const [q, path] of [["set the system timezone", "System_time"], ["mount a usb drive automatically", "Udisks"],
  ["encrypt a partition", "Dm-crypt/Device_encryption"], ["generate an ssh key", "SSH_keys"],
  ["create a swap file", "Swap"], ["check what is using disk space", "Core_utilities"]]) {
  fullOf[q] = html2txt(await article(path));
}
const caught = { slice: 0, full: 0 };
let refused = 0, m = 0;
for (let i = 0; i < ctx.length; i++) {
  const q = ctx[i].q, c = ctx[(i + 1) % ctx.length];
  if (ctx[i].needle.test(c.text)) continue;
  m++;
  const { content } = await ask([
    { role: "system", content: SYS_REFUSE },
    { role: "user", content: `Reference:\n${c.text}\n\nQuestion: ${q}` },
  ]);
  const no = /not found/i.test(content);
  refused += no ? 1 : 0;
  // the mismatched context came from c.q's article, so that is the "full" reference
  for (const [arm, ref] of [["slice", c.text], ["full", fullOf[c.q] ?? c.text]]) {
    caught[arm] += (no || ungrounded(content, ref, q)) ? 1 : 0;
  }
  if (!no) console.log(`  fabrication: "${content.slice(0, 110).replace(/\n/g, " ")}"\n      slice=${ungrounded(content, c.text, q) || "MISSED"} | full=${ungrounded(content, fullOf[c.q] ?? c.text, q) || "MISSED"}`);
}

console.log(`\n=== duty 1: ${correct} correct answers of ${n} samples ===`);
console.log(`false rejects — slice ref: ${rej.slice} | full-article ref: ${rej.full}`);
for (const s of seen.slice) console.log(`  slice: ${s}`);
for (const s of seen.full) console.log(`  full:  ${s}`);
console.log(`\n=== duty 2: ${m} mismatched contexts, model refused ${refused} ===`);
console.log(`safe (refused or caught) — slice ref: ${caught.slice}/${m} | full-article ref: ${caught.full}/${m}`);
