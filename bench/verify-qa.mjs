#!/usr/bin/env bun
// Verifies bench/fixture-qa.mjs against the live kiwix corpus: for every case,
// (1) prep(query) must return a hit whose title matches titleRe, uniquely among the 8 hits,
// (2) that article's plain text must contain needleRe,
// (3) reports the Arch Wiki top-5 for the same query, so the "SE is the right source"
//     judgement stays auditable.
import { search, article, html2txt, prep } from "./harness.mjs";
import { CASES } from "./fixture-qa.mjs";

const ARCH = "archlinux_en_all_maxi_2026-07";
let pass = 0;
for (const [q, intent, book, titleRe, needleRe] of CASES) {
  const p = prep(q);
  const [hits, aw] = await Promise.all([search(p, book, 8), search(p, ARCH, 5)]);
  const ms = hits.filter(h => titleRe.test(h.title));
  const hit = ms[0];
  const txt = hit ? html2txt(await article(hit.path, book)) : "";
  const ok = !!hit && ms.length === 1 && needleRe.test(txt);
  pass += ok ? 1 : 0;
  console.log(`${ok ? "PASS" : "FAIL"} [${intent}] ${q}`);
  console.log(`     title=${hit ? `rank ${hits.indexOf(hit)} "${hit.title}"` : "NO MATCH"} unique=${ms.length === 1}`);
  console.log(`     needle=${needleRe} ${hit ? (needleRe.test(txt) ? "found" : "MISSING") : "-"}  arch: ${aw.map(h => h.title).join(" | ") || "(no hits)"}`);
}
console.log(`\n${pass}/${CASES.length} cases pass checks 1 and 2`);
process.exit(pass === CASES.length ? 0 : 1);
