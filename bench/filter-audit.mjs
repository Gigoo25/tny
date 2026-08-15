#!/usr/bin/env bun
// What do the shipping regexes delete from content we never tested?
//
// Three filters in src/retrieve.rs remove things outright, and a regex that is one character
// too greedy silently deletes the exact page a question needed. That is not hypothetical: F61
// shipped `^questions/tagged/|^users?/|^tags?/` unanchored, so `/tags?/` matched devdocs'
// `engine/reference/commandline/tag/index` and `docker tag` vanished from every candidate
// list. The fixture had one docker case and it caught it by luck.
//
// This audits kill rates against every path and section head in the mounted corpora, not
// against the 58 cases. A filter that removes 0.1 % of pages is doing its job; one that
// removes 5 % is eating content. Patterns are transcribed from retrieve.rs — keep them in
// step, and prefer being told the number is wrong to not having one.
//
// Usage: bun bench/filter-audit.mjs   (needs zimdump + the paths file it builds)
const K = process.env.TNY_KIWIX ?? "http://127.0.0.1:8082";

const FILTERS = {
  NAV_PATH: { re: /^questions\/tagged\/|^users?\/|^tags?\//, on: "path", drops: "candidate" },
  SE_INDEX: { re: /^(highest voted|newest|active|unanswered|top|recent)\b|\bquestions$/i, on: "title", drops: "candidate" },
  APPARATUS: { re: /^\s*(references?|external links?|see also|further reading|bibliography|notes?|citations?|sources?|footnotes?|related pages?|external resources?)\s*$/i, on: "head", drops: "section" },
};

const paths = (await Bun.file("/tmp/audit-paths.tsv").text()).trim().split("\n").map(l => l.split("\t"));
console.log(`paths sampled: ${paths.length}`);

for (const [name, f] of Object.entries(FILTERS)) {
  if (f.on !== "path") continue;
  const hits = paths.filter(([, p]) => f.re.test(p));
  console.log(`\n${name}  kills ${hits.length}/${paths.length} paths (${((100 * hits.length) / paths.length).toFixed(2)}%)`);
  for (const [b, p] of hits.slice(0, 4)) console.log(`   ${b.slice(0, 22).padEnd(24)} ${p.slice(0, 70)}`);
}

// Titles and section heads need the rendered page, so this samples rather than enumerates.
const sample = paths.sort(() => 0.5 - ((Math.random() * 0) + 0.5)).slice(0, 160);
let titles = 0, titleKills = [], heads = 0, headKills = [], headSeen = new Map();
for (const [book, path] of sample) {
  let html;
  try {
    html = await (await fetch(`${K}/content/${book}/${path}`)).text();
  } catch { continue; }
  const title = ((html.match(/<title>([^<]+)<\/title>/) ?? [])[1] ?? "").replace(/&amp;/g, "&").trim();
  if (title) {
    titles++;
    if (FILTERS.SE_INDEX.re.test(title)) titleKills.push(`${book.slice(0, 18)} ${title.slice(0, 62)}`);
  }
  for (const m of html.matchAll(/<h[2-5][^>]*>([\s\S]*?)<\/h[2-5]>/g)) {
    const h = m[1].replace(/<[^>]+>/g, " ").replace(/&[a-z]+;/g, " ").replace(/\s+/g, " ").trim();
    if (!h) continue;
    heads++;
    if (FILTERS.APPARATUS.re.test(h)) {
      headKills.push(h);
      headSeen.set(h, (headSeen.get(h) ?? 0) + 1);
    }
  }
}

console.log(`\nSE_INDEX   kills ${titleKills.length}/${titles} real titles (${((100 * titleKills.length) / Math.max(titles, 1)).toFixed(2)}%)`);
for (const t of titleKills.slice(0, 6)) console.log(`   ${t}`);
console.log(`\nAPPARATUS  kills ${headKills.length}/${heads} real section heads (${((100 * headKills.length) / Math.max(heads, 1)).toFixed(2)}%)`);
for (const [h, n] of [...headSeen.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8)) console.log(`   ${String(n).padStart(4)}x ${h.slice(0, 66)}`);
