#!/usr/bin/env bun
// Retrieval precision on questions nobody here wrote, over pages the fixture never pinned.
//
// What this used to be, and why it was thrown away (F108): the query was the target page's own
// title, and the check was `ctx.includes(title.slice(0, 40))` against a context that prints the
// title of every retrieved article — so it passed whenever the page reached the top three, on a
// query that was the answer key. The "leak-free" body arm fed 220 verbatim characters of the
// target document, which carry ~20 terms that occur in exactly one page in the corpus; BM25
// cannot lose. It scored 55/57 and 54/57 while the human-written fixture scored rank-1 34/58,
// and when man pages cost six end-to-end answers it reported no change at all. A guard that
// cannot fail is not a guard.
//
// What it is now: the query is the Stack Exchange question *slug* — words a real user typed,
// which is where these pages came from — and the metric is strict rank-1 on the target path,
// the same metric rank-cli.mjs uses. Directly comparable to its 34/58.
//
// What it still is not: a generalisation test. The slug is the page's own title, so this
// measures whether ranking finds a page from its own words, not whether it survives a
// paraphrase. Paraphrases have to be written without sight of the page, and nothing in bench/
// produces one. Until they exist, the 58-case fixture is the only evidence of generalisation
// in this repo, and it has none.
import { TNY_ENV } from "./env.mjs";

const ARM = process.argv.slice(2).find(a => !a.startsWith("-"));
const rows = (await Bun.file("bench/holdout.tsv").text())
  .split("\n")
  .filter(Boolean)
  .map(l => l.split("\t"))
  .filter(([arm]) => !ARM || arm === ARM);

// questions/10012/non-case-sensitive-sed-openwrt -> "non case sensitive sed openwrt". The
// numeric id is not a query term and a real user never types it. A reference page's path often
// ends in `index`, which is the directory's name, not the page's: take the segment before it.
const queryOf = path => {
  const parts = path.replace(/\.html?$/, "").split("/").filter(Boolean);
  const last = parts.pop();
  const seg = last === "index" && parts.length ? parts.pop() : last;
  return seg.replace(/[-_]+/g, " ").replace(/\b\d+\b/g, "").trim();
};

const one = async ([arm, book, path]) => {
  const query = queryOf(path);
  const p = Bun.spawn(["./target/release/tny", "--rank", query], { env: TNY_ENV, stdout: "pipe", stderr: "pipe" });
  const out = await new Response(p.stdout).text();
  await new Response(p.stderr).text();
  await p.exited;
  const hits = out.trim().split("\n").filter(Boolean).map(l => l.split("\t"));
  // Match on the path, not the title: the title has to be fetched, and the path is what tny
  // would read. Book ids carry a date suffix the shortlist may print differently.
  const at = hits.findIndex(([b = "", , pth = ""]) => pth === path && b.startsWith(book.split("_20")[0]));
  return { arm, query, at, n: hits.length };
};

const results = [];
for (let i = 0; i < rows.length; i += 4) {
  results.push(...(await Promise.all(rows.slice(i, i + 4).map(one))));
}

const tally = {};
for (const r of results) {
  tally[r.arm] ??= { n: 0, rank1: 0, list: 0 };
  tally[r.arm].n++;
  if (r.at === 0) tally[r.arm].rank1++;
  if (r.at >= 0) tally[r.arm].list++;
  if (r.at !== 0) console.log(`${r.at < 0 ? "ABSENT" : `rank ${r.at + 1}`.padEnd(6)} ${r.arm} ${r.query}`);
}
console.log("");
for (const [arm, t] of Object.entries(tally)) {
  console.log(`${arm.padEnd(4)} rank-1 ${t.rank1}/${t.n}   in shortlist ${t.list}/${t.n}`);
}
