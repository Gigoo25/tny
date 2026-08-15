#!/usr/bin/env bun
// F91: why does a new book cost rank-1 precision, and does rank fusion fix it?
//
// `tny --rank` prints the shortlist the live ranker produced. For every fixture case this
// finds where the right article landed, and when it is not first, what beat it — the only way
// to tell a scoring bug (the answer is there, mis-ordered) from a retrieval one.
import { CASES } from "./grade.mjs";

const CACHE = "bench/.shortlists.json";
const rebuild = process.argv.includes("--rebuild");
const cache = !rebuild && await Bun.file(CACHE).exists() ? JSON.parse(await Bun.file(CACHE).text()) : {};

async function shortlist(query) {
  if (cache[query]) return cache[query];
  const p = Bun.spawn(["./target/release/tny", "--rank", query], { stdout: "pipe", stderr: "pipe" });
  const out = (await new Response(p.stdout).text()).trim();
  await p.exited;
  cache[query] = out
    .split("\n")
    .filter(Boolean)
    .map(l => {
      const [book, title, path] = l.split("\t");
      return { book, title, path };
    });
  return cache[query];
}

let at1 = 0, inList = 0, displaced = [];
for (const [set, query, , , titleRe] of CASES) {
  const rows = await shortlist(query);
  const hit = rows.findIndex(r => titleRe.test(r.title ?? ""));
  if (hit === 0) at1++;
  if (hit >= 0) inList++;
  // The interesting class: retrieved, but something else took the podium.
  if (hit > 0) displaced.push({ set, query, at: hit, right: rows[hit], by: rows.slice(0, hit) });
}
await Bun.write(CACHE, JSON.stringify(cache, null, 1));

console.log(`article@1 ${at1}/${CASES.length}   in shortlist ${inList}/${CASES.length}   displaced ${displaced.length}\n`);
// Which books do the displacing, and by how much: a book that wins slots it should not is
// visible as a repeat offender here long before it shows up as a lost answer.
const blame = {};
for (const d of displaced) for (const b of d.by) blame[b.book] = (blame[b.book] ?? 0) + 1;
console.log("displacing book                                    times");
for (const [b, n] of Object.entries(blame).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${b.slice(0, 46).padEnd(48)} ${n}`);
}
if (process.argv.includes("--show")) {
  console.log("");
  for (const d of displaced.slice(0, 12)) {
    console.log(`${d.set} "${d.query.slice(0, 52)}" — right article at ${d.at}`);
    for (const b of d.by) console.log(`    beaten by  ${b.book.slice(0, 34).padEnd(36)} ${(b.title ?? "").slice(0, 46)}`);
    console.log(`    wanted     ${d.right.book.slice(0, 34).padEnd(36)} ${(d.right.title ?? "").slice(0, 46)}`);
  }
}
