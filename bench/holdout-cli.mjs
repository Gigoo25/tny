#!/usr/bin/env bun
import { TNY_ENV } from "./env.mjs";
// Held-out measurement on pages nobody in this repo chose.
//
// The 58 hand-built cases are a regression guard: every scorer weight was tuned against them,
// so they cannot answer "does this work on a question we have never seen". This harness draws
// its sample from the ZIM index (`zimdump list`), so the engine under test has no say in what
// it is asked, and nothing here is ever tuned against.
//
// Regenerate the sample (deterministic — `--random-source=/dev/zero`):
//
//   for z in <se-books>;  do zimdump list zim/$z.zim | grep -E '^questions/[0-9]+/.' \
//     | shuf --random-source=/dev/zero -n 20 | sed "s|^|se\t$z\t|";  done > bench/holdout.tsv
//   for z in <ref-books>; do zimdump list zim/$z.zim | grep -E '^[A-Za-z][^/]*$' \
//     | shuf --random-source=/dev/zero -n 10 | sed "s|^|ref\t$z\t|"; done >> bench/holdout.tsv
//
// Three arms, because "did retrieval work" means different things per question shape:
//
//   title  Stack Exchange question, queried by its own title. LEAKY — retrieval is handed the
//          page's exact wording — so it is an upper bound, not accuracy.
//   body   the same question, queried by the asker's own description of the problem. Real
//          human phrasing of the same need, never the title, so the leak is gone. This is the
//          number that predicts day-to-day use: nobody types a well-formed page title.
//   ref    Wikipedia / Arch wiki / devdocs page, queried by its title. Tests routing across
//          every mounted book at once, which is the entity-lookup shape.
//
// And for the SE arms, given the page was routed at all:
//
//   evidence  does the slice we send carry the answer's own rare terms. Leak-free by
//             construction — the page already matched — so it measures section selection.
const K = process.env.TNY_KIWIX ?? "http://127.0.0.1:8082";
const SHOW = process.argv.includes("--show");
const LIST = process.argv.find(a => a.endsWith(".tsv")) ?? "bench/holdout.tsv";
const ARM = process.argv.find(a => ["title", "body", "ref"].includes(a));

const STOP = new Set(("the and for are was were with that this from have has been into than then when what which about their there these those being other also more most some such only over after before between using used use its his her you your they them will would could should here into onto upon while where whose whom does did done can may might must shall very just like each both any all not but our out off per via etc really much many make made need needed want first second another thing things something anything would could there").split(" "));

// Pages are titled "<question> - <site display name>", and the display name is only knowable
// from the catalog: cooking.stackexchange.com is "Cooking Q&A (Seasoned Advice)", and pages
// carry the parenthesised half. Feeding that suffix as the query measures boilerplate
// handling, not retrieval.
const catalog = await (await fetch(`${K}/catalog/search?count=-1`)).text();
const SITE = catalog.split("<entry>").slice(1).map(e => ({
  name: (e.match(/<name>([^<]+)<\/name>/) ?? [])[1] ?? "",
  title: ((e.match(/<title>([^<]+)<\/title>/) ?? [])[1] ?? "").replace(/&amp;/g, "&").trim(),
}));
const suffixesOf = (book) => {
  const t = SITE.find(s => s.name && book.startsWith(s.name))?.title ?? "";
  return [(t.match(/\(([^)]+)\)/) ?? [])[1], t.replace(/\s*\([^)]*\)\s*/g, " ").trim(), t]
    .filter(Boolean).sort((a, b) => b.length - a.length);
};
const strip = (raw, book) => suffixesOf(book)
  .reduce((s, x) => s.replace(new RegExp(`\\s*[-–|]\\s*${x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*$`, "i"), ""), raw)
  .replace(/\s*[-–|]\s*[^-–|]{0,40}(Stack Exchange|Ask Ubuntu|Stack Overflow|Super User)\s*$/i, "").trim();

function evidenceTerms(answer, asked) {
  const seen = new Set(asked.toLowerCase().match(/[a-z0-9_.\-]{3,}/g) ?? []);
  const freq = {};
  for (const w of answer.toLowerCase().match(/[a-z0-9_.\-]{5,}/g) ?? []) {
    if (STOP.has(w) || seen.has(w)) continue;
    freq[w] = (freq[w] ?? 0) + 1;
  }
  return Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 12).map(([w]) => w);
}

const ctxFor = async (q) => {
  const p = Bun.spawn(["./target/release/tny", "--context", q], { env: TNY_ENV, stdout: "pipe", stderr: "pipe" });
  const out = await new Response(p.stdout).text();
  await p.exited;
  return out.toLowerCase();
};

const rows = (await Bun.file(LIST).text()).trim().split("\n").map(l => l.split("\t"));
const tally = { title: [0, 0], body: [0, 0], ref: [0, 0] };
let evid = 0, evidN = 0, unusable = 0;

for (const [kind, book, path] of rows) {
  if (ARM && !(ARM === "ref" ? kind === "ref" : kind === "se")) continue;
  const html = await (await fetch(`${K}/content/${book}/${path}`)).text();
  const title = strip(((html.match(/<title>([^<]+)<\/title>/) ?? [])[1] ?? "")
    .replace(/&#(\d+);/g, (_, d) => String.fromCharCode(+d)).replace(/&[a-z]+;/g, "'"), book);
  const text = html.replace(/<script[\s\S]*?<\/script>/g, " ").replace(/<style[\s\S]*?<\/style>/g, " ")
    .replace(/<[^>]+>/g, " ").replace(/&[a-z]+;/g, " ").replace(/\s+/g, " ");
  if (!title || title.length < 10) { unusable++; continue; }
  // The page has to be findable in principle: if its own title is not in the context, no arm
  // of this measurement can distinguish ranking from a missing page.
  const hit = (ctx) => ctx.includes(title.toLowerCase().slice(0, 40));

  if (kind === "ref") {
    if (ARM && ARM !== "ref") continue;
    const ok = hit(await ctxFor(title));
    tally.ref[0] += ok ? 1 : 0;
    tally.ref[1]++;
    if (SHOW) console.log(`${ok ? "REF  ok  " : "REF  MISS"} ${title.slice(0, 62)}`);
    continue;
  }

  // Posts are `<div class="s-prose js-post-body">`: the first is the question, the rest are
  // answers. Text-marker heuristics ("N Answers N") match the page chrome instead, which is
  // how an earlier run of this harness reported a flat 0 % and meant nothing.
  const posts = html.split(/class="[^"]*js-post-body[^"]*"/).slice(1)
    // The split lands mid-tag, so the rest of the opening tag (`itemprop="text">`) is still
    // attached and would be fed to the engine as query terms.
    .map(s => s.replace(/^[^>]*>/, "").replace(/<[^>]+>/g, " ").replace(/&[a-z]+;/g, " ").replace(/\s+/g, " ").trim());
  if (posts.length < 2 || posts[1].length < 200) { unusable++; continue; }
  const answer = posts.slice(1).join(" ").slice(0, 2200);
  // The asker's own words: how a person describes the problem, never how they headline it.
  const body = posts[0].slice(0, 220);

  if (!ARM || ARM === "title") {
    const ctx = await ctxFor(title);
    const ok = hit(ctx);
    tally.title[0] += ok ? 1 : 0;
    tally.title[1]++;
    if (ok) {
      const terms = evidenceTerms(answer, title);
      const found = terms.filter(t => ctx.includes(t)).length;
      evidN++;
      if (found >= Math.max(3, Math.ceil(terms.length * 0.34))) evid++;
    }
  }
  if ((!ARM || ARM === "body") && body.length > 60) {
    const ok = hit(await ctxFor(body));
    tally.body[0] += ok ? 1 : 0;
    tally.body[1]++;
    if (SHOW) console.log(`${ok ? "BODY ok  " : "BODY MISS"} ${body.slice(0, 58).padEnd(60)} -> ${title.slice(0, 40)}`);
  }
}

const pct = ([a, b]) => `${a}/${b}${b ? ` (${Math.round((100 * a) / b)}%)` : ""}`;
console.log(`\nheld-out  ${rows.length} pages, ${unusable} unusable`);
console.log(`routed title  ${pct(tally.title)}  — leaky upper bound, query is the page's own title`);
console.log(`routed body   ${pct(tally.body)}  — the asker's own words: what day-to-day use looks like`);
console.log(`routed ref    ${pct(tally.ref)}  — wiki/devdocs entity lookup across every book`);
console.log(`evidence      ${pct([evid, evidN])}  — section selection, given the page was routed`);
