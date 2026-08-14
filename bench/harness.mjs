#!/usr/bin/env bun
// tny measurement harness — reproduces every benchmark in NOTES.md.
//   bun bench/harness.mjs [rank|judge|sections|answers|corpus|thinking|all]
//
// Requires three servers (see NOTES.md "Environment"):
//   kiwix-serve  --port 8082 ... zim/*.zim
//   llama-server -hf ggml-org/Qwen3.5-0.8B-GGUF:Q8_0 --jinja --port 8080
//   llama-server -hf ggml-org/bge-small-en-v1.5-Q8_0-GGUF --embeddings --pooling cls --port 8084

const KIWIX = process.env.TNY_KIWIX ?? "http://127.0.0.1:8082";
const CHAT = process.env.TNY_CHAT ?? "http://127.0.0.1:8080";
const EMBED = process.env.TNY_EMBED ?? "http://127.0.0.1:8084";
const BOOK = process.env.TNY_BOOK ?? "archlinux_en_all_maxi_2026-07";

// Asymmetric retrieval prefixes differ per embedder (F24):
//   bge-small: query prefix only          nomic: search_query:/search_document:
const QP = process.env.TNY_QP ?? "Represent this sentence for searching relevant passages: ";
const DP = process.env.TNY_DP ?? "";
const SYS = "Answer the question using the reference material. Be concise: at most two sentences plus the exact command if one applies.";
const NOCTX = "Answer the question. Be concise: at most two sentences plus the exact command if one applies.";

// ---------------------------------------------------------------- primitives

// F15/F35: raw conversational queries under-retrieve; strip question words/stopwords.
// F35: kiwix ANDs every term, so one term no document shares zeroes the whole query —
// "string versus str slice" returned 0 hits, and dropping "versus" returns `str` first.
// Comparison words are the common culprit and they carry no retrieval signal.
const STOP = /^(how|do|i|the|a|an|my|is|are|why|what|when|where|to|in|on|from|of|for|can|does|with|and|it|not|be|get|set|make|use|versus|vs|difference|differences|between|tradeoff|tradeoffs|pros|cons|better|worse|should|or|choose|choosing|compare|comparison|alternative|alternatives)$/i;
export const prep = q => q.toLowerCase().replace(/[^\w\s.:+#-]/g, " ").split(/\s+/).filter(w => w && !STOP.test(w)).join(" ");

// F8: a 350M model cannot tell a citation marker from a datum
export const denoise = s => s.replace(/\[\s*(\d+|edit|citation needed|note \d+)\s*\]/gi, " ").replace(/\s+/g, " ").trim();

export const html2txt = h => denoise(h
  .replace(/<script[\s\S]*?<\/script>/g, " ").replace(/<style[\s\S]*?<\/style>/g, " ")
  .replace(/<[^>]+>/g, " ")
  .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&").replace(/&quot;/g, '"')
  .replace(/&#(\d+);/g, (_, d) => String.fromCharCode(+d)));

const cos = (a, b) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / Math.sqrt(na * nb);
};

// F17: Reciprocal Rank Fusion over ranked lists of the same keys
export const rrf = (lists, k = 10) => {
  const s = new Map();
  for (const l of lists) l.forEach((key, i) => s.set(key, (s.get(key) ?? 0) + 1 / (k + i + 1)));
  return [...s.entries()].sort((a, b) => b[1] - a[1]).map(([key]) => key);
};

// F31: lexical term scoring. Measured 14/14 at section level against embeddings'
// 11/14, and it is the only signal that reaches §Protection (rank 11 of 41 by
// embedding, rank 36 for the elliptical form) in a 41-section article.
// deliberately NOT F15's `STOP`: that list strips set/make/use/get, which are exactly
// the verbs a section head uses ("Set system clock"). This list strips the follow-up
// filler instead ("only", "ones", "again", "instead"). 14/14 was measured with this.
const STOP_LEX = /^(how|do|i|the|a|an|my|is|are|why|what|when|where|to|in|on|from|of|for|can|does|with|and|it|only|ones|one|again|instead|all)$/i;
export const terms = q => [...new Set(q.toLowerCase().match(/[a-z0-9-]{2,}/g) ?? [])].filter(w => !STOP_LEX.test(w));

// head/title hits weigh 3; body hits saturate at 5 so one long section cannot win by
// repetition alone
export function lexScore(head, body, t) {
  const h = head.toLowerCase(), b = body.toLowerCase();
  let s = 0;
  for (const w of t) {
    if (h.includes(w)) s += 3;
    s += Math.min(b.split(w).length - 1, 5);
  }
  return s;
}

// F31: the answer is often deeper into a section than the head of it. Selecting the
// right section and then slicing its first 600 chars threw the answer away —
// §Protection in OpenSSH mentions PermitRootLogin well past that cut. Centre the
// window on the densest run of query terms instead of the section start.
export function window(text, t, budget) {
  if (text.length <= budget) return text;
  let best = 0, bestScore = -1;
  const step = Math.max(80, Math.floor(budget / 4));
  for (let at = 0; at + 1 <= text.length; at += step) {
    const slice = text.slice(at, at + budget).toLowerCase();
    let s = 0;
    for (const w of t) if (slice.includes(w)) s++;
    if (s > bestScore) { bestScore = s; best = at; }
  }
  // never start mid-word
  const cut = best === 0 ? 0 : text.indexOf(" ", best) + 1 || best;
  return (cut ? "… " : "") + text.slice(cut, cut + budget);
}

// F14: the English _maxi ZIM is full of localised duplicates
const LOCALISED = /\((Magyar|Deutsch|Español|Français|Português|Italiano|Polski|Русский|简体|正體|日本語|한국어|Türkçe|Nederlands|Čeština|Ελληνικά|עברית|فارسی|العربية|Indonesia|Tiếng Việt|Norsk|Dansk|Svenska|Suomi|Română|Български|Українська|Hrvatski|Slovenčina|Lietuvių|Català)\)/;

// F27: model-free grounding check. Both observed fabrication modes are detectable
// without a model: a command the reference never mentions, or an answer that merely
// restates the question. Returns "" when grounded, else the reason.
// F38/F32 interaction: `ref` is the source document (F32 — fewer false rejects on
// commands, since a neighbouring section legitimately names `cryptsetup`), but `seen`
// is the slice actually shown to the model. A claim about the *other* side of a
// comparison must be supported by what was shown: `chrony` appears in
// systemd-timesyncd's "See also", which was enough to license "chrony is the
// recommended alternative" under the wider reference.
export function ungrounded(answer, ref, q, seen = ref) {
  const a = answer.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  if (!a) return "empty";
  // word-boundary match, NOT substring: `du` must not be satisfied by "produce",
  // and a 2-char filter would drop du/df/ls/ip — the commands asked about most
  const inRef = c => new RegExp(`(?<![\\w-])${c.replace(/[.*+?^${}()|[\]\\/-]/g, "\\$&")}(?![\\w-])`).test(ref);
  // a command reaches us three ways: inline `cmd`, a fenced block, or a prompt line
  // ("# timedatectl set-timezone"). Missing the third made the echo rule misfire on a
  // correct answer — 1 false reject in 18 samples before this was widened.
  const code = [
    ...[...a.matchAll(/```[\w]*\s*([\s\S]*?)```/g)].map(m => m[1]),
    ...[...a.matchAll(/`([^`\n]+)`/g)].map(m => m[1]),
    ...[...a.matchAll(/^\s*[$#]\s+(.+)$/gm)].map(m => m[1]),
  ];
  const cmds = [...new Set(code
    .flatMap(s => s.split("\n"))
    .map(l => l.trim().replace(/^[$#]\s*/, "").split(/[\s|;]+/)[0])
    // paths are not commands: an answer quoting `/home/username/.ssh/id_ed25519` was
    // falsely rejected because the reference spells that example differently
    .filter(c => c.length >= 2 && /^[\w.-]+$/.test(c)))];
  const absent = cmds.filter(c => !inRef(c));
  // F38: a comparison answer may assert about a tool whose article was never
  // retrieved — "chrony is the recommended alternative", with chrony absent from the
  // reference. No command appears, so the command rule above cannot see it. Both
  // sides' names come from the question's own grammar, so this stays model-free.
  const sides = splitCompare(q);
  if (sides) {
    const has = (s, hay) => new RegExp(`(?<![\\w-])${s.replace(/[.*+?^${}()|[\]\\/-]/g, "\\$&")}`, "i").test(hay);
    const claimed = [sides.a, sides.b].filter(s => has(s, a));
    const unsupported = claimed.filter(s => !has(s, seen));
    if (unsupported.length) return `asserts about ${unsupported.join(", ")}, absent from what was shown`;
  }
  if (absent.length) return `command not in reference: ${absent.join(", ")}`;
  if (!cmds.length) {
    // narrow on purpose. A word-overlap ratio caught one real echo and produced two
    // false rejects (`du -h`, and unmarked "Use timedatectl set-timezone to …"), and a
    // false reject is worse: it turns a correct answer into "not found". Exact
    // containment catches the observed degeneration and cannot misfire on prose that
    // adds anything of its own.
    const norm = s => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    if (norm(q).includes(norm(a))) return "restates the question";
    // observed degeneration variant: the model asks a question back instead of answering
    if (/^[^.!]*\?\s*$/.test(a)) return "asks a question back";
  }
  return "";
}

// ---------------------------------------------------------------- services

export async function embed(inputs) {
  const r = await fetch(`${EMBED}/v1/embeddings`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input: inputs }),
  });
  const j = await r.json();
  if (!j.data) throw new Error(`embed failed: ${JSON.stringify(j).slice(0, 200)}`);
  return j.data.map(d => d.embedding);
}

// F19: thinking must be disabled, and empty content is an ERROR not an answer
export async function ask(messages, opts = {}) {
  const r = await fetch(`${CHAT}/v1/chat/completions`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages, temperature: 0.1, top_k: 50, repeat_penalty: 1.05, max_tokens: 160,
      chat_template_kwargs: { enable_thinking: false }, ...opts,
    }),
  });
  const j = await r.json(), m = j.choices?.[0]?.message ?? {};
  const content = m.content ?? "";
  if (!content.trim() && (m.reasoning_content ?? "").length) {
    return { content: "", error: "EMPTY_CONTENT_REASONING_ONLY", usage: j.usage };
  }
  return { content, usage: j.usage, finish: j.choices?.[0]?.finish_reason };
}

// F11: /search needs books.name=<filename stem>; /suggest needs content=<stem>
export async function search(query, book = BOOK, want = 8) {
  const url = `${KIWIX}/search?books.name=${book}&pattern=${encodeURIComponent(query)}&format=xml&pageLength=30`;
  const xml = await (await fetch(url)).text();
  const rows = xml.split("<item>").slice(1).map(it => ({
    title: (it.match(/<title>([^<]*)<\/title>/) ?? [])[1] ?? "",
    path: ((it.match(/<link>([^<]*)<\/link>/) ?? [])[1] ?? "").replace(/^.*\/content\/[^/]+\//, ""),
    snip: ((it.match(/<description>([\s\S]*?)<\/description>/) ?? [])[1] ?? "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim(),
  }));
  const seen = new Set();
  return rows.filter(c => {
    if (LOCALISED.test(c.title)) return false;
    const base = c.title.replace(/\s*\(.*\)$/, "");
    if (seen.has(base)) return false;
    seen.add(base);
    return true;
  }).slice(0, want);
}

// F12/F34: no books.name = every mounted ZIM, with real body-level FTS even on
// `_ftindex:no` books. Keeps each hit's book, which `search` above discards.
export async function searchAll(query, want = 8) {
  const url = `${KIWIX}/search?pattern=${encodeURIComponent(query)}&format=xml&pageLength=30`;
  const xml = await (await fetch(url)).text();
  const rows = xml.split("<item>").slice(1).map(it => {
    const link = ((it.match(/<link>([^<]*)<\/link>/) ?? [])[1] ?? "");
    return {
      title: (it.match(/<title>([^<]*)<\/title>/) ?? [])[1] ?? "",
      book: (link.match(/\/content\/([^/]+)/) ?? [])[1] ?? "",
      path: link.replace(/^.*\/content\/[^/]+\//, ""),
      snip: ((it.match(/<description>([\s\S]*?)<\/description>/) ?? [])[1] ?? "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim(),
    };
  });
  const seen = new Set();
  return rows.filter(c => {
    if (!c.book || LOCALISED.test(c.title)) return false;
    // dedupe per book: the same title in two books is two distinct answers
    const key = `${c.book}\u0000${c.title.replace(/\s*\(.*\)$/, "")}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  }).slice(0, want);
}

// F13: fuzzy title lookup for _ftindex:no books; keep only kind:"path" rows
export async function suggest(term, book) {
  const r = await fetch(`${KIWIX}/suggest?content=${book}&term=${encodeURIComponent(term)}`);
  const rows = JSON.parse((await r.text()).replace(/&apos;/g, "'"));
  return rows.filter(o => o.kind === "path").map(o => ({ title: o.value, path: o.path }));
}

export const article = async (path, book = BOOK) =>
  await (await fetch(`${KIWIX}/content/${book}/${path}`)).text();

// F13: anchors survive inside ZIM pages (240 in std/vec/struct.vec)
export function sliceAnchor(html, anchor, budget = 1200) {
  const i = html.indexOf(`id="${anchor}"`);
  return i < 0 ? null : html2txt(html.slice(i, i + 6000)).slice(0, budget);
}

// ---------------------------------------------------------------- sections

// F31: split on h2–h5, not h2–h3. OpenSSH's §Protection is one 12.9 KB h2 chunk with
// PermitRootLogin at offset 4,704: selection ranked it #1 and the window still missed
// the answer. Splitting deeper yields 77 sections of ≤3.4 KB and fixes it for free —
// embedding selection went 12/14 -> 14/14 at identical context size.
export function sectionsOf(html) {
  const h = [...html.matchAll(/<h[2345][^>]*>([\s\S]*?)<\/h[2345]>/g)].map(m => ({
    head: m[1].replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim(),
    at: m.index, end: m.index + m[0].length,
  }));
  return h.map((x, i) => ({ head: x.head, text: html2txt(html.slice(x.end, h[i + 1]?.at ?? html.length)) }))
    .filter(s => s.text.length > 80);
}

// F31: embedding selection stays. At identical h2–h5 splitting and windowing it is
// 14/14 at top-3 in ~1,640 chars, while pure lexical needs top-5 and ~2,680 chars for
// the same 14/14. Prefill dominates latency here, so 35 MB of embedder buys a 39 %
// smaller prompt — it is cheaper than the tokens it saves.
export async function pickSections(html, q, topN = 3, per = 600) {
  const secs = sectionsOf(html);
  const t = terms(q);
  if (!secs.length) return { heads: ["(lead)"], text: window(html2txt(html), t, per * topN) };
  const v = await embed([QP + q, ...secs.map(s => `${DP}${s.head}. ${s.text.slice(0, 400)}`)]);
  const top = secs.map((s, i) => ({ s, k: cos(v[0], v[i + 1]) })).sort((a, b) => b.k - a.k).slice(0, topN);
  return { heads: top.map(o => o.s.head), text: top.map(o => `## ${o.s.head}\n${window(o.s.text, t, per)}`).join("\n\n") };
}

// F31: the no-server fallback. Same 14/14, but needs top-5 to get there.
export function pickSectionsLex(html, q, topN = 5, per = 600) {
  const secs = sectionsOf(html);
  const t = terms(q);
  if (!secs.length) return { heads: ["(lead)"], text: window(html2txt(html), t, per * topN) };
  const top = secs.map(s => ({ s, k: lexScore(s.head, s.text, t) }))
    .sort((a, b) => b.k - a.k).slice(0, topN);
  return { heads: top.map(o => o.s.head), text: top.map(o => `## ${o.s.head}\n${window(o.s.text, t, per)}`).join("\n\n") };
}

// F31: article ranking — Xapian order fused with lexical title+snippet scoring.
// 9/10, matching the 3-way embedding version, with no embedding server.
export function rankArticles(q, cands) {
  const t = terms(q);
  const lex = cands.map((c, i) => ({ i, k: lexScore(c.title, denoise(c.snip).slice(0, 300), t) }))
    .sort((a, b) => b.k - a.k).map(o => cands[o.i].title);
  const xapian = cands.map(c => c.title);
  return { xapian, lex, fused: rrf([xapian, lex]) };
}

// F17 (superseded by F31): Xapian order + 2 embedding views, fused. Also 9/10.
export async function rankArticlesEmbed(q, cands) {
  const v = await embed([
    QP + q,
    ...cands.map(c => DP + c.title),
    ...cands.map(c => `${DP}${c.title}. ${denoise(c.snip).slice(0, 300)}`),
  ]);
  const qv = v[0], tv = v.slice(1, 1 + cands.length), sv = v.slice(1 + cands.length);
  const byT = cands.map((c, i) => ({ t: c.title, k: cos(qv, tv[i]) })).sort((a, b) => b.k - a.k).map(o => o.t);
  const byS = cands.map((c, i) => ({ t: c.title, k: cos(qv, sv[i]) })).sort((a, b) => b.k - a.k).map(o => o.t);
  return { xapian: cands.map(c => c.title), byT, byS, fused: rrf([cands.map(c => c.title), byT, byS]) };
}

// ---------------------------------------------------------------- benchmarks

// F17 article-selection benchmark
const RANK_CASES = [
  ["mount a usb drive automatically", /udisks|autofs|fstab|removable media/i],
  ["make a bootable usb from an iso", /usb flash installation|multiboot usb/i],
  ["set the system timezone", /system time/i],
  ["why is my wifi not connecting", /networkmanager|netctl|connman|wireless|wpa|iwd/i],
  ["encrypt a partition", /dm-crypt/i],
  ["list all systemd services", /^systemd/i],
  ["create a swap file", /swap/i],
  ["install packages with pacman", /pacman/i],
  ["configure a static ip address", /network configuration|systemd-networkd|netctl|networkmanager|dhcpcd/i],
  ["enable the ssh server", /openssh|ssh/i],
];

// F20/F21/F22 answering benchmark: article is fixed so the model is measured alone
const ANSWER_CASES = [
  ["set the system timezone", "System_time", /timedatectl set-timezone/i],
  ["mount a usb drive automatically", "Udisks", /udiskie|udisksctl|udevil/i],
  ["encrypt a partition", "Dm-crypt/Device_encryption", /cryptsetup.*luksFormat/i],
  ["generate an ssh key", "SSH_keys", /ssh-keygen/i],
  ["create a swap file", "Swap", /mkswap|fallocate|swapon/i],
  ["check what is using disk space", "Core_utilities", /\bdu\b|ncdu|gdu/i],
];

// F28/F29/F30: follow-up turns. Every second question is elliptical ("it", "one",
// "the failed ones") so it is meaningless without the first — that is the point.
// Each expectation was verified present in the corpus before the fixture was written.
const FOLLOWUP_CASES = [
  ["create a swap file", "how do I make it permanent across reboots", "Swap", /fstab/i, "same"],
  ["create a swap file", "how do I turn it off again", "Swap", /swapoff/i, "same"],
  ["encrypt a partition", "how do I unlock it at boot", "Dm-crypt/System_configuration", /crypttab|keyfile/i, "same"],
  ["list all systemd services", "how do I see only the failed ones", "Systemd", /--failed/i, "same"],
  ["install packages with pacman", "how do I remove one instead", "Pacman", /-Rs|--remove/i, "same"],
  // needle is the actionable command, not the word "ntp": a raw dump of `ntp.org`
  // server lines from the reference matched /ntp/i while answering nothing
  ["set the system timezone", "how do I keep it synced automatically", "Systemd-timesyncd", /set-ntp|enable.{0,12}systemd-timesyncd/i, "other"],
];

export async function buildFollowups() {
  const out = [];
  for (const [q1, q2, path, needle, kind] of FOLLOWUP_CASES) {
    const html = await article(path);
    const s1 = await pickSections(html, q1, 3, 600);
    const s2 = await pickSections(html, `${q1} ${q2}`, 3, 600);
    // `full` is the F32 grounding candidate: the whole source document, not the slice
    out.push({ q1, q2, path, needle, kind, ref1: s1.text, ref2: s2.text, heads2: s2.heads, full: html2txt(html), has: needle.test(s2.text) });
  }
  return out;
}

export async function buildContexts() {
  const out = [];
  for (const [q, path, needle] of ANSWER_CASES) {
    const html = await article(path);
    const s = await pickSections(html, q, 3, 600);
    // F32: `full` is what the grounding check measures against — the source document,
    // not the slice. The slice rejected a correct answer for citing `cryptsetup`.
    out.push({ q, needle, heads: s.heads, text: s.text, full: html2txt(html), has: needle.test(s.text) });
  }
  return out;
}

async function benchRank() {
  console.log("== F17/F31 article selection ==");
  const sc = { xapian: 0, lex: 0, fused: 0 }, emb = { byT: 0, byS: 0, fused: 0 };
  let n = 0;
  for (const [q, want] of RANK_CASES) {
    const c = await search(prep(q));
    if (!c.length) { console.log(`  ${q}: NO CANDIDATES`); continue; }
    n++;
    const r = rankArticles(q, c);
    for (const k of Object.keys(sc)) sc[k] += want.test(r[k][0]) ? 1 : 0;
    const e = await rankArticlesEmbed(q, c);
    for (const k of Object.keys(emb)) emb[k] += want.test(e[k][0]) ? 1 : 0;
    console.log(`  ${want.test(r.fused[0]) ? "OK" : "x "} ${q} -> ${r.fused[0]}`);
  }
  console.log(`  of ${n}: xapian ${sc.xapian} | lexical ${sc.lex} | xapian+lex RRF ${sc.fused}   (expect 9/10, model-free)`);
  console.log(`  of ${n}: embedding arms — embT ${emb.byT} | embS ${emb.byS} | 3-way RRF ${emb.fused}   (F17, superseded: same 9/10 for 35 MB + a server)`);
}

// F34: cross-book retrieval. With three ZIMs mounted, does searching all of them at
// once still put the right article first — and does fusing per-book result lists beat
// the single all-books query? Expectations were probed per book before being written:
// every case's target was confirmed rank-1 within its own book.
const CROSS_BOOKS = {
  bash: "devdocs_en_bash_2026-04",
  rust: "devdocs_en_rust_2026-07",
  arch: "archlinux_en_all_maxi_2026-07",
};
const CROSS_CASES = [
  ["shell parameter expansion", "bash", /parameter expansion/i],
  ["brace expansion bash", "bash", /brace expansion/i],
  ["bash trap builtin signal", "bash", /signals|bourne shell builtins/i],
  ["bash here document redirection", "bash", /redirection|here doc/i],
  ["bash shell function definition syntax", "bash", /shell function|definitions/i],
  ["HashMap entry api", "rust", /hashmap/i],
  ["Arc Mutex shared state", "rust", /concurrency|sync::|mutex/i],
  ["Box dyn Error trait object", "rust", /error|box/i],
  ["iterator collect into Vec", "rust", /iterator|vec/i],
  ["String versus str slice", "rust", /string|str/i],
  ["create a swap file", "arch", /^swap/i],
  ["encrypt a partition luks", "arch", /dm-crypt/i],
  ["mount a usb drive automatically", "arch", /udisks|udev|removable|fstab/i],
  ["list all systemd services", "arch", /^systemd/i],
  ["install packages with pacman", "arch", /pacman/i],
];

export async function benchCross() {
  console.log("== F34 cross-book retrieval: all-books vs oracle vs fused ==");
  const sc = { oracle: 0, allRaw: 0, allRRF: 0, perFused: 0 };
  const bk = { allRaw: 0, allRRF: 0, perFused: 0 };
  const ms = { oracle: 0, allRaw: 0, perFused: 0 };
  for (const [q, tag, want] of CROSS_CASES) {
    const book = CROSS_BOOKS[tag];

    // A: oracle — we are told the right book. Upper bound on retrieval.
    let t0 = Date.now();
    const oracle = await search(prep(q), book);
    ms.oracle += Date.now() - t0;

    // B/C: one query over every ZIM, raw order then fused with lexical
    t0 = Date.now();
    const all = await searchAll(prep(q));
    ms.allRaw += Date.now() - t0;
    const t = terms(q);
    const lex = all.map((c, i) => ({ i, k: lexScore(c.title, denoise(c.snip).slice(0, 300), t) }))
      .sort((a, b) => b.k - a.k).map(o => all[o.i].title);
    const fusedTitles = rrf([all.map(c => c.title), lex]);
    const byTitle = Object.fromEntries(all.map(c => [c.title, c]));
    const cRRF = byTitle[fusedTitles[0]];

    // D: search each book separately, then fuse the three ranked lists
    t0 = Date.now();
    const per = {};
    for (const b of Object.values(CROSS_BOOKS)) per[b] = await search(prep(q), b);
    ms.perFused += Date.now() - t0;
    const tagged = Object.entries(per).flatMap(([b, rows]) => rows.map(r => ({ ...r, book: b })));
    const key = c => `${c.book}\u0000${c.title}`;
    const lookup = Object.fromEntries(tagged.map(c => [key(c), c]));
    const perLists = Object.entries(per).map(([b, rows]) => rows.map(r => key({ ...r, book: b })));
    const dFused = lookup[rrf(perLists)[0]];

    const hit = (c, needBook = true) => !!c && want.test(c.title) && (!needBook || c.book === book || c.book === undefined);
    const r = {
      oracle: hit(oracle[0], false),
      allRaw: hit(all[0]),
      allRRF: hit(cRRF),
      perFused: hit(dFused),
    };
    for (const k of Object.keys(sc)) sc[k] += r[k] ? 1 : 0;
    for (const [k, c] of [["allRaw", all[0]], ["allRRF", cRRF], ["perFused", dFused]]) bk[k] += c?.book === book ? 1 : 0;
    console.log(`  oracle:${r.oracle ? "OK" : "x "} all:${r.allRaw ? "OK" : "x "} all+RRF:${r.allRRF ? "OK" : "x "} perFused:${r.perFused ? "OK" : "x "} [${tag}] ${q.slice(0, 34)}`);
    if (!r.allRRF) console.log(`      all+RRF picked [${cRRF?.book?.split("_")[0] ?? "none"}] ${cRRF?.title?.slice(0, 50) ?? "—"} (oracle: ${oracle[0]?.title?.slice(0, 40)})`);
  }
  const n = CROSS_CASES.length;
  console.log(`  right article rank-1 — oracle ${sc.oracle}/${n} | all-books ${sc.allRaw}/${n} | all+RRF ${sc.allRRF}/${n} | per-book fused ${sc.perFused}/${n}`);
  console.log(`  right BOOK rank-1  — all-books ${bk.allRaw}/${n} | all+RRF ${bk.allRRF}/${n} | per-book fused ${bk.perFused}/${n}`);
  console.log(`  search latency per query — oracle(1 req) ${Math.round(ms.oracle / n)}ms | all-books(1 req) ${Math.round(ms.allRaw / n)}ms | per-book(3 req) ${Math.round(ms.perFused / n)}ms`);
}


// F37: split a comparison question into its two sides, model-free. The shared context
// words must go to BOTH sides: "bash versus zsh startup files" split as ["bash",
// "zsh startup files"] retrieved a bash-docs index page for the bare left side, while
// the right side was correct — the tail was doing the work.
const SCAFFOLD = /^(what|whats|which|is|are|the|a|an|should|i|do|does|use|using|prefer|choose|choosing|between|difference|differences|compare|comparison|better|worse|for|to|when|in|on|with|my|of|and|or|about|me|tell|simple|good)$/i;
export function splitCompare(q) {
  const m = q.match(/^(.*?)\bdifference between\b(.+?)\band\b(.+)$/i)
    ?? q.match(/^(.*?)\b(?:versus|vs\.?|or)\b(.+)$/i)?.slice(0, 1).concat(q.match(/^(.*?)\b(?:versus|vs\.?|or)\b(.+)$/i).slice(1));
  if (!m) return null;
  const [left, right] = m.length === 4 ? [m[2], m[3]] : [m[1], m[2]];
  const lw = left.trim().split(/\s+/).filter(w => w && !SCAFFOLD.test(w));
  const rw = right.trim().split(/\s+/).filter(w => w && !SCAFFOLD.test(w));
  if (!lw.length || !rw.length) return null;
  // left side's own words are its core; the right side's first word is its core and
  // everything after it is context shared by both
  const a = lw[lw.length - 1], b = rw[0], tail = rw.slice(1).join(" ");
  return { a, b, tail, qa: `${a} ${tail}`.trim(), qb: `${b} ${tail}`.trim() };
}
// F36/F37: comparison questions. The corpus has no single "X vs Y" article, so either
// the answer must be synthesised from two articles, or `tny` must ask the user which
// side they mean. Both are measured; the trigger for asking must be model-free.
const SYNTH_CASES = [
  ["what is the difference between netctl and NetworkManager", "Netctl", /profile/i, "NetworkManager", /nmcli|nmtui/i],
  ["should I use ext4 or btrfs", "Ext4", /journal/i, "Btrfs", /subvolume|snapshot/i],
  ["bash versus zsh startup files", "Bash", /bashrc/i, "Zsh", /zshrc/i],
  ["systemd-timesyncd or chrony for time sync", "Systemd-timesyncd", /sntp|timesyncd/i, "Chrony", /chrony/i],
  ["iptables or nftables for a simple firewall", "Iptables", /iptables/i, "Nftables", /nft\b|nftables/i],
  ["grub or systemd-boot for booting", "GRUB", /grub-install|grub\.cfg/i, "Systemd-boot", /bootctl/i],
];

// A pair only measures synthesis if BOTH sides' facts survive retrieval; otherwise it
// measures retrieval. Pairs that fail that check are reported and skipped.
export async function buildSynth() {
  const out = [];
  for (const [q, aP, aN, bP, bN] of SYNTH_CASES) {
    try {
      const [aH, bH] = [await article(aP), await article(bP)];
      const a = await pickSections(aH, q, 3, 600), b = await pickSections(bH, q, 3, 600);
      const ok = aN.test(a.text) && bN.test(b.text);
      out.push({ q, aP, aN, bP, bN, aText: a.text, bText: b.text, aFull: html2txt(aH), bFull: html2txt(bH), usable: ok });
    } catch { out.push({ q, aP, bP, usable: false, fetchFailed: true }); }
  }
  return out;
}

export async function benchSynth() {
  console.log("== F36 two-article synthesis, and what one-sided context does ==");
  const sy = (await buildSynth()).filter(c => { if (!c.usable) console.log(`  SKIP ${c.q.slice(0, 44)} — a side's fact did not survive retrieval`); return c.usable; });
  const sc = { both: 0, aOnly: 0, invented: 0, refused: 0 };
  for (const c of sy) {
    // arm 1: both articles in context — can it combine them?
    const two = await ask([
      { role: "system", content: SYS },
      { role: "user", content: `Reference:\n## ${c.aP}\n${c.aText}\n\n## ${c.bP}\n${c.bText}\n\nQuestion: ${c.q}` },
    ], { max_tokens: 220 });
    const gotA = c.aN.test(two.content), gotB = c.bN.test(two.content);
    if (gotA && gotB) sc.both++;

    // arm 2: ONLY side A in context. The B side is absent, so mentioning B's specifics
    // is fabrication — the F26 failure mode, in the shape users actually hit.
    const one = await ask([
      { role: "system", content: SYS_REFUSE },
      { role: "user", content: `Reference:\n## ${c.aP}\n${c.aText}\n\nQuestion: ${c.q}` },
    ], { max_tokens: 220 });
    const leaked = c.bN.test(one.content);
    const declined = /not found/i.test(one.content);
    // F32 reference for commands, F38 `seen` slice for claims about the other side
    const caught = !!ungrounded(one.content, c.aFull, c.q, c.aText);
    if (c.aN.test(one.content)) sc.aOnly++;
    if (leaked && !declined) sc.invented++;
    if (declined || caught) sc.refused++;
    console.log(`  both:${gotA && gotB ? "OK" : `x (A:${gotA ? "y" : "n"} B:${gotB ? "y" : "n"})`} | one-sided: ${declined ? "declined" : leaked ? "INVENTED B" : caught ? "caught by F27" : "answered A only"}  ${c.q.slice(0, 40)}`);
    if (leaked && !declined) console.log(`      ${one.content.slice(0, 170).replace(/\n/g, " ")}`);
  }
  const n = sy.length;
  console.log(`  of ${n}: both sides synthesised ${sc.both} | one-sided arm — invented the missing side ${sc.invented}, declined-or-caught ${sc.refused}`);
  return sy;
}

// F37: the ask-the-user trigger must be model-free and must not fire on normal
// questions. A false trigger is worse than a missed one: it turns a working answer into
// an interrogation. Same asymmetry as F27's false rejects.
//
// First attempt matched a comparison word and then required two retrieved titles to
// contain query terms. That fired on only 4/6, because retrieval for an unsplit
// comparison query surfaces neither side (F35). Splitting first is both simpler and
// strictly better: the two sides come from the question's own grammar.
export async function needsClarify(q) {
  const s = splitCompare(q);
  if (!s) return null;
  const [A, B] = [await searchAll(prep(s.qa), 3), await searchAll(prep(s.qb), 3)];
  const a = A[0], b = B[0];
  // only ask when the two sides genuinely resolve to different articles; if they land
  // on the same one, that article already answers the comparison
  if (!a || !b || a.title === b.title) return null;
  return { ...s, aHit: a, bHit: b };
}

export async function benchClarify() {
  console.log("== F37 ask-the-user trigger: fires on comparisons, silent otherwise ==");
  let fire = 0, quiet = 0;
  for (const [q] of SYNTH_CASES) {
    const c = await needsClarify(q);
    fire += c ? 1 : 0;
    console.log(`  ${c ? "ASK " : "x   "} ${q.slice(0, 44)}${c ? ` -> "${c.aHit.title.slice(0, 22)}" or "${c.bHit.title.slice(0, 22)}"?` : ""}`);
  }
  // "String versus str slice" lives in CROSS_CASES but IS a comparison question, so it
  // is scored as one, not counted as a false trigger.
  const isCompare = q => !!splitCompare(q);
  const normal = [...CROSS_CASES.map(x => x[0]), ...ANSWER_CASES.map(x => x[0]), ...FOLLOWUP_CASES.map(x => `${x[0]} ${x[1]}`)].filter(q => !isCompare(q));
  const falseFires = [];
  for (const q of normal) {
    const c = await needsClarify(q);
    if (c) falseFires.push(`${q} -> ${c.aHit.title} | ${c.bHit.title}`); else quiet++;
  }
  console.log(`  fires on ${fire}/${SYNTH_CASES.length} comparison questions | silent on ${quiet}/${normal.length} normal questions`);
  for (const f of falseFires) console.log(`      FALSE TRIGGER: ${f.slice(0, 100)}`);
  console.log(`  (${[...CROSS_CASES.map(x => x[0]), ...ANSWER_CASES.map(x => x[0])].filter(isCompare).length} question(s) excluded from "normal" as genuine comparisons)`);
}

// F31: section selection, embedding vs model-free lexical, at equal splitting and
// windowing. Deterministic — no model, no sampling — so one run is the answer.
// Reports context cost, because that is what the two arms actually trade.
export async function benchSelect() {
  console.log("== F31 section selection: embedding top-3 vs lexical top-3/top-5 ==");
  const cases = [
    ...ANSWER_CASES.map(([q, path, needle]) => [q, path, needle, "F22"]),
    ...FOLLOWUP_CASES.map(([q1, q2, path, needle]) => [`${q1} ${q2}`, path, needle, "F28"]),
    ["disable root login over ssh", "OpenSSH", /PermitRootLogin/i, "large"],
    ["how do I stop root logging in", "OpenSSH", /PermitRootLogin/i, "large"],
  ];
  const sc = { emb: 0, lex3: 0, lex5: 0 }, ch = { emb: 0, lex3: 0, lex5: 0 };
  for (const [q, path, needle, tag] of cases) {
    const html = await article(path);
    const arms = {
      emb: await pickSections(html, q, 3, 600),
      lex3: pickSectionsLex(html, q, 3, 600),
      lex5: pickSectionsLex(html, q, 5, 600),
    };
    const hit = {};
    for (const k of Object.keys(arms)) {
      hit[k] = needle.test(arms[k].text);
      sc[k] += hit[k] ? 1 : 0;
      ch[k] += arms[k].text.length;
    }
    console.log(`  emb:${hit.emb ? "OK" : "x "} lex3:${hit.lex3 ? "OK" : "x "} lex5:${hit.lex5 ? "OK" : "x "} [${tag}] ${q.slice(0, 44)}`);
    if (!hit.emb) console.log(`      emb picked §${arms.emb.heads.join(" | §")}`);
  }
  const n = cases.length;
  for (const k of ["emb", "lex3", "lex5"]) console.log(`  ${k.padEnd(5)} ${sc[k]}/${n}  avg ctx ${Math.round(ch[k] / n)}ch`);
  console.log("  (embedding wins per token: same score as lex5 in ~39% less context)");
}

async function benchJudge() {
  console.log("== F16 model-as-judge (expected: worse than xapian rank-1) ==");
  let judge = 0, rank1 = 0, n = 0;
  const picks = [];
  for (const [q, want] of RANK_CASES.slice(0, 6)) {
    const c = await search(prep(q));
    if (!c.length) continue;
    n++;
    const list = c.map((x, i) => `${i + 1}. ${x.title} — ${x.snip.slice(0, 140)}`).join("\n");
    const { content } = await ask([
      { role: "system", content: "Pick the article that best answers the question. Reply with only its number." },
      { role: "user", content: `Question: ${q}\n\nArticles:\n${list}\n\nBest article number:` },
    ], { max_tokens: 6 });
    const pick = +((content.match(/\d+/) ?? [])[0] ?? 0);
    picks.push(pick);
    judge += want.test(c[pick - 1]?.title ?? "") ? 1 : 0;
    rank1 += want.test(c[0].title) ? 1 : 0;
  }
  console.log(`  of ${n}: judge ${judge} | xapian rank-1 ${rank1} | picks ${picks.join(",")}`);
  console.log("  (350M emitted a near-constant index; that is the F16 signature)");
}

// F39: the within-book ranking wall. F34's 4 misses were identical across every arm —
// oracle, all-books, RRF and per-book fusion all pick `Netboot` for "mount a usb drive
// automatically". Title+snippet is all those arms ever see. This promotes *section*
// evidence into article ranking: fetch the top-k candidates and score their best
// section, which is the signal that fixed extraction (F31).
export async function benchRerank(k = 5) {
  console.log(`== F39 article rerank by section evidence (top-${k} refetched) ==`);
  const cases = [
    ...RANK_CASES.map(([q, want]) => [q, want, "F17"]),
    ...CROSS_CASES.map(([q, , want]) => [q, want, "F34"]),
  ];
  const sc = { base: 0, sect: 0, full: 0, fused: 0 };
  let ms = 0, fetches = 0;
  for (const [q, want, tag] of cases) {
    const cands = await searchAll(prep(q), 8);
    if (!cands.length) { console.log(`  —   ${q.slice(0, 40)}: no candidates`); continue; }
    const baseOrder = rankArticles(q, cands).fused;
    const t = terms(q);
    const t0 = Date.now();
    const scored = [];
    for (const title of baseOrder.slice(0, k)) {
      const c = cands.find(x => x.title === title);
      if (!c) continue;
      let html = "";
      try { html = await article(c.path, c.book); fetches++; } catch { continue; }
      const txt = html2txt(html);
      const secs = sectionsOf(html);
      // best single section beats whole-article density: a long article accumulates
      // term hits everywhere without any one passage answering the question
      const best = secs.length ? Math.max(...secs.map(s => lexScore(s.head, s.text, t))) : 0;
      scored.push({ title, best, dens: lexScore(c.title, txt.slice(0, 20000), t) });
    }
    ms += Date.now() - t0;
    const bySect = [...scored].sort((a, b) => b.best - a.best).map(o => o.title);
    const byFull = [...scored].sort((a, b) => b.dens - a.dens).map(o => o.title);
    const fused = rrf([baseOrder, bySect]);
    const hit = { base: want.test(baseOrder[0] ?? ""), sect: want.test(bySect[0] ?? ""), full: want.test(byFull[0] ?? ""), fused: want.test(fused[0] ?? "") };
    for (const key of Object.keys(sc)) sc[key] += hit[key] ? 1 : 0;
    if (hit.base !== hit.sect) {
      console.log(`  ${hit.sect ? "FIXED" : "BROKE"} [${tag}] ${q.slice(0, 38)}`);
      console.log(`      base -> ${baseOrder[0]?.slice(0, 34)} | section-ranked -> ${bySect[0]?.slice(0, 34)}`);
    }
  }
  const n = cases.length;
  console.log(`  of ${n}: base(title+snippet) ${sc.base} | best-section ${sc.sect} | whole-article density ${sc.full} | RRF(base,section) ${sc.fused}`);
  console.log(`  cost: ${fetches} article fetches, ${Math.round(ms / n)}ms extra per query`);
}

// F39b: article ranking cannot be improved (all four section-evidence formulations
// scored worse than title+snippet), but recall@3 is 24/25 against recall@1's 21/25 —
// the right article is usually rank 2 or 3. So widen instead of rerank: spread the same
// context budget over the top-3 articles. 3 articles x 1 section is 1,842 ch versus
// 1 article x 3 sections at 1,603 ch, and it contains the answer the narrow one missed.
export async function benchWiden() {
  console.log("== F39b widen retrieval: 1 article x 3 sections vs 3 articles x 1 ==");
  const sc = { narrow: 0, wide: 0 }, ch = { narrow: 0, wide: 0 }, lat = { narrow: 0, wide: 0 };
  for (const [q, path, needle] of ANSWER_CASES) {
    const cands = await searchAll(prep(q), 8);
    const order = rankArticles(q, cands).fused;
    const build = async (nArt, per) => {
      const parts = [];
      for (const title of order.slice(0, nArt)) {
        const c = cands.find(x => x.title === title);
        if (!c) continue;
        const s = await pickSections(await article(c.path, c.book), q, per, 600);
        parts.push(`## ${title}\n${s.text}`);
      }
      return parts.join("\n\n");
    };
    const narrow = await build(1, 3), wide = await build(3, 1);
    for (const [k, ctx] of [["narrow", narrow], ["wide", wide]]) {
      ch[k] += ctx.length;
      const t0 = Date.now();
      const { content } = await ask([
        { role: "system", content: SYS },
        { role: "user", content: `Reference:\n${ctx}\n\nQuestion: ${q}` },
      ]);
      lat[k] += Date.now() - t0;
      const ok = needle.test(content);
      sc[k] += ok ? 1 : 0;
      if (k === "wide") console.log(`  narrow:${needle.test(narrow) ? "ctx" : "—  "} wide:${needle.test(wide) ? "ctx" : "—  "} | answer narrow:${sc.narrow > 0 ? "" : ""}${ok ? "OK" : "x "} ${q.slice(0, 34)}`);
    }
  }
  const n = ANSWER_CASES.length;
  console.log(`  answers — 1 article x3 sections ${sc.narrow}/${n} @ ${Math.round(ch.narrow / n)}ch, ${(lat.narrow / n / 1000).toFixed(1)}s | 3 articles x1 ${sc.wide}/${n} @ ${Math.round(ch.wide / n)}ch, ${(lat.wide / n / 1000).toFixed(1)}s`);
}

// F41: local files. `tny "summarize src/main.rs"` is in the CLI surface with no
// measurement behind it. Code has no <h2> headings, so F31's section split does not
// apply — the only reusable primitive is the term-centred window. A 40 KB source file
// is ~11k tokens, past the 8192 context, so selection is mandatory, not an optimisation.
const FILE_CASES = [
  ["bench/harness.mjs", "which port does the embedding server use", /8084/],
  ["bench/harness.mjs", "what does the ungrounded function reject", /command|restat|question/i],
  ["bench/harness.mjs", "how many sections does pickSections take by default", /\b3\b|three/i],
  ["bench/harness.mjs", "what does splitCompare return for a non-comparison question", /null/i],
  ["PLAN.md", "how many rust files will the implementation have", /\bfour\b|\b4\b/i],
  ["PLAN.md", "which model answers questions", /0\.8B|qwen/i],
];

// three ways to fit a file into the budget, mirroring F31's arms
export function fileWindows(text, q, budget = 1800, n = 3) {
  const t = terms(q);
  const head = text.slice(0, budget);
  const one = window(text, t, budget);
  const size = Math.max(400, Math.floor(budget / n));
  const fixed = [];
  for (let i = 0; i < text.length; i += size) fixed.push({ at: i, text: text.slice(i, i + size) });
  const pick = chunks => chunks.map(c => ({ ...c, k: lexScore("", c.text, t) }))
    .sort((a, b) => b.k - a.k).slice(0, n).sort((a, b) => a.at - b.at)
    .map(c => c.text.trim()).join("\n…\n");
  // same chunk cap as the fixed arm, so `struct` vs `many` isolates boundary quality
  // rather than context volume
  return { head, one, many: pick(fixed), struct: pick(structChunks(text, size)) };
}

// F41: fixed-size chunks cut mid-function and mid-table. The excerpt then contains the
// answer but not coherently, and answering scored 3/6 against selection's 6/6. Split on
// structure instead — markdown headings, or a line starting a top-level declaration —
// which is what makes the ZIM path work (F31).
export function structChunks(text, max = 1200) {
  const lines = text.split("\n");
  const isBoundary = l => /^#{1,6}\s/.test(l) || /^(export |function |class |const |async function |\/\/ -+|## )/.test(l);
  const out = [];
  let cur = [], at = 0, pos = 0;
  for (const l of lines) {
    const len = l.length + 1;
    if (cur.length && isBoundary(l) && cur.join("\n").length > 200) {
      out.push({ at, text: cur.join("\n") });
      cur = []; at = pos;
    }
    cur.push(l);
    pos += len;
    if (cur.join("\n").length > max) { out.push({ at, text: cur.join("\n") }); cur = []; at = pos; }
  }
  if (cur.length) out.push({ at, text: cur.join("\n") });
  return out.filter(c => c.text.trim().length > 40);
}

// F42: piped input — `tny "what is wrong" < paste.txt`. Different from F41: a paste is
// small enough to send whole, so there is no selection problem, and the paste itself is
// the grounding reference. The question is whether 0.8B can diagnose real tool output.
//
// Needles are deliberately strict after F41, where /command|restat|question/ was
// satisfied by "rejects command, restat, and question questions". Each needle demands a
// specific identifier or fix that cannot be produced by echoing the question.
const STDIN_CASES = [
  ["what is wrong here", `error[E0502]: cannot borrow \`v\` as mutable because it is also borrowed as immutable
 --> src/main.rs:4:5
  |
3 |     let first = &v[0];
  |                  - immutable borrow occurs here
4 |     v.push(4);
  |     ^^^^^^^^^ mutable borrow occurs here
5 |     println!("{first}");
  |               ------- immutable borrow later used here`,
    /first|immutable borrow|&v\[0\]/i, "rust borrow error"],

  ["why did this fail", `$ systemctl start nginx
Job for nginx.service failed because the control process exited with error code.
See "systemctl status nginx.service" and "journalctl -xeu nginx.service" for details.
$ journalctl -xeu nginx.service | tail -3
nginx: [emerg] bind() to 0.0.0.0:80 failed (98: Address already in use)
nginx: configuration file /etc/nginx/nginx.conf test failed`,
    /already in use|port 80|:80|another process/i, "port conflict"],

  ["what is wrong here", `$ ssh deploy@prod
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
The RSA host key for prod has changed,
and the key for the corresponding host is unknown.
Offending key in /home/rob/.ssh/known_hosts:14`,
    /known_hosts|host key|line 14/i, "changed host key"],

  ["explain this failure", `FAIL  test/auth.test.ts > refresh token rotation
AssertionError: expected 401 to be 200
  - Expected  200
  + Received  401
   at test/auth.test.ts:88:24
   at Module.runTest (node_modules/vitest/dist/chunk.js:120:9)`,
    /401|unauthor|token|expected 200/i, "test assertion"],

  ["what does this say about disk usage", `Filesystem      Size  Used Avail Use% Mounted on
/dev/mapper/root  220G  209G  1.2G  99% /
/dev/nvme0n1p1    511M  312M  199M  62% /boot
tmpfs             3.9G  1.1M  3.9G   1% /run`,
    /99%|1\.2G|root|nearly full|out of space/i, "df output"],

  ["why is the container restarting", `CONTAINER ID   IMAGE          STATUS                          NAMES
9f2c1a4b7e88   api:latest     Restarting (137) 3 seconds ago   api
1b3d5f7a9c02   postgres:16    Up 2 hours (healthy)             db
$ docker logs api | tail -2
FATAL: could not connect to database: connection refused
Killed`,
    /137|out of memory|oom|memory limit|killed/i, "OOM exit 137"],
];

// A paste needs no retrieval, so this measures the model on tool output directly. The
// paste is also the grounding reference, so F27 applies unchanged.
export async function benchStdin() {
  console.log("== F42 piped input: diagnosing real tool output ==");
  let ok = 0, t = 0, falseReject = 0, chars = 0;
  for (const [q, paste, needle, label] of STDIN_CASES) {
    chars += paste.length;
    const t0 = Date.now();
    const { content } = await ask([
      { role: "system", content: "Diagnose the pasted output. Be concise: at most two sentences naming the specific cause." },
      { role: "user", content: `${paste}\n\nQuestion: ${q}` },
    ]);
    t += Date.now() - t0;
    const hit = needle.test(content);
    ok += hit ? 1 : 0;
    const ug = ungrounded(content, paste, q);
    if (hit && ug) falseReject++;
    console.log(`  ${hit ? "OK" : "x "} ${label}${ug ? ` [F27: ${ug}]` : ""}\n      ${content.slice(0, 140).replace(/\n/g, " ")}`);
  }
  const n = STDIN_CASES.length;
  console.log(`  ${ok}/${n} diagnosed | ${(t / n / 1000).toFixed(1)}s per answer | avg paste ${Math.round(chars / n)}ch | F27 false rejects ${falseReject}/${ok}`);
}

// Selection is deterministic and free; only answering costs a model call. Measuring all
// three arms with the model would be 18 calls, and one call here is ~32 s because ~500
// uncached prompt tokens of prefill dominate. So: score every arm model-free, then
// spend 6 calls on the arm that wins.
export async function benchFile() {
  console.log("== F41 local file reading: how to fit a source file into context ==");
  const arms = ["head", "one", "many", "struct"];
  const ctxHit = { head: 0, one: 0, many: 0, struct: 0 };
  const rows = [];
  for (const [path, q, needle] of FILE_CASES) {
    const text = await Bun.file(`/home/rob/Projects/Personal/tny/${path}`).text();
    const w = fileWindows(text, q);
    for (const arm of arms) ctxHit[arm] += needle.test(w[arm]) ? 1 : 0;
    rows.push({ path, q, needle, w, kb: text.length / 1024 });
    console.log(`  ${arms.map(a => `${a}:${needle.test(w[a]) ? "OK" : "x "}`).join(" ")} ${path} "${q.slice(0, 40)}" (${(text.length / 1024).toFixed(0)} KB)`);
  }
  const n = FILE_CASES.length;
  console.log(`  answer present in excerpt — ${arms.map(a => `${a} ${ctxHit[a]}/${n}`).join(" | ")}`);
  // tie goes to `struct`: equal presence, but coherent boundaries — that is the thing
  // being tested, since fixed chunks scored 6/6 presence yet only 3/6 answers
  const best = arms.reduce((a, b) => (ctxHit[b] > ctxHit[a] ? b : ctxHit[b] === ctxHit[a] && b === "struct" ? b : a));
  console.log(`  best selection: ${best} — answering with that arm only (${n} model calls)`);
  let ok = 0, t = 0;
  for (const r of rows) {
    const t0 = Date.now();
    const { content } = await ask([
      { role: "system", content: "Answer the question using the file excerpt. Be concise: at most two sentences." },
      { role: "user", content: `File ${r.path}:\n${r.w[best]}\n\nQuestion: ${r.q}` },
    ]);
    t += Date.now() - t0;
    const hit = r.needle.test(content);
    ok += hit ? 1 : 0;
    console.log(`  ${hit ? "OK" : "x "} ${r.q.slice(0, 44)}\n      ${content.slice(0, 130).replace(/\n/g, " ")}`);
  }
  console.log(`  answers with ${best} excerpt: ${ok}/${n} | ${(t / n / 1000).toFixed(1)}s per answer`);
}

async function benchSections() {
  console.log("== F22 section selection (expect answer present 6/6) ==");
  const ctx = await buildContexts();
  for (const c of ctx) console.log(`  ${c.has ? "OK" : "x "} ${c.q} [${c.text.length}ch] §${c.heads.join(" | §")}`);
  console.log(`  answer present: ${ctx.filter(c => c.has).length}/${ctx.length}`);
  return ctx;
}

async function benchAnswers(ctx) {
  console.log("== F20 answering (expect Qwen 5/6, 350M 2/6) ==");
  ctx ??= await buildContexts();
  let ok = 0, t = 0, falseReject = 0;
  for (const c of ctx) {
    const t0 = Date.now();
    const { content, error } = await ask([
      { role: "system", content: SYS },
      { role: "user", content: `Reference:\n${c.text}\n\nQuestion: ${c.q}` },
    ]);
    t += Date.now() - t0;
    const hit = c.needle.test(content);
    const ug = ungrounded(content, c.full ?? c.text, c.q); // F27/F32: source doc, not slice
    ok += hit ? 1 : 0;
    if (hit && ug) falseReject++;
    console.log(`  ${hit ? "OK" : "x "} ${c.q}${error ? ` [${error}]` : ""}${ug ? ` [F27 would reject: ${ug}]` : ""}\n      ${content.slice(0, 150).replace(/\n/g, " ")}`);
  }
  console.log(`  ${ok}/${ctx.length} correct | ${(t / ctx.length / 1000).toFixed(1)}s per answer | F27 false rejects ${falseReject}/${ok}`);
}

async function benchCorpus() {
  console.log("== F21 corpus lift (expect 350M 1->2/6, Qwen 3->5/6) ==");
  const ctx = await buildContexts();
  for (const grounded of [false, true]) {
    let ok = 0;
    for (const c of ctx) {
      const msgs = grounded
        ? [{ role: "system", content: SYS }, { role: "user", content: `Reference:\n${c.text}\n\nQuestion: ${c.q}` }]
        : [{ role: "system", content: NOCTX }, { role: "user", content: c.q }];
      const { content } = await ask(msgs);
      const hit = c.needle.test(content);
      ok += hit ? 1 : 0;
      if (!grounded && !hit) console.log(`     unretrieved miss: ${c.q} -> ${content.slice(0, 90).replace(/\n/g, " ")}`);
    }
    console.log(`  ${grounded ? "WITH" : "WITHOUT"} corpus: ${ok}/${ctx.length}`);
  }
}

// F19: proves the empty answers are real, not a parsing artifact.
// Needs the chat server started with --reasoning-format none.
async function benchThinking() {
  console.log("== F19 thinking mode (needs --reasoning-format none) ==");
  const ctx = await buildContexts();
  const c = ctx[0];
  for (const max_tokens of [512, 2048]) {
    const t0 = Date.now();
    const r = await fetch(`${CHAT}/v1/chat/completions`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [{ role: "system", content: SYS }, { role: "user", content: `Reference:\n${c.text}\n\nQuestion: ${c.q}` }],
        temperature: 0.1, top_k: 50, max_tokens,
      }),
    });
    const j = await r.json(), m = j.choices?.[0];
    const body = m?.message?.content ?? "";
    const closed = /<\/think>/.test(body);
    console.log(`  max_tokens=${max_tokens}: finish=${m?.finish_reason} gen=${j.usage?.completion_tokens} in ${((Date.now() - t0) / 1000).toFixed(1)}s closed</think>=${closed}`);
    if (closed) console.log(`     answer: ${body.split("</think>").pop().trim().slice(0, 160)}`);
  }
}
// F26: fabrication safety. Each question is paired with a DIFFERENT question's
// sections, so the answer is absent by construction. Declining is the only correct
// behaviour; anything else is a fabrication served as fact.
const SYS_REFUSE = SYS + " If the reference material does not contain the answer, reply exactly: not found.";

async function benchRefuse(ctx) {
  console.log("== F26 refusal on mismatched context (want: refuse all, fabricate none) ==");
  ctx ??= await buildContexts();
  let refused = 0, safe = 0, n = 0, t = 0;
  for (let i = 0; i < ctx.length; i++) {
    const q = ctx[i].q, c = ctx[(i + 1) % ctx.length];
    if (ctx[i].needle.test(c.text)) continue; // skip if the answer leaked in anyway
    n++;
    const t0 = Date.now();
    const { content } = await ask([
      { role: "system", content: SYS_REFUSE },
      { role: "user", content: `Reference:\n${c.text}\n\nQuestion: ${q}` },
    ]);
    t += Date.now() - t0;
    const no = /not found/i.test(content);
    // F32: the mismatched context came from c's article, so that article is the source
    const ug = ungrounded(content, c.full ?? c.text, q);
    refused += no ? 1 : 0;
    safe += (no || ug) ? 1 : 0; // F27 catches what the model failed to decline
    console.log(`  ${no ? "OK" : ug ? "F27" : "x  "} ${q}  [ctx §${c.heads[0]}]${ug && !no ? ` -> caught: ${ug}` : ""}\n      ${content.slice(0, 140).replace(/\n/g, " ")}`);
  }
  console.log(`  model refused ${refused}/${n} | with F27 grounding check ${safe}/${n} | ${(t / n / 1000).toFixed(1)}s per answer`);
}

// F28: does a follow-up turn need conversation history, or is a re-retrieved
// reference enough? Two arms on identical contexts. The no-history arm is what a
// stateless CLI does; if it ties, history is dead weight.
export async function benchFollowup(fu) {
  console.log("== F28 follow-up turns: with history vs without ==");
  fu ??= await buildFollowups();
  const sc = { hist: 0, plain: 0 }, lat = { hist: 0, plain: 0 };
  let n = 0, falseReject = 0;
  for (const c of fu) {
    if (!c.has) { console.log(`  SKIP ${c.q2}: needle absent from context`); continue; }
    n++;
    const t0 = Date.now();
    const a1 = await ask([
      { role: "system", content: SYS },
      { role: "user", content: `Reference:\n${c.ref1}\n\nQuestion: ${c.q1}` },
    ]);
    const t1 = Date.now();
    const withHist = await ask([
      { role: "system", content: SYS },
      { role: "user", content: `Reference:\n${c.ref1}\n\nQuestion: ${c.q1}` },
      { role: "assistant", content: a1.content },
      { role: "user", content: `Reference:\n${c.ref2}\n\nQuestion: ${c.q2}` },
    ]);
    lat.hist += Date.now() - t1;
    const t2 = Date.now();
    const noHist = await ask([
      { role: "system", content: SYS },
      { role: "user", content: `Reference:\n${c.ref2}\n\nQuestion: ${c.q1} — specifically: ${c.q2}` },
    ]);
    lat.plain += Date.now() - t2;
    const okH = c.needle.test(withHist.content), okP = c.needle.test(noHist.content);
    sc.hist += okH ? 1 : 0;
    sc.plain += okP ? 1 : 0;
    // F32: reference is the source document, not the ~1.5 KB slice. The slice rejected
    // a correct answer for citing `cryptsetup`, which the Dm-crypt article does contain.
    const ref = c.full ?? `${c.ref1}\n${c.ref2}`;
    for (const [arm, ok, a] of [["hist", okH, withHist.content], ["plain", okP, noHist.content]]) {
      const why = ok && ungrounded(a, ref, c.q2);
      if (why) {
        falseReject++;
        console.log(`      FALSE REJECT [${arm}] [${why}]`);
      }
    }
    console.log(`  hist:${okH ? "OK" : "x "} plain:${okP ? "OK" : "x "} ${c.q2}  [${c.kind} article, turn1 ${((t1 - t0) / 1000).toFixed(1)}s]`);
    console.log(`      hist:  ${withHist.content.slice(0, 130).replace(/\n/g, " ")}`);
    console.log(`      plain: ${noHist.content.slice(0, 130).replace(/\n/g, " ")}`);
  }
  // invariant: a false reject requires a correct answer, so "0 correct + N rejects" is
  // impossible. An edit once clobbered the two score lines above and this benchmark
  // reported exactly that for 2B — a model verdict that was purely my bug.
  if (falseReject > sc.hist + sc.plain) throw new Error(`benchFollowup: ${falseReject} false rejects but only ${sc.hist + sc.plain} correct answers — scoring is broken`);
  console.log(`  of ${n}: with history ${sc.hist} @ ${(lat.hist / n / 1000).toFixed(1)}s | without ${sc.plain} @ ${(lat.plain / n / 1000).toFixed(1)}s | F27 false rejects ${falseReject}`);
  return fu;
}

// F29: how to build the search query for an elliptical follow-up. Model-free
// concatenation is the lazy candidate; a model rewrite is the expensive one.
export async function benchRewrite(fu) {
  console.log("== F29 follow-up query construction (article + section retrieval) ==");
  fu ??= await buildFollowups();
  const sc = { raw: 0, concat: 0, rewrite: 0 }, sec = { raw: 0, concat: 0, rewrite: 0 };
  let n = 0, ms = 0;
  for (const c of fu) {
    n++;
    const t0 = Date.now();
    const { content } = await ask([
      { role: "system", content: "Rewrite the follow-up as a standalone search query. Reply with only the query." },
      { role: "user", content: `Earlier question: ${c.q1}\nFollow-up: ${c.q2}\n\nStandalone query:` },
    ], { max_tokens: 40 });
    ms += Date.now() - t0;
    const rewrite = content.trim().replace(/^["']|["']$/g, "").split("\n")[0];
    const qs = { raw: c.q2, concat: `${c.q1} ${c.q2}`, concat_: null, rewrite };
    delete qs.concat_;
    const html = await article(c.path);
    for (const k of Object.keys(qs)) {
      const cands = await search(prep(qs[k]));
      sc[k] += cands.length && cands[0].title.toLowerCase().includes(c.path.split("/")[0].toLowerCase().slice(0, 6)) ? 1 : 0;
      const s = await pickSections(html, qs[k], 3, 600);
      sec[k] += c.needle.test(s.text) ? 1 : 0;
    }
    console.log(`  ${c.q2}\n      rewrite: "${rewrite}"`);
  }
  console.log(`  of ${n} — right article rank-1: raw ${sc.raw} | concat ${sc.concat} | rewrite ${sc.rewrite}`);
  console.log(`  of ${n} — answer in top-3 sections: raw ${sec.raw} | concat ${sec.concat} | rewrite ${sec.rewrite} (rewrite costs ${(ms / n / 1000).toFixed(1)}s/turn)`);
}

// to "stop root logging in" sits in §Protection — rank 11 for a good query, 36 for
// the elliptical one. top-3 cannot reach it; this measures what k would.
export async function benchDepth() {
  console.log("== F30 section depth on large articles ==");
  const CASES = [
    ["OpenSSH", "disable root login over ssh", /PermitRootLogin/i],
    ["OpenSSH", "how do I stop root logging in", /PermitRootLogin/i],
    ["Systemd", "how do I see only the failed ones", /--failed/i],
    ["Dm-crypt/System_configuration", "how do I unlock it at boot", /crypttab|keyfile/i],
  ];
  for (const [path, q, want] of CASES) {
    const html = await article(path);
    const total = sectionsOf(html).length;
    const row = [];
    for (const k of [3, 5, 8]) {
      const s = await pickSections(html, q, k, 600);
      row.push(`top-${k}:${want.test(s.text) ? "OK" : "x "}(${s.text.length}ch)`);
    }
    console.log(`  ${path} [${total} sections] "${q}"\n      ${row.join("  ")}`);
  }
}

// F27 self-check: pure, no servers, no network. The rules are load-bearing for
// safety, so they get the one runnable test in this file. Exits non-zero on failure.
const DU = "du alternatives. Use du -h to inspect usage, or ncdu for an interactive view.";
const SWAP = "Swap file creation. Use mkswap to set up the file, then swapon to enable it.";
const TZ = "Set system clock. Use timedatectl set-timezone to change the timezone.";
const SSH = "Generating an SSH key pair. Run ssh-keygen; the key is saved under ~/.ssh by default.";
// F32: the reference must be the source document, not the slice sent to the model. The
// live failure: a correct "unlock it at boot" answer cited `cryptsetup`, which the
// Dm-crypt article contains but the 1.5 KB windowed slice did not.
const SLICE = "Unlocking in early userspace. Add the device to /etc/crypttab with its UUID, then rebuild the initramfs.";
const DOC = SLICE + " Encrypting devices with LUKS mode. Run cryptsetup luksFormat /dev/sdX to create the container.";
const GROUND_CASES = [
  [0, DU, "check what is using disk space", "Use `du -h` to see what is using disk space.", "short command survives"],
  [0, DU, "check what is using disk space", "Use `du` or `ncdu` to check disk usage.", "two commands"],
  [0, SWAP, "create a swap file", "Create a swap file using `mkswap`, then activate it with `swapon`.", "grounded prose"],
  [0, TZ, "set the system timezone", "# timedatectl set-timezone", "prompt line, no backticks"],
  [0, TZ, "set the system timezone", "```bash\ntimedatectl set-timezone Europe/London\n```", "fenced block"],
  [0, TZ, "set the system timezone", "Use timedatectl set-timezone to set the timezone.", "unmarked command"],
  [0, DU, "create a swap file", "not found", "bare refusal"],
  [0, DU, "create a swap file", "The reference does not contain the answer. not found.", "sentence refusal"],
  [0, SSH, "generate an ssh key", "Generate a key with `ssh-keygen` and save it to `/home/username/.ssh/id_ed25519`.", "quoted path is not a command"],
  [0, DOC, "unlock it at boot", "Add the device to `/etc/crypttab`, created earlier with `cryptsetup`.", "F32: command elsewhere in the source document is grounded"],
  [1, SLICE, "unlock it at boot", "Add the device to `/etc/crypttab`, created earlier with `cryptsetup`.", "F32: same answer IS ungrounded against the slice alone"],
  [1, SWAP, "generate an ssh key", "Generate an SSH key using `ssh-keygen -t rsa -b 4096`.", "parametric leak"],
  [1, DU, "mount a usb drive automatically", "```\nmkfs.ext4 /dev/sdb1\n```", "destructive, fenced"],
  [1, DU, "mount a usb drive automatically", "# mkfs.ext4 /dev/sdb1", "destructive, prompt line"],
  [1, DU, "create a swap file", "create a swap file", "echo degeneration"],
  [1, DU, "create a swap file", "Create a swap file.", "echo with punctuation"],
  [1, DU, "create a swap file", "swap file", "partial echo"],
  [1, DU, "check what is using disk space", "Which tool is currently consuming the most disk space?", "asks a question back"],
  // F38: recorded verbatim from the one-sided synthesis arm, where the reference held
  // only the first tool. Both were confident prose about a tool that was never
  // retrieved, and the command rule could not see them.
  [1, "Systemd-timesyncd is an SNTP client. Enable it with systemctl enable systemd-timesyncd.",
    "systemd-timesyncd or chrony for time sync",
    "chrony is the recommended alternative to systemd-timesyncd for time synchronization in most systems.",
    "F38: asserts about the unretrieved side"],
  [1, "Iptables is a firewall tool. Rules are managed with the iptables command.",
    "iptables or nftables for a simple firewall",
    "iptables is the recommended tool for a simple firewall, whereas nftables is primarily designed for complex network configurations.",
    "F38: comparative claim about the absent side"],
  [0, "Systemd-timesyncd is an SNTP client. Enable it with systemctl enable systemd-timesyncd.",
    "systemd-timesyncd or chrony for time sync",
    "Systemd-timesyncd is an SNTP client; enable it with `systemctl enable systemd-timesyncd`.",
    "F38: answering only the retrieved side is fine"],
  [0, "Chrony docs. Chrony is a full NTP implementation; systemd-timesyncd is SNTP only.",
    "systemd-timesyncd or chrony for time sync",
    "Chrony is a full NTP implementation, while systemd-timesyncd is an SNTP client only.",
    "F38: both sides present in the reference is fine"],
];

function benchGround() {
  console.log("== F27 grounding check self-test (pure, no servers) ==");
  let bad = 0;
  for (const [mustCatch, ref, q, ans, label] of GROUND_CASES) {
    const r = ungrounded(ans, ref, q);
    const ok = !!r === !!mustCatch;
    if (!ok) bad++;
    console.log(`  ${ok ? "ok  " : "FAIL"} ${mustCatch ? "catch" : "allow"}: ${label}${r ? ` -> ${r}` : ""}`);
  }
  if (bad) process.exitCode = 1;
}


// importable: only dispatch when run directly, so helpers can be reused ad hoc
if (import.meta.main) {
  const cmd = process.argv[2] ?? "all";
  if (cmd === "rank") await benchRank();
  else if (cmd === "judge") await benchJudge();
  else if (cmd === "sections") await benchSections();
  else if (cmd === "answers") await benchAnswers();
  else if (cmd === "corpus") await benchCorpus();
  else if (cmd === "thinking") await benchThinking();
  else if (cmd === "refuse") await benchRefuse();
  else if (cmd === "rerank") await benchRerank();
  else if (cmd === "widen") await benchWiden();
  else if (cmd === "file") await benchFile();
  else if (cmd === "stdin") await benchStdin();
  else if (cmd === "followup") await benchFollowup();
  else if (cmd === "rewrite") await benchRewrite();
  else if (cmd === "depth") await benchDepth();
  else if (cmd === "select") await benchSelect();
  else if (cmd === "cross") await benchCross();
  else if (cmd === "synth") await benchSynth();
  else if (cmd === "clarify") await benchClarify();
  else if (cmd === "ground") benchGround();
  else if (cmd === "all") {
    benchGround();
    await benchSelect();
    await benchRank();
    await benchCross();
    const ctx = await benchSections();
    await benchAnswers(ctx);
    await benchJudge();
    await benchCorpus();
    await benchRefuse(ctx);
    const fu = await benchFollowup();
    await benchRewrite(fu);
    await benchDepth();
  } else {
    console.error("usage: bun bench/harness.mjs [rank|judge|sections|answers|corpus|refuse|ground|followup|rewrite|depth|select|thinking|all]");
    process.exit(2);
  }
}
