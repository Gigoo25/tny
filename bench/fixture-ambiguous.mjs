// Cross-domain collision fixture: queries whose *words* live in several corpora but whose
// intended sense has exactly one right source. Companion to fixture-instructional.mjs and
// fixture-qa.mjs, which are entirely computing-focused; this file is the mirror measurement —
// five ambiguous terms asked twice, once in the everyday sense and once in the computing
// sense, so the two directions can be compared on the same term.
//
// Format, identical to the sibling fixtures: [query, intent, bookStem, titleRe, needleRe].
// Every case verified live: searchBook(prep(query), bookStem, 8) returned the article,
// article() + html2txt() contained the needle, and the surrounding paragraph was read to
// confirm it answers the question. titleRe matches exactly one of the 8 hits.
//
// Terms:  kernel · shell · cookie · python · memory      (5 everyday sense, 5 computing sense)
// Books:  wikipedia_en_top_nopic_2026-06       5   (875,265 articles — general knowledge)
//         wikipedia_en_computer_nopic_2026-06  5   (244,179 — computing)
//
// MEASURED, and the reason this file exists (trailing comments carry the per-case numbers):
//
// 1. The everyday sense is *not* contested by the computing corpora — it is contested by
//    NOISE. "kernel corn" against the computing Wikipedia returns Asus ZenFone / Google
//    Pixel (Android kernel version tables), and the 5,032-article wikipedia_en_100 book
//    heads the union with Native Americans in the United States. No computing article about
//    kernels competes. The failure mode for everyday queries is a stopword-ish term
//    ("corn", "made", "biology") pulling unrelated books in, not sense confusion.
//
// 2. The computing sense is contested by DUPLICATION, not by the everyday sense. All five
//    computing articles below also sit at per-book rank 0 in the 875k general corpus
//    (byte-differing dumps of the same page: Kernel (operating system) 67,767 vs 67,384
//    chars, Shell (computing) 17,724 vs 17,570, HTTP cookie 71,814 vs 71,542, Python
//    (programming language) 87,348 vs 87,695, Virtual memory 35,050 vs 34,774). So for
//    cases 6-10 a book-level miss against the general corpus is a duplicate, not a wrong
//    answer; only a miss on some *other* title is a real failure.
//
// 3. Union interleaving buries the right article. searchUnion(prep(q), 5) is per-book
//    round-robin, not a global ranking, so the correct article lands at union index 5-23 on
//    8 of 10 cases even when it is per-book rank 0-2. Case 5 is the sharp one: Hippocampus
//    is per-book rank 6, so a top-5 union never contains it at all (union index -1).
//
// Dropped in verification: "what is a daemon in mythology" and three rephrasings
// ("...in ancient greek religion", "what is a daemon spirit", "what is a daemon in
// philosophy"). The general corpus has no Daimon/Daemon-as-spirit article; the query
// returns Magic in the Greco-Roman world, and "daemon philosophy" returns Systemd at rank
// 0 — a genuine sense inversion, but with no correct article to pin it to. Total drops: 4
// candidate queries, 1 term (daemon). No regex was weakened to save a case.

const TOP = "wikipedia_en_top_nopic_2026-06";     // 875,265 articles
const COMP = "wikipedia_en_computer_nopic_2026-06"; // 244,179 articles

export const CASES = [
  // ---- everyday sense wanted, computing sense must lose ----------------------------
  ["what is a kernel in corn", "concept", TOP,
    /^Maize$/i, /known as kernels or seeds/i],
  // prep: "kernel corn". TOP r=2 (Sweet corn, Grits, Maize); union idx 12 of 15.
  // COMP for the same query: Asus ZenFone, Google Pixel, GNU/Linux naming controversy —
  // no kernel-as-software article competes. Maize: "The ears yield grain, known as kernels
  // or seeds", and later the caryopsis definition.

  ["what is a shell in biology", "concept", TOP,
    /^Mollusca$/i, /chitin reinforced with calcium carbonate/i],
  // prep: "shell biology". TOP r=4 (Bivalvia, Turtle shell, Scallop, Cephalopod, Mollusca);
  // union idx 1 of 21. COMP: Galaxy (computational biology), BioRuby, NetworkX — "biology"
  // pulls bioinformatics tools, not shells. Mollusca: the shell "is made of proteins and
  // chitin reinforced with calcium carbonate, and is secreted by a mantle".

  ["what is a cookie made of", "concept", TOP,
    /^Biscuit$/i, /flour-based baked food/i],
  // prep: "cookie made". TOP r=2 (Oreo, Girl Scout Cookies, Biscuit) with HTTP cookie at
  // r=3 and Cross-site request forgery at r=4 — the only case where the computing sense
  // genuinely intrudes inside the winning book. Union idx 23 of 29, behind three wget/curl
  // cookie threads from Stack Exchange. Biscuit: "A biscuit is a flour-based baked food
  // item", and it records that US English calls the sweet ones cookies.

  ["how do python snakes kill their prey", "concept", TOP,
    /^Reticulated python$/i, /killing by constriction/i],
  // prep: "python snakes kill their prey". TOP r=0 (then Burmese python, Central African
  // rock python); COMP returns ZERO hits — "prey"/"kill" AND-out every computing page.
  // Union idx 1 of only 6, behind archlinux/List of games. Reticulated python: "an ambush
  // predator, usually waiting until prey wanders within strike range before seizing it in
  // its coils and killing by constriction".

  ["how does the brain consolidate long term memory", "concept", TOP,
    /^Hippocampus$/i, /consolidation of information from short-term memory to long-term memory/i],
  // prep: "brain consolidate long term memory". TOP r=6 (Memory, Anterograde amnesia,
  // Spatial memory, Emotion and memory, Flashback, Amygdala, Hippocampus) — inside want=8,
  // outside a top-5 union, so union idx = -1 of 13. COMP: The Magical Number Seven,
  // Optogenetics. Hippocampus: it "plays important roles in the consolidation of
  // information from short-term memory to long-term memory".

  // ---- computing sense wanted, everyday sense must lose ----------------------------
  ["what is a kernel in an operating system", "concept", COMP,
    /^Kernel \(operating system\)$/i, /computer program at the core of a computer/i],
  // prep: "kernel operating system". COMP r=0; union idx 7 of 48. TOP holds the same page
  // at r=0 too (duplicate, not a rival sense); no corn/maize article appears anywhere.
  // Needle: "A kernel is a computer program at the core of a computer's operating system
  // that always has complete control over everything in the system."

  ["what is a shell in computing", "concept", COMP,
    /^Shell \(computing\)$/i, /relatively thin layer around an operating system/i],
  // prep: "shell computing". COMP r=0 (Web shell, Remote Shell, Shell account follow);
  // union idx 10 of 48. TOP: same page at r=0; zero mollusc hits. Needle: "The term shell
  // refers to how it is a relatively thin layer around an operating system."

  ["what is an http cookie used for", "reference", COMP,
    /^HTTP cookie$/i, /small block of data created by a web server/i],
  // prep: "http cookie used". COMP r=0; union idx 5 of 30, behind four curl/wget cookie
  // threads. TOP: same page at r=0; no Biscuit/Oreo. Needle: "An HTTP cookie ... is a small
  // block of data created by a web server while a user is browsing a website", used to
  // store stateful information on the device.

  ["what is python the programming language", "concept", COMP,
    /^Python \(programming language\)$/i, /significant indentation/i],
  // prep: "python programming language". COMP r=0; union idx 8 of 35. The one case with a
  // live cross-book rival: archlinux/Python also lands in the union. TOP: same page at r=0,
  // no snake articles. Needle: Python "emphasizes code readability, simplicity, and
  // ease-of-writing with the use of significant indentation".

  ["what is virtual memory in a computer", "concept", COMP,
    /^Virtual memory$/i, /maps memory addresses used by a program, called virtual addresses/i],
  // prep: "virtual memory computer". COMP r=0; union idx 6 of 35. TOP: same page at r=0;
  // no Hippocampus/amnesia. Needle: the OS "maps memory addresses used by a program, called
  // virtual addresses, into physical addresses in computer memory".
];

// Unanswerable queries: the library genuinely cannot answer these, so the only correct
// behaviour is a refusal. Each was run through searchUnion(prep(query), 5); the trailing
// comment records the candidate count and whether anything came back plausible.
// PLAUSIBLE-BUT-WRONG is the dangerous class — a confident on-topic article that does not
// contain the answer — and is what a refusal path must catch. 3 of 6 are that kind.
export const NEGATIVE = [
  "what is the current stable version of the rust compiler",
  // union n=22. PLAUSIBLE-BUT-WRONG, worst case in this list: wikipedia_en_computer/Rust
  // (programming language) is r=2 in the union and its infobox reads "Stable release 1.96 /
  // May 28, 2026". Grounded, on-topic, precise — and stale by construction, since "current"
  // is unanswerable from a 2026-06 dump. A ranker cannot detect this; only the question can.

  "who won the 2028 united states presidential election",
  // union n=11. PLAUSIBLE-BUT-WRONG: wikipedia_en_100/President of the United States heads
  // the union and literally contains the string "2028" ("...electoral votes allocated
  // following the 2020 census ... for the 2024 and 2028 presidential elections") plus an
  // election-year navbox ending 2028. Term overlap is total; the result does not exist in
  // any dump (latest results present: 2020, 2024).

  "what is the recommended tirzepatide dose adjustment for stage 4 chronic kidney disease",
  // union n=4, all from wikipedia_en_top: Anti-obesity medication, Metabolic
  // dysfunction-associated steatotic liver disease, Obesity, Sleep apnea. PLAUSIBLE-BUT-WRONG:
  // Anti-obesity medication does name tirzepatide (FDA weight-management approvals) but the
  // article contains neither "chronic kidney" nor "dose adjust" — no clinical dosing
  // protocol exists anywhere in the library. Partial term match, zero answer.

  "what does article 17 of the estonian e-residency act require to register a company",
  // union n=13. Nothing plausible: wikipedia_en_100/Native Americans in the United States,
  // California gold rush, Catholic Church lead on "article"/"register"/"company";
  // wikipedia_en_computer/Digital identity and Electronic voting by country do mention
  // Estonian e-residency but no statute text. No legal corpus is mounted, so no article
  // number can ever be quoted.

  "why is the purple of seventeen louder than tuesday",
  // union n=1: wikipedia_en_top/Avril Lavigne (the album "Seventeen"/colour words). Nonsense
  // question, and kiwix's AND semantics collapse it to a single absurd hit — the easy
  // refusal, detectable from candidate count alone.

  "how many kilograms does the colour of my neighbour's wifi password weigh",
  // union n=0 across all 11 books. Category-error question; every term-AND is empty. The
  // only negative where retrieval itself refuses, with no candidate to be tempted by.
];
