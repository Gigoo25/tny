//! tny — a small-model terminal answerer.
//!
//! `tny "how do I create a swap file"` → grounded answer on stdout, source on stderr.
//!
//! The design is measured, not guessed: see NOTES.md (47 findings) and PLAN.md. The two
//! load-bearing results are that the model must never *select* anything — every choice it
//! was given lost to a deterministic rule — and that a regex grounding check, not a bigger
//! model, is what stops a fabricated shell command being served as fact.

mod corpus;
mod ground;
mod retrieve;

use ground::{command_vocab, html2txt, split_compare, ungrounded, ungrounded_detail, ungrounded_shape};
use retrieve::{article, pick_sections, prep, rank_articles, search_union, select_terms};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

// F19/F20/F47: thinking OFF is mandatory (Qwen opens `<think>` and never closes it inside
// 512 tokens, burning 95 s for zero answer). The second sentence is measured too: it took
// model-alone refusal on a mismatched context from 4/6 to 6/6 while cutting output from 31
// to 29 tokens. Command-only variants ("reply with that command and nothing else") lost a
// correct answer, 5/6.
const SYS: &str = "Answer the question using the reference material. Be concise: at most two sentences plus the exact command if one applies. Use only facts written in the reference. Never add a flag, option, version, or path that does not appear there.";

const CHAT_PORT: u16 = 8080;
const KIWIX_PORT: u16 = 8082;
// F26/F43/F46: 0.8B is the floor. 350M degenerates, 230M refuses nothing unaided, 2B costs
// 2.2× for the same 6/6, and Q4_K_M halves RAM but breaks the grounding check's recovery.
const MODEL: &str = "ggml-org/Qwen3.5-0.8B-GGUF:Q8_0";
const MODEL_DIR: &str = "models--ggml-org--Qwen3.5-0.8B-GGUF";
// F31: lexical section selection needs top-5 × 600 chars for 14/14.
const TOP_SECTIONS: usize = 5;
const PER_SECTION: usize = 600;
// F58: the answer is in the top-1 article for 36/58 cases and in the top-3 for 45/58.
const TOP_ARTICLES: usize = 3;
// F63: hits per book. Deeper costs nothing — one request per book either way, just a longer
// response — and it lifts the recall ceiling from 54/58 to 55/58 (`Hippocampus` answers
// "how does the brain consolidate long term memory" from its book's 6th hit). 12 adds nothing
// further, and neither 8 nor 12 disturbs the ranking: article@1 and answer@3 are unmoved, so
// the extra candidates are inert rather than noise.
const PER_BOOK: usize = 8;
/// F68: how many terms reach kiwix. Measured on one book: 8 terms returned the most hits of
/// any length tried, and 24 returned none. Overridable so a sweep costs a run rather than a
/// rebuild — `TNY_SEARCH_TERMS=999` reproduces the unlimited query this replaced.
fn search_terms() -> usize {
    std::env::var("TNY_SEARCH_TERMS").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
}

const NEED_LLAMA: &str = "llama-server not on PATH. Install llama.cpp:\n  \
    arch:   pacman -S llama.cpp\n  \
    nix:    nix-shell -p llama-cpp\n  \
    other:  prebuilt binaries at github.com/ggml-org/llama.cpp/releases";
const NEED_KIWIX: &str = "kiwix-serve not on PATH. Install kiwix-tools:\n  \
    debian: apt install kiwix-tools\n  \
    fedora: dnf install kiwix-tools\n  \
    arch:   pacman -S kiwix-tools\n  \
    nix:    nix-shell -p kiwix-tools\n  \
    other:  download.kiwix.org/release/kiwix-tools";

struct Cfg {
    chat: String,
    kiwix: String,
    zim: PathBuf,
    models: PathBuf,
    cache: PathBuf,
    verbose: bool,
    /// Retrieval only: print the shortlist and stop, so ranking is measurable without
    /// paying 21 s of generation per case (bench/rank-cli.mjs).
    rank_only: bool,
    /// Dump every retrieved candidate as JSON and stop, for offline scorer sweeps.
    dump: bool,
    /// Print the exact text handed to the model and stop, without loading it. Answers the
    /// only question that separates a retrieval failure from a model failure (F67).
    context: bool,
}

fn main() {
    let mut question = String::new();
    let mut verbose = false;
    let mut rank_only = false;
    let mut dump = false;
    let mut context = false;
    let mut follow = false;
    let mut corpus_args: Option<Vec<String>> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-v" | "--verbose" => verbose = true,
            "--rank" => rank_only = true,
            "--dump" => dump = true,
            "--context" => context = true,
            "-f" | "--follow" => follow = true,
            "--corpus" => corpus_args = Some(args.by_ref().collect()),
            "-h" | "--help" => {
                usage();
                return;
            }
            _ if a.starts_with('-') => {
                eprintln!("tny: unknown flag {a}");
                std::process::exit(1);
            }
            _ => {
                if !question.is_empty() {
                    question.push(' ');
                }
                question.push_str(&a);
            }
        }
    }

    let cfg = match config(verbose, rank_only, dump, context) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("tny: {e}");
            std::process::exit(1);
        }
    };

    let r = match corpus_args {
        Some(sub) => corpus_cmd(&cfg, &sub),
        None if question.trim().is_empty() => {
            usage();
            std::process::exit(1);
        }
        None => run(&cfg, &question, follow),
    };
    match r {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("tny: {e}");
            std::process::exit(1);
        }
    }
}

fn usage() {
    eprintln!(
        "tny \"question\"               grounded answer from local ZIM corpora\n\
         \n\
           -f, --follow               treat this as a follow-up to the last question\n\
           -v, --verbose              per-stage timings on stderr\n\
         \n\
         tny --corpus list            mounted ZIM files\n\
         tny --corpus search <text>   find ZIMs in the kiwix library\n\
         tny --corpus add <name>      download a ZIM (resumable, byte-verified)\n\
         tny --corpus update          check the library for newer editions\n\
         \n\
         needs llama-server and kiwix-serve on PATH\n\
         env: TNY_ZIM, TNY_MODELS, TNY_CHAT, TNY_KIWIX"
    );
}

fn config(verbose: bool, rank_only: bool, dump: bool, context: bool) -> Result<Cfg, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME is not set")?;
    let xdg_data = std::env::var("XDG_DATA_HOME").unwrap_or(format!("{home}/.local/share"));
    let xdg_cache = std::env::var("XDG_CACHE_HOME").unwrap_or(format!("{home}/.cache"));
    // a checked-out repo has ./zim and ./models; otherwise use XDG
    let local = |name: &str, fallback: String| -> PathBuf {
        let here = PathBuf::from(name);
        if here.is_dir() {
            here
        } else {
            PathBuf::from(fallback)
        }
    };
    let cache = PathBuf::from(format!("{xdg_cache}/tny"));
    std::fs::create_dir_all(&cache).map_err(|e| format!("cannot create {}: {e}", cache.display()))?;
    Ok(Cfg {
        chat: std::env::var("TNY_CHAT").unwrap_or(format!("http://127.0.0.1:{CHAT_PORT}")),
        kiwix: std::env::var("TNY_KIWIX").unwrap_or(format!("http://127.0.0.1:{KIWIX_PORT}")),
        zim: std::env::var("TNY_ZIM").map(PathBuf::from).unwrap_or_else(|_| local("zim", format!("{xdg_data}/tny/zim"))),
        models: std::env::var("TNY_MODELS").map(PathBuf::from).unwrap_or_else(|_| local("models", format!("{xdg_data}/tny/models"))),
        cache,
        verbose,
        rank_only,
        dump,
        context,
    })
}

// ------------------------------------------------------------------ corpus commands

fn corpus_cmd(cfg: &Cfg, args: &[String]) -> Result<i32, String> {
    match args.first().map(String::as_str).unwrap_or("list") {
        "list" => {
            let local = corpus::local(&cfg.zim);
            if local.is_empty() {
                eprintln!("tny: no ZIMs in {} — try: tny --corpus search bash", cfg.zim.display());
                return Ok(3);
            }
            for stem in local {
                println!("{stem}");
            }
            Ok(0)
        }
        "search" => {
            let q = args[1..].join(" ");
            if q.trim().is_empty() {
                return Err("usage: tny --corpus search <text>".into());
            }
            let entries = match corpus::cached(&cfg.cache) {
                Some(e) => e,
                None => corpus::fetch(&cfg.cache)?,
            };
            let hits = corpus::suggest(&entries, &q, 10);
            if hits.is_empty() {
                eprintln!("tny: nothing in the library matched {q:?}");
                return Ok(3);
            }
            for e in hits {
                println!("{:<34} {:>9}  {:>8} arts  {}", e.name, e.size_human(), e.articles, e.title);
            }
            Ok(0)
        }
        "add" => {
            let name = args.get(1).ok_or("usage: tny --corpus add <name>")?;
            corpus::add(&cfg.zim, &cfg.cache, name)?;
            // kiwix-serve mounts its library once at startup, so a new ZIM needs a new
            // process; drop it and the next query picks up the wider library.
            remount(cfg);
            Ok(0)
        }
        "update" => {
            let entries = corpus::fetch(&cfg.cache)?;
            let stale = corpus::outdated(&cfg.zim, &entries);
            corpus::write_stale_note(&cfg.cache, &stale);
            if stale.is_empty() {
                println!("all {} corpora are current", corpus::local(&cfg.zim).len());
            } else {
                for (name, have, newest) in &stale {
                    println!("{name:<34} {have} → {newest}");
                }
                eprintln!("tny: refresh with: tny --corpus add <name>");
            }
            Ok(0)
        }
        other => Err(format!("unknown corpus command {other:?} — list, search, add, update")),
    }
}

// ------------------------------------------------------------------ the pipeline

fn run(cfg: &Cfg, question: &str, follow: bool) -> Result<i32, String> {
    let t_all = Instant::now();
    serve_kiwix(cfg)?;

    let prev = if follow { last_turn(cfg) } else { None };
    // F29: the retrieval query is `<prev question> <this question>`. NEVER a model rewrite:
    // asked to rephrase, 0.8B inverted "how do I turn it off" into "how do I turn it back
    // on". Concatenation scored 5/6 against the rewrite's 4/6, and it is free.
    let retrieval_q = match &prev {
        Some((q, _)) => format!("{q} {question}"),
        None => question.to_string(),
    };

    // F37: a comparison question has no single source article, and synthesis from two is
    // unreliable (2/5 — the model invents the side it was not shown). The split is
    // model-free, both names coming from the question's own grammar, and it must run BEFORE
    // `prep`, which strips the very words it needs. Fires 6/6, silent on 26/26.
    let books = corpus::local(&cfg.zim);
    if let Some((a, b)) = split_compare(&retrieval_q) {
        let one = |s: &str| {
            let c = search_union(&cfg.kiwix, &prep(s), &books, PER_BOOK);
            rank_articles(s, &c).into_iter().next()
        };
        let (ha, hb) = (one(&a), one(&b));
        if let (Some(x), Some(y)) = (ha.as_ref(), hb.as_ref()) {
            if x.title != y.title {
                eprintln!("tny: that compares two topics — ask about one:\n  · {}\n  · {}", x.title, y.title);
                return Ok(2);
            }
        }
    }

    let t = Instant::now();
    // F68: a sentence is not a search query. kiwix scores 0 hits at 24 terms, so the query is
    // cut to its most informative 8 and widened only if that finds nothing — the backoff is
    // free on every query that already works, and the two extra searches cost ~300 ms on the
    // ones that do not. Ranking still sees the *whole* question: only the search is cut.
    // F71: rarity was measured and rejected as a term-selection objective. Real document
    // frequency, summed per book over every mounted ZIM and cached on disk, ordered terms
    // correctly - `the` 2500, `uname` 556, `harina` 107 - and selecting the rarest eight
    // scored 37/67 against the shape heuristic's 63/67, reproduced on one server instance.
    // Xapian already weights rarity inside the engine, so pre-selecting rare terms counts it
    // twice and spends the query budget on incidentals; what the engine cannot recover is
    // topic coverage. That is the second time an IDF signal has lost here, and the first
    // (23/58 against 32/58) was blamed on biased statistics. The statistics were fine.
    let prepped = prep(&retrieval_q);
    let mut query = select_terms(&prepped, search_terms());
    let mut cands = search_union(&cfg.kiwix, &query, &books, PER_BOOK);
    // F72: an empty shortlist means one of two very different things, and tny reported the
    // wrong one. kiwix-serve dies under sustained load on a small machine — measured here
    // repeatedly, SIGSEGV then SIGABRT with 4.5 GB of ZIMs and 400 MB of free RAM — after
    // which every book's search fails silently and the user is told "no local corpus matched",
    // about an answer sitting on their disk. Ask whether the server is alive before believing
    // its silence, and give it one restart.
    if cands.is_empty() && !up(&format!("{}/", cfg.kiwix)) {
        eprintln!("tny: kiwix-serve is not responding — restarting it");
        serve_kiwix(cfg)?;
        cands = search_union(&cfg.kiwix, &query, &books, PER_BOOK);
    }
    // F70: widening a thin shortlist was measured and rejected. Three of five held-out misses
    // had six candidates or fewer, so backing off at twelve looked obvious — and it scored
    // *worse*, 60/67 against 62/67, because the extra candidates outranked the answer. Recall
    // was never the constraint: every one of those pages was already inside its own book's
    // top 50. Ranking precision is the constraint, so the backoff stays empty-only.
    for cap in [5, 3] {
        if !cands.is_empty() {
            break;
        }
        query = select_terms(&prepped, cap);
        for c in search_union(&cfg.kiwix, &query, &books, PER_BOOK) {
            if !cands.iter().any(|x| x.book == c.book && x.path == c.path) {
                cands.push(c);
            }
        }
    }
    let t_search = t.elapsed();
    if cands.is_empty() {
        return Ok(no_local_match(cfg, &query));
    }
    let ranked = rank_articles(&retrieval_q, &cands);
    // Retrieval is 2 % of a query's wall time, so measuring ranking through full generation
    // costs 80 s per case and hides the thing under test. `--rank` stops here and prints the
    // whole shortlist, because rank-1 alone cannot distinguish a scoring miss from a
    // candidate that was never retrieved (F49's actual lesson).
    if cfg.rank_only {
        for c in ranked.iter().take(8) {
            println!("{}\t{}\t{}", c.book, c.title, c.path);
        }
        return Ok(0);
    }
    if cfg.dump {
        // Every candidate, unranked, as JSON: lets a scorer be tried offline in
        // milliseconds instead of a 100-second run against live kiwix per variant.
        let rows: Vec<serde_json::Value> = cands
            .iter()
            .map(|c| {
                serde_json::json!({
                    "book": c.book, "title": c.title, "path": c.path, "snip": c.snip,
                    "rank": c.rank, "kind": format!("{:?}", c.kind),
                })
            })
            .collect();
        println!("{}", serde_json::to_string(&rows).unwrap_or_default());
        return Ok(0);
    }
    // Only now is the model needed: everything above is retrieval. `--context` stops before
    // the load, so inspecting what the model was given costs a search and a fetch.
    if !cfg.context {
        serve_chat(cfg)?;
    }
    let best = &ranked[0];

    // F58: three articles, not one. Rank-1 carries the answer for 36 of 58 verified cases,
    // but the top *three* carry it for 45 — retrieval's misses are near-misses, and a
    // ranking gap of 9 cases is closed by fetching 2 more articles at 15-41 ms each rather
    // than by a better scorer (F53 closed content reranking; F56 exhausted lexical signals).
    // F39b measured the same shape at the section level: 3 articles x 1 section beat
    // 1 article x 3 sections, 5/6 vs 4/6.
    let t = Instant::now();
    let mut docs: Vec<(&retrieve::Candidate, String)> = Vec::new();
    for c in ranked.iter().take(TOP_ARTICLES) {
        match article(&cfg.kiwix, &c.book, &c.path) {
            Ok(html) => docs.push((c, html)),
            Err(e) if docs.is_empty() => return Err(e),
            Err(_) => {}
        }
    }
    let t_fetch = t.elapsed();

    // The budget is split, not multiplied: the same ~3 KB of context, sourced from three
    // articles. F41 measured that a bigger window does not buy accuracy — placement does.
    let per_doc = TOP_SECTIONS.div_ceil(docs.len().max(1));
    let mut parts = Vec::new();
    let mut heads = Vec::new();
    for (c, html) in &docs {
        let p = pick_sections(html, &retrieval_q, per_doc, PER_SECTION);
        if p.text.trim().is_empty() {
            continue;
        }
        // Name the source inline: with three articles in context the model must be able to
        // attribute, and the grounding check compares against the union below.
        parts.push(format!("## {}\n{}", c.title, p.text));
        heads.extend(p.heads);
    }
    let picked = retrieve::Picked { heads, text: parts.join("\n\n") };
    if cfg.context {
        println!("{}", picked.text);
        return Ok(0);
    }

    // F32: grounding reads the whole article, not the slice sent to the model — the slice
    // rejected a correct answer for citing `cryptsetup` from a neighbouring section. With
    // three sources the reference is the union of all three, or a correct answer taken from
    // the second article would be rejected as ungrounded.
    let full = docs.iter().map(|(_, h)| html2txt(h)).collect::<Vec<_>>().join("\n");
    let vocab = docs.iter().flat_map(|(_, h)| command_vocab(h)).collect::<Vec<_>>();

    let t = Instant::now();
    let answer = ask(cfg, question, &picked.text, prev.as_ref())?;
    let t_gen = t.elapsed();

    // F27/F44/F45: three rules, each with its own reference. A false reject is the worst
    // outcome — it turns a correct answer into "not found" — so each was tuned against
    // correct answers as hard as against fabrications.
    let why = [
        ungrounded(&answer, &full, question, &picked.text),
        ungrounded_detail(&answer, &full),
        ungrounded_shape(&answer, question, &vocab),
    ]
    .into_iter()
    .find(|r| !r.is_empty())
    .unwrap_or_default();

    if cfg.verbose {
        eprintln!(
            "  search {} ms · fetch {} ms · generate {} ms",
            t_search.as_millis(),
            t_fetch.as_millis(),
            t_gen.as_millis()
        );
    }
    eprintln!(
        "{} · {} · §{} ({:.1}s)",
        best.book,
        best.title,
        picked.heads.join(", §"),
        t_all.elapsed().as_secs_f64()
    );
    if let Some(note) = corpus::stale_note(&cfg.cache) {
        eprint!("{note}");
    }

    if !why.is_empty() {
        eprintln!("tny: rejected — {why}");
        // F57: a rejection means the mounted corpora did not carry this answer. That is the
        // one moment a download suggestion is certainly not noise.
        suggest_corpus(cfg, &query);
        println!("not found");
        return Ok(3);
    }
    println!("{}", answer.trim());
    save_turn(cfg, question, &answer);
    Ok(0)
}

/// F40/F57: the catalog is the index — 1,286 English ZIMs, cached locally at 1.5 MB, so a
/// suggestion costs a file read and no network. A lexical match over catalog metadata names
/// the right ZIM for 8/8 queries the local corpus cannot answer, with 0 false suggestions on
/// 5 it can.
///
/// Two triggers, both a *measured* failure signal rather than a guess: zero retrieved
/// candidates, and a grounding rejection. The second was specified in F40 and never wired —
/// "not found" is precisely the moment the user needs to know which corpus would have held
/// the answer.
fn suggest_corpus(cfg: &Cfg, query: &str) {
    match corpus::cached(&cfg.cache) {
        Some(entries) => {
            let hits = corpus::suggest(&entries, query, 3);
            if !hits.is_empty() {
                eprintln!("     the library has:");
                for e in hits {
                    eprintln!("       {:<30} {:>9}  tny --corpus add {}", e.title, e.size_human(), e.name);
                }
            }
        }
        None => eprintln!("     tny --corpus search {query}   (fetches the library catalog)"),
    }
}

fn no_local_match(cfg: &Cfg, query: &str) -> i32 {
    println!("not found");
    eprintln!("tny: no local corpus matched {query:?} ({} mounted)", corpus::local(&cfg.zim).len());
    suggest_corpus(cfg, query);
    3
}

fn ask(cfg: &Cfg, question: &str, reference: &str, prev: Option<&(String, String)>) -> Result<String, String> {
    let mut messages = vec![serde_json::json!({ "role": "system", "content": SYS })];
    // F28: keep the prior turn in the message list. History carries the antecedent for
    // elliptical follow-ups ("how do I unlock *it* at boot") — 83 % vs 75 % stateless — and
    // it is cheaper than it looks, because turn 1's prefix is still in the KV cache.
    if let Some((q, a)) = prev {
        messages.push(serde_json::json!({ "role": "user", "content": q }));
        messages.push(serde_json::json!({ "role": "assistant", "content": a }));
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": format!("Reference:\n{reference}\n\nQuestion: {question}")
    }));

    let body = serde_json::json!({
        "messages": messages,
        "temperature": 0.1,
        "top_k": 50,
        "repeat_penalty": 1.05,
        "max_tokens": 160,
        "chat_template_kwargs": { "enable_thinking": false },
    });
    let resp = ureq::post(&format!("{}/v1/chat/completions", cfg.chat))
        .timeout(Duration::from_secs(300))
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| format!("chat request failed: {e}"))?
        .into_string()
        .map_err(|e| format!("chat body: {e}"))?;
    let j: serde_json::Value = serde_json::from_str(&resp).map_err(|e| format!("chat json: {e}"))?;
    let msg = &j["choices"][0]["message"];
    let content = msg["content"].as_str().unwrap_or("");
    // F19: empty content with reasoning present is an ERROR, not an answer.
    if content.trim().is_empty() && !msg["reasoning_content"].as_str().unwrap_or("").is_empty() {
        return Err("model emitted reasoning only — thinking is not disabled".into());
    }
    Ok(content.to_string())
}

// ------------------------------------------------------------------ servers

fn on_path(bin: &str) -> bool {
    std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).any(|d| d.join(bin).is_file()))
        .unwrap_or(false)
}

/// Reuse a live server, else spawn it. F25: `--no-mmproj` always — without it llama.cpp
/// downloads a vision projector nobody uses.
///
/// Split in two because retrieval does not need the model: `--rank` measures ranking on a
/// machine with no llama.cpp at all, and demanding it there cost a 15-minute benchmark run
/// that reported 0/58 for a missing PATH entry.
fn serve_kiwix(cfg: &Cfg) -> Result<(), String> {
    if !up(&format!("{}/", cfg.kiwix)) {
        let zims: Vec<PathBuf> = std::fs::read_dir(&cfg.zim)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "zim"))
            .collect();
        if zims.is_empty() {
            return Err(format!("no ZIM files in {} — try: tny --corpus search bash", cfg.zim.display()));
        }
        if !on_path("kiwix-serve") {
            return Err(NEED_KIWIX.into());
        }
        let mut cmd = Command::new("kiwix-serve");
        cmd.arg("--port").arg(KIWIX_PORT.to_string());
        for z in &zims {
            cmd.arg(z);
        }
        spawn(cmd, cfg, "kiwix")?;
        wait_up(&format!("{}/", cfg.kiwix), 120, "kiwix-serve")?;
    }
    Ok(())
}

fn serve_chat(cfg: &Cfg) -> Result<(), String> {

    if !up(&format!("{}/health", cfg.chat)) {
        if !on_path("llama-server") {
            return Err(NEED_LLAMA.into());
        }
        std::fs::create_dir_all(&cfg.models).map_err(|e| format!("cannot create {}: {e}", cfg.models.display()))?;
        // llama.cpp fetches `-hf` models into LLAMA_CACHE itself; say so, because a silent
        // 800 MB first run looks like a hang.
        if !cfg.models.join(MODEL_DIR).is_dir() {
            eprintln!("tny: downloading {MODEL} (~800 MB, once) into {}", cfg.models.display());
        }
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let mut cmd = Command::new("llama-server");
        cmd.args(["-hf", MODEL, "--no-mmproj", "--jinja", "--host", "127.0.0.1"])
            .args(["-t", &threads.to_string(), "-c", "8192", "--port", &CHAT_PORT.to_string()])
            .env("LLAMA_CACHE", &cfg.models);
        spawn(cmd, cfg, "chat")?;
        wait_up(&format!("{}/health", cfg.chat), 1800, "llama-server")?;
    }
    Ok(())
}

fn spawn(mut cmd: Command, cfg: &Cfg, what: &str) -> Result<(), String> {
    let log = cfg.cache.join(format!("{what}.log"));
    let out = std::fs::File::create(&log).map_err(|e| format!("cannot write {}: {e}", log.display()))?;
    let err = out.try_clone().map_err(|e| format!("log dup: {e}"))?;
    cmd.stdin(Stdio::null()).stdout(out).stderr(err);
    let child = cmd.spawn().map_err(|e| format!("cannot start {what}: {e}"))?;
    let _ = std::fs::write(cfg.cache.join(format!("{what}.pid")), child.id().to_string());
    Ok(())
}

fn remount(cfg: &Cfg) {
    let pidfile = cfg.cache.join("kiwix.pid");
    if let Ok(pid) = std::fs::read_to_string(&pidfile) {
        if let Ok(pid) = pid.trim().parse::<u32>() {
            let _ = Command::new("kill").arg(pid.to_string()).status();
            let _ = std::fs::remove_file(&pidfile);
            eprintln!("tny: kiwix-serve will restart with the new corpus on the next query");
        }
    }
}

fn up(url: &str) -> bool {
    ureq::builder()
        .timeout(Duration::from_millis(1500))
        .build()
        .get(url)
        .call()
        .is_ok()
}

fn wait_up(url: &str, secs: u64, what: &str) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < deadline {
        if up(url) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(400));
    }
    Err(format!("{what} did not come up within {secs}s — see the log in the tny cache dir"))
}

fn last_turn(cfg: &Cfg) -> Option<(String, String)> {
    let raw = std::fs::read_to_string(cfg.cache.join("last.json")).ok()?;
    let j: serde_json::Value = serde_json::from_str(&raw).ok()?;
    Some((j["q"].as_str()?.to_string(), j["a"].as_str()?.to_string()))
}

fn save_turn(cfg: &Cfg, q: &str, a: &str) {
    let j = serde_json::json!({ "q": q, "a": a });
    if let Ok(mut f) = std::fs::File::create(cfg.cache.join("last.json")) {
        let _ = f.write_all(j.to_string().as_bytes());
    }
}
