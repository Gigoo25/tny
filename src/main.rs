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
mod tui;

use ground::{command_vocab, html2txt, split_compare, ungrounded, ungrounded_detail, ungrounded_shape};
use retrieve::{article, pick_sections, prep, rank_articles, search_union, select_terms};
use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

// F19/F20/F47: thinking OFF is mandatory (Qwen opens `<think>` and never closes it inside
// 512 tokens, burning 95 s for zero answer). The second sentence is measured too: it took
// model-alone refusal on a mismatched context from 4/6 to 6/6 while cutting output from 31
// to 29 tokens. Command-only variants ("reply with that command and nothing else") lost a
// correct answer, 5/6.
/// The two rules that survive every measurement: answer from the reference, and never invent
/// a flag. How *long* the answer runs is the user's business — `Len` supplies that clause.
const SYS: &str = "Answer the question using the reference material. Use only facts written in the reference. Never add a flag, option, version, or path that does not appear there.";

/// F102: how much answer you want, separate from how long you will wait for it. A one-line
/// answer to "what does -p do" is right and a paragraph is padding; a one-line answer to "how
/// do I set up a swap file" is useless. The model cannot judge which it is looking at, so the
/// dial is the user's.
#[derive(Clone, Copy, PartialEq)]
enum Len {
    Low,
    Medium,
    Max,
}

impl Len {
    fn clause(self) -> &'static str {
        match self {
            Len::Low => " Answer in one sentence, plus the exact command if one applies.",
            Len::Medium => " Be concise: at most three sentences, plus the exact command if one applies.",
            // Still bounded: unbounded generation on a 0.8B rambles, and every token costs a
            // second of wall clock on the machines this runs on.
            Len::Max => " Answer fully, in at most three short paragraphs. Include the exact commands that apply.",
        }
    }
    fn tokens(self) -> u32 {
        match self {
            Len::Low => 80,
            Len::Medium => 160,
            Len::Max => 512,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Len::Low => "low",
            Len::Medium => "medium",
            Len::Max => "max",
        }
    }
    fn parse(s: &str) -> Option<Len> {
        match s {
            "low" | "short" => Some(Len::Low),
            "medium" => Some(Len::Medium),
            "max" | "long" => Some(Len::Max),
            _ => None,
        }
    }
    fn next(self) -> Option<Len> {
        match self {
            Len::Low => Some(Len::Medium),
            Len::Medium => Some(Len::Max),
            Len::Max => None,
        }
    }
    fn prev(self) -> Option<Len> {
        match self {
            Len::Low => None,
            Len::Medium => Some(Len::Low),
            Len::Max => Some(Len::Medium),
        }
    }
}

const CHAT_PORT: u16 = 8080;
const KIWIX_PORT: u16 = 8082;
// F26/F43/F46: 0.8B is the floor. 350M degenerates, 230M refuses nothing unaided, 2B costs
// 2.2× for the same 6/6, and Q4_K_M halves RAM but breaks the grounding check's recovery.
/// F101: the model is its own dial, not a consequence of the speed one. They are independent
/// questions — how much text to read, and who reads it — and binding them meant a 4B could
/// only ever be tried with the deepest context, which is the slowest possible way to find out
/// whether it is worth anything (F104: 4.44 tok/s prefill makes that an 11-minute answer).
///
/// Named models are the ones measured here; anything else is passed to llama-server as `-hf`
/// verbatim, so trying a new one costs a flag rather than a rebuild.
const MODELS: [(&str, &str, &str); 3] = [
    // key      repo:quant                              size
    ("0.8b", "ggml-org/Qwen3.5-0.8B-GGUF:Q8_0", "~800 MB"),
    ("2b", "ggml-org/Qwen3.5-2B-GGUF:Q4_K_M", "~1.2 GB"),
    ("4b", "unsloth/Qwen3.5-4B-GGUF:Q4_K_M", "~2.5 GB"),
];
const MODEL: &str = MODELS[0].1;
// F31: lexical section selection needs top-5 × 600 chars for 14/14. F80: also the prompt
// size, and prefill is 19–22 s of a 20–40 s answer — so these are latency constants as much
// as accuracy ones. Overridable so a sweep costs a run rather than a rebuild.
fn top_sections(cfg: &Cfg) -> usize {
    std::env::var("TNY_TOP_SECTIONS").ok().and_then(|v| v.parse().ok()).unwrap_or(cfg.mode.dims().1)
}
fn per_section(cfg: &Cfg) -> usize {
    std::env::var("TNY_PER_SECTION").ok().and_then(|v| v.parse().ok()).unwrap_or(cfg.mode.dims().2)
}
// F58: the answer is in the top-1 article for 36/58 cases and in the top-3 for 45/58.
// F82: overridable, to measure what a cheap first pass over one article would reach.
fn top_articles(cfg: &Cfg) -> usize {
    std::env::var("TNY_TOP_ARTICLES").ok().and_then(|v| v.parse().ok()).unwrap_or(cfg.mode.dims().0)
}
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

/// F94: this laptop is slow and every laptop running a 0.8B on CPU will be, so the honest
/// knob is not "make it fast" but "how long are you willing to wait for this question".
/// Context size is the only dial that matters — prefill is 85-90 % of an answer (F80) — and
/// it buys real accuracy: `--oneline`'s definition is 10 KB into git-log(1), and F82 measured
/// one article carrying the answer for 40 of 58 cases against three articles' 46.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// No model at all: retrieval plus the best passage, verbatim (F95).
    Ultrafast,
    Fast,
    Medium,
    Slow,
    Molasses,
}

impl Cfg {
    /// True when a TUI is drawing: answers are returned, not printed.
    fn hosted(&self) -> bool {
        self.progress.is_some()
    }
}

impl Mode {
    /// (articles, sections, chars per section)
    fn dims(self) -> (usize, usize, usize) {
        match self {
            Mode::Ultrafast => (1, 2, 600),
            Mode::Fast => (1, 2, 600),
            Mode::Medium => (3, 5, 600),
            Mode::Slow => (3, 5, 1200),
            Mode::Molasses => (3, 6, 2000),
        }
    }
    /// What the model is for: everything except the first tier.
    fn generates(self) -> bool {
        self != Mode::Ultrafast
    }
    fn prev(self) -> Option<Mode> {
        match self {
            Mode::Ultrafast => None,
            Mode::Fast => Some(Mode::Ultrafast),
            Mode::Medium => Some(Mode::Fast),
            Mode::Slow => Some(Mode::Medium),
            Mode::Molasses => Some(Mode::Slow),
        }
    }
    fn next(self) -> Option<Mode> {
        match self {
            Mode::Ultrafast => Some(Mode::Fast),
            Mode::Fast => Some(Mode::Medium),
            Mode::Medium => Some(Mode::Slow),
            Mode::Slow => Some(Mode::Molasses),
            Mode::Molasses => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Mode::Ultrafast => "ultrafast",
            Mode::Fast => "fast",
            Mode::Medium => "medium",
            Mode::Slow => "slow",
            Mode::Molasses => "molasses",
        }
    }
    fn parse(s: &str) -> Option<Mode> {
        match s {
            "ultrafast" => Some(Mode::Ultrafast),
            "fast" => Some(Mode::Fast),
            "medium" => Some(Mode::Medium),
            "slow" => Some(Mode::Slow),
            "molasses" => Some(Mode::Molasses),
            _ => None,
        }
    }
}

#[derive(Clone)]
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
    /// Skip the answer cache for this question and replace what is in it.
    fresh: bool,
    mode: Mode,
    len: Len,
    /// `repo:quant` for llama-server's `-hf`.
    model: String,
    /// Where stage labels go when a TUI owns the screen. `None` is the plain CLI: print.
    progress: Option<std::sync::Arc<std::sync::Mutex<String>>>,
}

fn main() {
    let mut question = String::new();
    let mut verbose = false;
    let mut rank_only = false;
    let mut dump = false;
    let mut context = false;
    let mut fresh = false;
    let mut mode: Option<Mode> = None;
    let mut len: Option<Len> = None;
    let mut model: Option<String> = None;
    let mut follow = false;
    let mut corpus_args: Option<Vec<String>> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-v" | "--verbose" => verbose = true,
            "--rank" => rank_only = true,
            "--dump" => dump = true,
            "--context" => context = true,
            "--fresh" => fresh = true,
            "--ultrafast" => mode = Some(Mode::Ultrafast),
            "--fast" => mode = Some(Mode::Fast),
            "--medium" => mode = Some(Mode::Medium),
            "--slow" => mode = Some(Mode::Slow),
            "--molasses" => mode = Some(Mode::Molasses),
            "--model" => model = args.next(),
            "--low" | "--short" => len = Some(Len::Low),
            "--max" | "--long" => len = Some(Len::Max),
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

    let cfg = match config(verbose, rank_only, dump, context, fresh, mode, len, model) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("tny: {e}");
            std::process::exit(1);
        }
    };

    // F96: a terminal gets the TUI, a pipe gets a line of text. The benchmark harness, the
    // `--context`/`--rank`/`--dump` diagnostics and `tny q > file` all redirect stdout, so
    // they keep the exact behaviour they had — and an interactive user never gets a screen
    // full of escape codes in their pager.
    let tui = std::io::stdout().is_terminal()
        && std::io::stdin().is_terminal()
        && !cfg.rank_only
        && !cfg.dump
        && !cfg.context;
    let r = match corpus_args {
        Some(sub) => corpus_cmd(&cfg, &sub),
        None if tui => tui::run(&cfg, &question),
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
    // A raw string, because `\n\` continuations swallow the leading whitespace of the next
    // line and every attempt to indent a sub-line inside one silently comes out flush left.
    eprint!(
        r#"tny "question"          an answer from the ZIM corpora on this disk
tny                     the same, interactive

speed        --ultrafast   the best passage from the page, no model      0.3 s
             --fast        one article                                    14 s
             --medium      three articles (default)                       39 s
             --slow        three articles, read twice as deep             50 s
             --molasses    three articles, as deep as it gets             90 s

length       --low         one sentence
             --max         up to three paragraphs

model        --model 0.8b  also 2b, 4b, or any huggingface repo:quant

             what you pick in the interface is what you get next time

  -f, --follow    treat this as a follow-up to the last question
      --fresh     re-answer instead of reusing the cached answer
  -v, --verbose   per-stage timings on stderr

corpus       tny --corpus list             mounted ZIM files
             tny --corpus search <text>    find ZIMs in the kiwix library
             tny --corpus pack [name]      a whole shelf: mini small medium large huge
             tny --corpus add <name>       one ZIM (resumable, byte-verified)
             tny --corpus update           check for newer editions

keys         i ask · 1-9 pick a source · ⏎ read it · j k scroll · r again
             + - speed · < > length · :model 4b · q quit

needs llama-server and kiwix-serve on PATH
env: TNY_ZIM, TNY_MODELS, TNY_CHAT, TNY_KIWIX, TNY_MODE, TNY_LEN, TNY_MODEL
"#
    );
}

fn config(
    verbose: bool,
    rank_only: bool,
    dump: bool,
    context: bool,
    fresh: bool,
    mode: Option<Mode>,
    len: Option<Len>,
    model: Option<String>,
) -> Result<Cfg, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME is not set")?;
    let xdg_data = std::env::var("XDG_DATA_HOME").unwrap_or(format!("{home}/.local/share"));
    let xdg_cache = std::env::var("XDG_CACHE_HOME").unwrap_or(format!("{home}/.cache"));
    // F77: the corpus location must not depend on where the user is standing. This used to
    // prefer `./zim` and `./models` when they existed, so the same question answered from a
    // checked-out repo and from $HOME searched different libraries — "no local corpus matched
    // (2 mounted)" in one directory and a correct answer in another. One fixed place, and
    // TNY_ZIM/TNY_MODELS for the cases that genuinely need another (the benchmarks, a corpus
    // on external disk). ZIMs run to gigabytes and models to hundreds of megabytes, so they
    // live under XDG *data*, not config; `~/.cache/tny` keeps only what is regenerable.
    let cache = PathBuf::from(format!("{xdg_cache}/tny"));
    std::fs::create_dir_all(&cache).map_err(|e| format!("cannot create {}: {e}", cache.display()))?;
    let (saved_mode, saved_len, saved_model) = load_prefs(&cache);
    Ok(Cfg {
        chat: std::env::var("TNY_CHAT").unwrap_or(format!("http://127.0.0.1:{CHAT_PORT}")),
        kiwix: std::env::var("TNY_KIWIX").unwrap_or(format!("http://127.0.0.1:{KIWIX_PORT}")),
        zim: env_path("TNY_ZIM", format!("{xdg_data}/tny/zim")),
        models: env_path("TNY_MODELS", format!("{xdg_data}/tny/models")),
        cache,
        verbose,
        rank_only,
        dump,
        context,
        fresh,
        progress: None,
        // Flag, then env for a shell that always wants one, then what was last chosen.
        mode: mode
            .or_else(|| std::env::var("TNY_MODE").ok().and_then(|v| Mode::parse(&v)))
            .or(saved_mode)
            .unwrap_or(Mode::Medium),
        len: len
            .or_else(|| std::env::var("TNY_LEN").ok().and_then(|v| Len::parse(&v)))
            .or(saved_len)
            .unwrap_or(Len::Medium),
        model: model
            .or_else(|| std::env::var("TNY_MODEL").ok())
            .map(|m| resolve_model(&m))
            .or(saved_model)
            .unwrap_or_else(|| MODEL.to_string()),
    })
}

fn env_path(var: &str, fallback: String) -> PathBuf {
    std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(fallback))
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
            let added = vec![name.clone()];
            // kiwix-serve mounts its library once at startup, so a new ZIM needs a new
            // process; drop it and the next query picks up the wider library.
            remount(cfg);
            warn_unsearchable(cfg, &added);
            Ok(0)
        }
        "pack" => {
            let entries = match corpus::cached(&cfg.cache) {
                Some(e) => e,
                None => corpus::fetch(&cfg.cache)?,
            };
            let Some(name) = args.get(1) else {
                println!("tny --corpus pack <name>\n");
                for p in corpus::PACK_NAMES {
                    let (keys, what) = corpus::pack(p).expect("named pack exists");
                    let plan = corpus::pack_plan(&cfg.zim, &entries, &keys);
                    println!(
                        "  {:<7} {:>9} to fetch   {} books, {} already here\n          {}",
                        p,
                        human_bytes(plan.bytes),
                        keys.len(),
                        plan.present.len(),
                        what
                    );
                    if !plan.missing.is_empty() {
                        eprintln!("          not in the catalog: {}", plan.missing.join(", "));
                    }
                }
                return Ok(0);
            };
            let (keys, _) = corpus::pack(name)
                .ok_or_else(|| format!("no pack {name:?} — {}", corpus::PACK_NAMES.join(", ")))?;
            let plan = corpus::pack_plan(&cfg.zim, &entries, &keys);
            let want = plan.want;
            if !plan.missing.is_empty() {
                eprintln!("tny: not in the catalog, skipping: {}", plan.missing.join(", "));
            }
            if want.is_empty() {
                println!("pack {name} is complete — all {} books are mounted", plan.present.len());
                return Ok(0);
            }
            println!("pack {name}: {} to download, {}", want.len(), human_bytes(plan.bytes));
            for (n, b) in &want {
                println!("  {:<40} {:>9}", n, human_bytes(*b));
            }
            // A pack can be tens of gigabytes. Never start that without a keypress, and
            // never ask for one when nobody is watching (a script gets the plan and stops).
            if !std::io::stdin().is_terminal() {
                eprintln!("tny: not a terminal — re-run interactively to confirm");
                return Ok(0);
            }
            eprint!("download? [y/N] ");
            std::io::stderr().flush().ok();
            if !matches!(key(), Some(b'y') | Some(b'Y')) {
                eprintln!("no");
                return Ok(0);
            }
            eprintln!("yes");
            let mut failed = 0;
            for (i, (n, _)) in want.iter().enumerate() {
                eprintln!("\n[{}/{}] {n}", i + 1, want.len());
                if let Err(e) = corpus::add(&cfg.zim, &cfg.cache, n) {
                    eprintln!("tny: {n} failed — {e}");
                    failed += 1;
                }
            }
            remount(cfg);
            warn_unsearchable(cfg, &want.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>());
            if failed > 0 {
                eprintln!("\n{failed} of {} failed — re-run to resume", want.len());
            }
            Ok(if failed > 0 { 3 } else { 0 })
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
        other => Err(format!("unknown corpus command {other:?} — list, search, add, pack, update")),
    }
}

/// F88: `--corpus update` only ever warned, and only when the user thought to run it, so a
/// library quietly ages until someone notices an answer citing a two-year-old page. The check
/// comes to the user instead — after the answer, never before it, because nobody asked a
/// question in order to be told about downloads.
///
/// Weekly at most, and declining snoozes it for another week: a prompt that appears every
/// time is a prompt that gets answered without reading. Offline or catalog fetch failing is
/// not an error here — it silently counts as "checked" so a laptop on a train stays quiet.
const UPDATE_EVERY: u64 = 7 * 24 * 60 * 60;

fn maybe_offer_update(cfg: &Cfg) {
    let path = cfg.cache.join("update.json");
    let state: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    let now = now_secs();
    if now.saturating_sub(state["checked"].as_u64().unwrap_or(0)) < UPDATE_EVERY {
        return;
    }
    let mark = |asked: bool| {
        let _ = std::fs::write(
            &path,
            serde_json::json!({ "checked": now, "asked": asked }).to_string(),
        );
    };
    let Ok(entries) = corpus::fetch(&cfg.cache) else {
        mark(false);
        return;
    };
    let stale = corpus::outdated(&cfg.zim, &entries);
    corpus::write_stale_note(&cfg.cache, &stale);
    mark(!stale.is_empty());
    if stale.is_empty() {
        return;
    }
    let bytes: u64 = stale
        .iter()
        .filter_map(|(k, _, d)| entries.iter().find(|e| e.key() == *k && e.date() == *d))
        .map(|e| e.bytes)
        .sum();
    eprintln!(
        "\n\x1b[2m  {} newer edition{} available ({}): {}\x1b[0m",
        stale.len(),
        if stale.len() == 1 { "" } else { "s" },
        human_bytes(bytes),
        stale.iter().map(|(k, _, _)| k.as_str()).collect::<Vec<_>>().join(", ")
    );
    eprint!("  update now? [y/N] ");
    std::io::stderr().flush().ok();
    if !matches!(key(), Some(b'y') | Some(b'Y')) {
        eprintln!("not now — asking again in a week");
        return;
    }
    eprintln!("yes");
    for (k, _, d) in &stale {
        if let Some(e) = entries.iter().find(|e| e.key() == *k && e.date() == *d) {
            if let Err(err) = corpus::add(&cfg.zim, &cfg.cache, &e.name) {
                eprintln!("tny: {} failed — {err}", e.name);
            }
        }
    }
    corpus::write_stale_note(&cfg.cache, &[]);
    remount(cfg);
}

fn human_bytes(b: u64) -> String {
    const U: [(&str, u64); 3] = [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)];
    for (unit, div) in U {
        if b >= div {
            return format!("{:.1} {unit}", b as f64 / div as f64);
        }
    }
    format!("{b} B")
}

// ------------------------------------------------------------------ the pipeline
/// A question takes 20–60 s on a CPU, and silence for that long reads as a hang. The spinner
/// is stderr-only and TTY-only, so pipes, the harness and `tny … > file` are untouched.
///
/// Streaming the answer as it generates would be better and cannot be done: F27/F44/F45 reject
/// ungrounded answers *after* reading them, and an answer that was printed cannot be
/// unprinted. Progress is the honest substitute — it reports the stage, never the content.
struct Spin {
    tx: Option<std::sync::mpsc::Sender<Option<String>>>,
    join: Option<std::thread::JoinHandle<()>>,
    cell: Option<std::sync::Arc<std::sync::Mutex<String>>>,
}

impl Spin {
    /// Hosted: the label goes to the TUI's shared cell and no thread is spawned — the TUI
    /// already redraws on its own clock, so a second animator would just fight it.
    fn hosted(cell: &std::sync::Arc<std::sync::Mutex<String>>) -> Spin {
        Spin { tx: None, join: None, cell: Some(cell.clone()) }
    }

    fn start(on: bool, first: &str) -> Spin {
        if !on {
            return Spin { tx: None, join: None, cell: None };
        }
        let (tx, rx) = std::sync::mpsc::channel::<Option<String>>();
        let mut label = first.to_string();
        let t0 = Instant::now();
        let join = std::thread::spawn(move || {
            const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            for i in 0.. {
                match rx.recv_timeout(std::time::Duration::from_millis(90)) {
                    Ok(Some(next)) => label = next,
                    Ok(None) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(_) => {}
                }
                eprint!("\r\x1b[2K\x1b[2m{} {label}  {:.0}s\x1b[0m", FRAMES[i % 10], t0.elapsed().as_secs_f64());
                let _ = std::io::stderr().flush();
            }
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
        });
        Spin { tx: Some(tx), join: Some(join), cell: None }
    }

    fn say(&self, label: impl Into<String>) {
        if let Some(cell) = &self.cell {
            if let Ok(mut c) = cell.lock() {
                *c = label.into();
            }
            return;
        }
        if let Some(tx) = &self.tx {
            let _ = tx.send(Some(label.into()));
        }
    }

    fn stop(&mut self) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(None);
        }
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// `wikipedia_en_top_nopic_2026-06` is a filename, not a source. Strip the language, flavour
/// and date a ZIM name carries so the line under an answer reads like a citation.
fn human_book(name: &str) -> String {
    let stem = name.split("_en_").next().unwrap_or(name);
    match stem {
        "wikipedia" => "Wikipedia".into(),
        "archlinux" => "Arch Wiki".into(),
        s if s.starts_with("devdocs") => {
            let topic = name.split("_en_").nth(1).unwrap_or("").split('_').next().unwrap_or("");
            format!("devdocs {topic}").trim().to_string()
        }
        s => s.trim_end_matches(".com").to_string(),
    }
}

/// One answered question. The CLI prints it; the TUI draws it.
struct Answered {
    code: i32,
    text: String,
    sources: Vec<Source>,
    /// Every candidate that survived ranking — a superset of `sources`, and what steering
    /// picks from, because the right page is often one the answer was not built from.
    shortlist: Vec<Source>,
}

/// An article the answer was built from, kept so the prompt can open or re-use it.
#[derive(Clone)]
struct Source {
    book: String,
    path: String,
    title: String,
}

enum Next {
    Ask(String, bool),
    /// Re-ask the same question against one chosen result — the steer.
    Use(usize),
    Open(usize),
    /// Print the whole shortlist, not just what was read.
    More,
    /// Re-ask the same question with more context — the "think harder" key.
    Harder,
    Again,
    Quit,
}

/// F75: a terminal answerer that exits after one answer makes the user retype `tny` and pay
/// the library and model start-up again for a question they already had in mind. The prompt
/// keeps the process — and both servers — warm, and costs nothing when stdin is not a
/// terminal, so pipes and the benchmark harness behave exactly as before.
fn run(cfg: &Cfg, question: &str, follow: bool) -> Result<i32, String> {
    let mut q = question.to_string();
    let mut follow = follow;
    let mut focus: Option<Source> = None;
    let mut cfg = Cfg { mode: cfg.mode, ..cfg.clone() };
    loop {
        let Answered { code, sources, shortlist, .. } = answer_once(&cfg, &q, follow, focus.as_ref())?;
        focus = None;
        let interactive = std::io::stdin().is_terminal()
            && std::io::stderr().is_terminal()
            && !cfg.rank_only
            && !cfg.dump
            && !cfg.context;
        if !interactive {
            return Ok(code);
        }
        // After the answer, and only here: a check that delays an answer is a check that
        // gets disabled.
        maybe_offer_update(&cfg);
        if shortlist.is_empty() && sources.is_empty() {
            return Ok(code);
        }
        // Steering works off the shortlist, which is a superset of what was read: the answer
        // may have come from the wrong three, and the right page is often the fourth.
        let list = if shortlist.is_empty() { sources.clone() } else { shortlist };
        let mut showing_all = false;
        loop {
            match prompt(&list, sources.len(), showing_all, cfg.mode)? {
                Next::Quit => return Ok(code),
                Next::Again => continue,
                Next::More => {
                    showing_all = true;
                    for (i, s) in list.iter().enumerate() {
                        let mark = if i < sources.len() { "·" } else { " " };
                        eprintln!("\x1b[2m {mark}{} {} · {}\x1b[0m", i + 1, human_book(&s.book), s.title);
                    }
                }
                Next::Open(i) => open_source(&cfg, &list, i),
                // The steer: same question, that source, no retrieval.
                Next::Use(i) => match list.get(i) {
                    Some(s) => {
                        eprintln!("\x1b[2m  using {}\x1b[0m", s.title);
                        focus = Some(s.clone());
                        break;
                    }
                    None => continue,
                },
                // F94: the same question, read more deeply. Cheaper to press than to retype,
                // and it is the honest response to "that answer looks thin".
                Next::Harder => match cfg.mode.next() {
                    Some(m) => {
                        cfg.mode = m;
                        eprintln!("\x1b[2m  reading more — {} mode\x1b[0m", m.name());
                        break;
                    }
                    None => {
                        eprintln!("\x1b[2m  already at molasses\x1b[0m");
                        continue;
                    }
                },
                Next::Ask(text, f) => {
                    q = text;
                    follow = f;
                    break;
                }
            }
        }
    }
}

/// One keypress, not a line: `q` must quit on the key, not on the key plus Enter. `stty` is
/// POSIX and tny already shells out to two servers, so this costs no raw-mode dependency.
/// `-isig` rather than plain `-icanon` matters: with signals left on, a Ctrl-C during the read
/// would kill the process with echo still off and leave the user's shell unusable. Instead
/// Ctrl-C arrives as a byte we handle, and the terminal is always restored by the line below.
fn key() -> Option<u8> {
    let tty = |args: &[&str]| {
        Command::new("stty").args(args).stdin(Stdio::inherit()).output().ok()
    };
    let saved = tty(&["-g"])?;
    let saved = String::from_utf8_lossy(&saved.stdout).trim().to_string();
    tty(&["-icanon", "-echo", "-isig", "min", "1", "time", "0"])?;
    let mut b = [0u8; 1];
    let n = std::io::stdin().read(&mut b).unwrap_or(0);
    tty(&[&saved]);
    (n == 1).then_some(b[0])
}

fn prompt(list: &[Source], read: usize, showing_all: bool, mode: Mode) -> Result<Next, String> {
    // The digits are the steer, not the browser: pointing tny at the right page is the thing
    // a user needs most when the first answer missed, so it costs one keypress. Opening a
    // page in a browser is the rarer want, so it costs two (`o` then the digit).
    let hint = if showing_all || list.len() <= read {
        format!("1-{} use that source", list.len())
    } else {
        format!("1-{read} use that source · s see all {} matches", list.len())
    };
    let harder = if mode.next().is_some() { " · + read more" } else { "" };
    eprint!("\x1b[2m  ▸ ask a follow-up · {hint}{harder} · o open · n new · q quit\x1b[0m\n> ");
    std::io::stderr().flush().ok();
    // No stty means no single-key mode; one answer and out beats a broken terminal.
    let Some(k) = key() else { return Ok(Next::Quit) };
    Ok(match k {
        b'q' | 3 | 4 | 27 => {
            eprintln!();
            Next::Quit
        }
        // Re-ask the same question against one chosen result.
        b'1'..=b'9' => {
            eprintln!();
            Next::Use((k - b'1') as usize)
        }
        b's' => {
            eprintln!();
            Next::More
        }
        b'+' => {
            eprintln!();
            Next::Harder
        }
        // `o` then a digit: open in the browser. kiwix already serves the corpora over HTTP,
        // so this works with no network.
        b'o' => {
            eprint!("open which? ");
            std::io::stderr().flush().ok();
            match key() {
                Some(d @ b'1'..=b'9') => {
                    eprintln!();
                    Next::Open((d - b'1') as usize)
                }
                _ => {
                    eprintln!();
                    Next::Open(0)
                }
            }
        }
        b'\r' | b'\n' => Next::Again,
        // F29: a follow-up carries the previous turn into the retrieval query; `n` drops it,
        // for when the subject changes and the old question would only poison the search.
        b'n' => {
            eprint!("\x1b[2m new topic\x1b[0m\n> ");
            std::io::stderr().flush().ok();
            match line()? {
                Some(q) => Next::Ask(q, false),
                None => Next::Quit,
            }
        }
        // Any other key is the first character of a question. Raw mode ate it, so echo it and
        // read the rest cooked - Backspace and the shell's line editing then work as usual.
        c if c.is_ascii_graphic() || c == b' ' => {
            eprint!("{}", c as char);
            std::io::stderr().flush().ok();
            match line()? {
                Some(rest) => Next::Ask(format!("{}{rest}", c as char), true),
                None => Next::Quit,
            }
        }
        _ => Next::Again,
    })
}

/// The rest of a typed line. `None` is EOF, which is a quit rather than an error.
fn line() -> Result<Option<String>, String> {
    let mut s = String::new();
    if std::io::stdin().read_line(&mut s).map_err(|e| e.to_string())? == 0 {
        eprintln!();
        return Ok(None);
    }
    let s = s.trim().to_string();
    Ok(if s.is_empty() { None } else { Some(s) })
}

/// The corpora are already served over HTTP by kiwix-serve, so "open the source" is a URL the
/// browser can read — offline, from the same ZIM the answer came from.
fn open_source(cfg: &Cfg, sources: &[Source], i: usize) {
    let Some(s) = sources.get(i) else { return };
    let url = format!("{}/content/{}/{}", cfg.kiwix, s.book, s.path);
    let opener = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    match Command::new(opener).arg(&url).stdout(Stdio::null()).stderr(Stdio::null()).spawn() {
        Ok(_) => eprintln!("\x1b[2m  opening {}\x1b[0m", s.title),
        Err(_) => eprintln!("  {url}"),
    }
}

/// One question, one answer. Returns the exit code and the articles the answer was built
/// from, so the prompt underneath can offer to open them.
/// `focus` is the steer: when the user picks a result, retrieval is skipped and the answer is
/// built from that one article. One turn should be enough — this is the safety net for when
/// it is not, not an excuse for it not to be.
fn answer_once(
    cfg: &Cfg,
    question: &str,
    follow: bool,
    focus: Option<&Source>,
) -> Result<Answered, String> {
    let t_all = Instant::now();
    let quiet = cfg.rank_only || cfg.dump || cfg.context;
    let mut spin = match &cfg.progress {
        Some(cell) => Spin::hosted(cell),
        None => Spin::start(std::io::stderr().is_terminal() && !quiet, "starting the library"),
    };
    serve_kiwix(cfg)?;

    let prev = if follow { recent_turns(cfg, 2) } else { Vec::new() };
    // F85: a repeat costs a file read instead of 40 s. The key includes the turn being
    // followed, so the same words after a different question are a different question.
    let key = cache_key(cfg, question, prev.last());
    if focus.is_none() && !cfg.fresh && !cfg.rank_only && !cfg.dump && !cfg.context {
        if let Some((answer, sources, shortlist)) = cached(cfg, &key) {
            spin.stop();
            if !cfg.hosted() {
                println!("{answer}");
                eprintln!("\n\x1b[2m  {}   cached\x1b[0m", cite_lines(&sources).join("\n  "));
            }
            save_turn(cfg, question, &answer);
            return Ok(Answered { code: 0, text: answer, sources, shortlist });
        }
    }
    // F29: the retrieval query is `<prev question> <this question>`. NEVER a model rewrite:
    // asked to rephrase, 0.8B inverted "how do I turn it off" into "how do I turn it back
    // on". Concatenation scored 5/6 against the rewrite's 4/6, and it is free. Only the
    // immediately previous question joins it — F84 widened the model's history, not this.
    let retrieval_q = match prev.last() {
        // The current question leads: `search_terms` cuts the query to its eight most
        // informative terms, and with the previous question first its terms took the budget.
        Some((q, _)) => format!("{question} {q}"),
        None => question.to_string(),
    };

    // F37 measured synthesis from two topics at 2/5 and made comparisons refuse: "ask about
    // one". But the failure it recorded was the model inventing *the side it was not shown* —
    // a context bug, not a reasoning limit, because retrieval returned one article for a
    // two-sided question. F86 shows both sides instead of refusing. The split is model-free,
    // both names coming from the question's own grammar, and it must run BEFORE `prep`, which
    // strips the very words it needs. Fires 6/6, silent on 26/26.
    let books = corpus::local(&cfg.zim);
    let compare = split_compare(&retrieval_q).and_then(|(a, b)| {
        let one = |s: &str| {
            let c = search_union(&cfg.kiwix, &prep(s), &books, PER_BOOK);
            rank_articles(s, &c).into_iter().next()
        };
        match (one(&a), one(&b)) {
            // Both sides landing on one article is not a comparison — "SIGTERM vs SIGKILL"
            // is answered by the signals page, and splitting it would read it twice.
            (Some(x), Some(y)) if x.title != y.title => Some(vec![x, y]),
            _ => None,
        }
    });

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
    spin.say(format!("searching {} corpora", books.len()));
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
        spin.stop();
        return Ok(Answered {
            code: no_local_match(cfg, &query),
            text: String::from("not found"),
            sources: vec![],
            shortlist: vec![],
        });
    }
    // A comparison replaces the shortlist with one article per side, so the model is shown
    // both things it is being asked to compare rather than one and its own imagination.
    let ranked = match focus {
        // Steered: one article, chosen by the user, and no ranking to argue with.
        Some(s) => vec![retrieve::Candidate {
            title: s.title.clone(),
            book: s.book.clone(),
            path: s.path.clone(),
            snip: String::new(),
            kind: retrieve::page_kind(&s.title, &s.path),
            rank: 0,
        }],
        None => match compare {
            // F91: TNY_RANK=rrf swaps the linear score for reciprocal rank fusion, so the
            // two can be compared on one build rather than two.
            Some(pair) => pair,
            None if std::env::var("TNY_RANK").as_deref() == Ok("rrf") => {
                retrieve::rank_articles_rrf(&retrieval_q, &cands)
            }
            None => rank_articles(&retrieval_q, &cands),
        },
    };
    // Retrieval is 2 % of a query's wall time, so measuring ranking through full generation
    // costs 80 s per case and hides the thing under test. `--rank` stops here and prints the
    // whole shortlist, because rank-1 alone cannot distinguish a scoring miss from a
    // candidate that was never retrieved (F49's actual lesson).
    if cfg.rank_only {
        for c in ranked.iter().take(8) {
            println!("{}\t{}\t{}", c.book, c.title, c.path);
        }
        return Ok(Answered { code: 0, text: String::new(), sources: vec![], shortlist: vec![] });
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
        return Ok(Answered { code: 0, text: String::new(), sources: vec![], shortlist: vec![] });
    }
    // Only now is the model needed: everything above is retrieval. `--context` stops before
    // the load, so inspecting what the model was given costs a search and a fetch.
    if !cfg.context && cfg.mode.generates() {
        spin.say("loading the model");
        serve_chat(cfg)?;
    }


    // F58: three articles, not one. Rank-1 carries the answer for 36 of 58 verified cases,
    // but the top *three* carry it for 45 — retrieval's misses are near-misses, and a
    // ranking gap of 9 cases is closed by fetching 2 more articles at 15-41 ms each rather
    // than by a better scorer (F53 closed content reranking; F56 exhausted lexical signals).
    // F39b measured the same shape at the section level: 3 articles x 1 section beat
    // 1 article x 3 sections, 5/6 vs 4/6.
    spin.say(format!("reading {}", ranked[0].title));
    let t = Instant::now();
    let mut docs: Vec<(&retrieve::Candidate, String)> = Vec::new();
    // The eight results the user can steer with, not just the three that were read.
    let shortlist: Vec<Source> = ranked
        .iter()
        .take(8)
        .map(|c| Source { book: c.book.clone(), path: c.path.clone(), title: c.title.clone() })
        .collect();

    // F91: one book can win every slot — "how do I change the hostname" put six man pages
    // above the ArchWiki page, so all three articles the model reads came from one source.
    // Capping articles per book was the obvious fix and was measured and rejected: cap=2 is
    // neutral on the fixture (46/58 either way) and costs a case on the held-out set
    // (evidence 50/55 -> 49/55); cap=1 costs three (43/58), because two sections of one
    // article often carry an answer between them. Flooding is real and this is not its fix.
    for c in ranked.iter().take(top_articles(cfg)) {
        match article(&cfg.kiwix, &c.book, &c.path) {
            Ok(html) => docs.push((c, html)),
            Err(e) if docs.is_empty() => return Err(e),
            Err(_) => {}
        }
    }
    let t_fetch = t.elapsed();

    // The budget is split, not multiplied: the same context, sourced from three articles. F41
    // measured that a bigger window does not buy accuracy — placement does.
    let pick = |arts: usize, secs: usize| {
        let docs = &docs[..arts.min(docs.len())];
        let n = docs.len().max(1);
        let mut parts = Vec::new();
        let mut heads = Vec::new();
        for (i, (c, html)) in docs.iter().enumerate() {
            // Distribute exactly `secs`. `div_ceil` per document rounded *up* every time: 5
            // sections over 3 articles was 6, 3787 chars against the 3000 the constant names,
            // a fifth of every prefill spent on a section nobody budgeted.
            let per_doc = secs / n + usize::from(i < secs % n);
            if per_doc == 0 {
                continue;
            }
            let p = pick_sections(html, &retrieval_q, per_doc, per_section(cfg));
            if p.text.trim().is_empty() {
                continue;
            }
            // Name the source inline: with three articles in context the model must be able to
            // attribute, and the grounding check compares against the union below.
            parts.push(format!("## {}\n{}", c.title, p.text));
            heads.extend(p.heads);
        }
        retrieve::Picked { heads, text: parts.join("\n\n") }
    };
    let budget = (top_articles(cfg), top_sections(cfg));
    if cfg.context {
        println!("{}", pick(budget.0, budget.1).text);
        return Ok(Answered { code: 0, text: String::new(), sources: vec![], shortlist: vec![] });
    }

    // F32: grounding reads the whole article, not the slice sent to the model — the slice
    // rejected a correct answer for citing `cryptsetup` from a neighbouring section. With
    // three sources the reference is the union of all three, or a correct answer taken from
    // the second article would be rejected as ungrounded.
    let full = docs.iter().map(|(_, h)| html2txt(h)).collect::<Vec<_>>().join("\n");
    let vocab = docs.iter().flat_map(|(_, h)| command_vocab(h)).collect::<Vec<_>>();

    spin.say(if cfg.mode.generates() { "answering" } else { "reading" });
    let t = Instant::now();
    // F82: one article and two sections carry the fact for 40 of 58 cases at ~310 prefill
    // tokens, against 46/58 at ~913 — 69 % of questions answered for a third of the wall
    // clock. The escalation trigger is not a new signal: the grounding rules already reject an
    // answer its context does not support, and that rejection is exactly the request for more
    // to read. This cannot lose ground, because the second rung IS the old single pass, and a
    // rejected first rung used to end the turn at "not found" with nothing further tried.
    let rungs: Vec<(usize, usize)> = if cfg.mode.generates() && budget.0 > 1 && budget.1 > 2 {
        vec![(1, 2), budget]
    } else {
        vec![budget]
    };
    let mut picked = retrieve::Picked { heads: Vec::new(), text: String::new() };
    let mut answer = String::new();
    let mut why = String::new();
    for (arts, secs) in &rungs {
        picked = pick(*arts, *secs);
        // F95: ultrafast answers from the page itself. Nothing to ground, because nothing was
        // written — the text is the source's own, and the grounding rules exist to catch a
        // model saying more than its reference does.
        answer = if cfg.mode.generates() {
            ask(cfg, question, &picked.text, &prev)?
        } else {
            retrieve::best_passage(&picked.text, question)
        };
        // F27/F44/F45: three rules, each with its own reference. A false reject is the worst
        // outcome — it turns a correct answer into "not found" — so each was tuned against
        // correct answers as hard as against fabrications.
        why = if cfg.mode.generates() {
            [
                ungrounded(&answer, &full, question, &picked.text),
                ungrounded_detail(&answer, &full),
                ungrounded_shape(&answer, question, &vocab),
            ]
            .into_iter()
            .find(|r| !r.is_empty())
            .unwrap_or_default()
        } else {
            String::new()
        };
        // F111: the deterministic rules ran and found nothing, so only now is the judge worth a
        // call. It runs on every rung deliberately: on an early rung its "no" escalates, which
        // costs context rather than the answer, and only the last rung's "no" is a refusal.
        // Gating it on the last rung made it dead code — the ladder breaks at rung 1 whenever
        // the deterministic rules pass, which is most questions.
        if why.is_empty() && judged_offtopic(cfg, question, &answer) {
            why = "answer is not about the question's subject".into();
        }
        if why.is_empty() {
            break;
        }
    }
    let t_gen = t.elapsed();

    // F83: cite the articles the answer came from, not every article that was read. Three go
    // into context and "why is the sky blue" credited `Sky Blue Sky`, a Wilco album, because
    // it ranked first — the answer came from the other two. An article earns its line by
    // containing the answer's distinctive words: the same evidence the grounding check uses,
    // applied per article instead of to the union of all three.
    let sources: Vec<Source> = supporting(&answer, &docs)
        .into_iter()
        .map(|(c, _)| Source {
            book: c.book.clone(),
            path: c.path.clone(),
            title: c.title.clone(),
        })
        .collect();

    spin.stop();
    if cfg.verbose {
        eprintln!(
            "  search {} ms · fetch {} ms · generate {} ms · §{}",
            t_search.as_millis(),
            t_fetch.as_millis(),
            t_gen.as_millis(),
            picked.heads.join(", §")
        );
    }
    if let Some(note) = corpus::stale_note(&cfg.cache) {
        eprint!("{note}");
    }

    if !why.is_empty() {
        if cfg.hosted() {
            return Ok(Answered {
                code: 3,
                text: format!("not found — {why}"),
                sources: vec![],
                shortlist,
            });
        }
        eprintln!("tny: rejected — {why}");
        // F57: a rejection means the mounted corpora did not carry this answer. That is the
        // one moment a download suggestion is certainly not noise.
        suggest_corpus(cfg, &query);
        println!("not found");
        return Ok(Answered { code: 3, text: String::from("not found"), sources, shortlist });
    }
    if !cfg.hosted() {
        println!("{}", answer.trim());
    }
    // F75: the source line was `wikipedia_en_top_nopic_2026-06 · Sky Blue Sky · §Release and
    // reception, §Composition, §Sky and sea, §Artificial blues, …` — a filename, a rank-1
    // article that was not where the answer came from, and six section names that mean
    // something only to whoever wrote the ranker. What a reader needs is which works were
    // consulted; sections are diagnostics and moved behind `-v`.
    // Numbered, because the prompt underneath opens them by number.
    if !cfg.hosted() {
        eprintln!(
            "\n\x1b[2m  {}   {:.1}s\x1b[0m",
            cite_lines(&sources).join("\n  "),
            t_all.elapsed().as_secs_f64()
        );
    }
    save_turn(cfg, question, &answer);
    cache_put(cfg, &key, answer.trim(), &sources, &shortlist);
    Ok(Answered { code: 0, text: answer.trim().to_string(), sources, shortlist })
}

fn cite_lines(sources: &[Source]) -> Vec<String> {
    sources
        .iter()
        .enumerate()
        .map(|(i, s)| format!("{} {} · {}", i + 1, human_book(&s.book), s.title))
        .collect()
}

/// Which of the read articles actually support this answer. Distinctive words only — short
/// and common ones match everywhere and would put every article back on the list. An article
/// is kept when it carries most of what the best one carries, so a fact stated in two
/// articles cites both, and the Wilco album cites nothing.
fn supporting<'a>(
    answer: &str,
    docs: &'a [(&'a retrieve::Candidate, String)],
) -> Vec<&'a (&'a retrieve::Candidate, String)> {
    let words: Vec<String> = answer
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|w| w.len() >= 5)
        .map(str::to_string)
        .collect();
    if words.is_empty() || docs.len() < 2 {
        return docs.iter().collect();
    }
    let share = |html: &str| {
        let text = html2txt(html).to_lowercase();
        words.iter().filter(|w| text.contains(w.as_str())).count() as f64 / words.len() as f64
    };
    let scored: Vec<(f64, &(&retrieve::Candidate, String))> =
        docs.iter().map(|d| (share(&d.1), d)).collect();
    let best = scored.iter().map(|(s, _)| *s).fold(0.0, f64::max);
    // Nothing matched anywhere: the answer is odd, so name everything read rather than
    // silently citing nothing.
    if best <= 0.0 {
        return docs.iter().collect();
    }
    scored.into_iter().filter(|(s, _)| *s >= best * 0.8).map(|(_, d)| d).collect()
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

fn ask(cfg: &Cfg, question: &str, reference: &str, prev: &[(String, String)]) -> Result<String, String> {
    let mut messages = vec![serde_json::json!({ "role": "system", "content": format!("{SYS}{}", cfg.len.clause()) })];
    // F28: keep the prior turns in the message list. History carries the antecedent for
    // elliptical follow-ups ("how do I unlock *it* at boot") — 83 % vs 75 % stateless — and
    // it is cheaper than it looks, because turn 1's prefix is still in the KV cache.
    // F84: two exchanges rather than one, so a thread's third question can still see its
    // first. Not more: every turn is prefill, and prefill is 85 % of the answer.
    for (q, a) in prev {
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
        "max_tokens": cfg.len.tokens(),
        "chat_template_kwargs": { "enable_thinking": false },
    });
    // F107: 300 s silently converted the 4B's slowest answers into "not found" — the request
    // died, the caller saw an empty answer, and the benchmark scored four *timeouts* as
    // refusals. The ceiling has to clear the slowest model anyone can select: the 4B needs
    // 328 s for a 900-token prompt on this CPU, so 20 minutes covers a bigger one on a worse
    // machine, and a real hang is still bounded.
    let resp = ureq::post(&format!("{}/v1/chat/completions", cfg.chat))
        .timeout(Duration::from_secs(1200))
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

/// F111: the last resort, and the only rule that can see a *grounded* answer about the wrong
/// thing. "why is the sky blue" answered from `Sky Blue Sky`, the Wilco album, is faithful to
/// its reference — every number and identifier in it is there — so `ungrounded_detail` is blind
/// by construction, and the question's terms are fully contained in the album's title, so no
/// lexical check sees it either.
///
/// This is NOT F16's judge. That one chose among eight candidates and a 350M emitted a
/// near-constant index; this reads one question and one answer, emits one token, and can only
/// ever turn an answer into a refusal — it chooses nothing and it cannot invent. Opt-in until
/// the number exists: `TNY_JUDGE=1`.
///
/// Positive phrasing only (F7: "Do NOT" measurably backfires).
fn judged_offtopic(cfg: &Cfg, question: &str, answer: &str) -> bool {
    if std::env::var("TNY_JUDGE").ok().as_deref() != Some("1") {
        return false;
    }
    let body = serde_json::json!({
        "messages": [{
            "role": "user",
            "content": format!(
                "Question: {question}\nAnswer: {answer}\n\nIs that answer about the same subject the question asks about? Reply with one word, yes or no."
            )
        }],
        "temperature": 0.0,
        "max_tokens": 4,
        "chat_template_kwargs": { "enable_thinking": false },
    });
    let Ok(resp) = ureq::post(&format!("{}/v1/chat/completions", cfg.chat))
        .timeout(Duration::from_secs(120))
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
    else {
        return false; // a judge that cannot be reached must not reject anything
    };
    let Ok(text) = resp.into_string() else { return false };
    let Ok(j) = serde_json::from_str::<serde_json::Value>(&text) else { return false };
    let verdict = j["choices"][0]["message"]["content"].as_str().unwrap_or("").to_lowercase();
    // Only an explicit "no" rejects. An empty or unparseable verdict is not evidence, and F16's
    // failure mode was a model answering constantly — silence must cost nothing.
    verdict.trim_start().starts_with("no")
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
    let zim_count = std::fs::read_dir(&cfg.zim)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "zim"))
        .count();
    // F89: a live server is not necessarily serving the library on disk. One catalog request
    // settles it, and a mismatch means the server predates a download — restart it rather
    // than answer from a library the user no longer has.
    if up(&format!("{}/", cfg.kiwix)) && mounted_books(cfg).is_some_and(|m| m != zim_count) {
        eprintln!("tny: kiwix-serve is mounting fewer books than are on disk — restarting it");
        remount(cfg);
    }
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

/// The model id the running server reports, e.g. `Qwen3.5-0.8B-Q8_0`. `None` if it is not
/// answering — treated as "not the one we want", which just means it gets started.
fn serving(cfg: &Cfg) -> Option<String> {
    let body = ureq::get(&format!("{}/v1/models", cfg.chat))
        .timeout(Duration::from_millis(1500))
        .call()
        .ok()?
        .into_string()
        .ok()?;
    let j: serde_json::Value = serde_json::from_str(&body).ok()?;
    let id = j["data"][0]["id"].as_str()?;
    Some(model_id(id))
}

/// `ggml-org/Qwen3.5-0.8B-GGUF:Q8_0` and the path llama-server reports both reduce to the
/// GGUF's stem, which is the only part the two representations share.
fn model_id(s: &str) -> String {
    let tail = s.rsplit('/').next().unwrap_or(s);
    tail.trim_end_matches(".gguf").replace("-GGUF:", "-").to_string()
}

/// Kill the chat server we started, so the next question can start a different model.
fn stop_chat(cfg: &Cfg) {
    let pidfile = cfg.cache.join("chat.pid");
    if let Ok(pid) = std::fs::read_to_string(&pidfile) {
        if let Ok(pid) = pid.trim().parse::<u32>() {
            let _ = Command::new("kill").arg(pid.to_string()).status();
        }
    }
    let _ = std::fs::remove_file(&pidfile);
    for _ in 0..40 {
        if !up(&format!("{}/health", cfg.chat)) {
            return;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Physical cores, never logical. F3 measured a 4th thread on a 2-core box as *net negative*
/// for decode (9.72 t/s against 3 threads' 11.09), and `available_parallelism` counts
/// hyperthreads — on a P/E hybrid it also counts efficiency cores, and every ggml barrier then
/// waits on the slowest one. `/proc/cpuinfo` lists a (physical id, core id) pair per logical
/// CPU; the unique pairs are the cores. No `/proc`, or a kernel that omits the pair: trust
/// `available_parallelism` and let `TNY_THREADS` be the answer.
fn physical_cores() -> usize {
    let fallback = || std::thread::available_parallelism().map_or(4, |n| n.get());
    let Ok(txt) = std::fs::read_to_string("/proc/cpuinfo") else { return fallback() };
    let mut cores: Vec<(String, String)> = Vec::new();
    let (mut pkg, mut core) = (String::new(), String::new());
    for line in txt.lines() {
        let Some((k, v)) = line.split_once(':') else { continue };
        match k.trim() {
            "physical id" => pkg = v.trim().to_string(),
            "core id" => core = v.trim().to_string(),
            _ => continue,
        }
        if !pkg.is_empty() && !core.is_empty() {
            let pair = (std::mem::take(&mut pkg), std::mem::take(&mut core));
            if !cores.contains(&pair) {
                cores.push(pair);
            }
        }
    }
    if cores.is_empty() { fallback() } else { cores.len() }
}

fn serve_chat(cfg: &Cfg) -> Result<(), String> {
    if !cfg.mode.generates() {
        return Ok(());
    }
    let model = cfg.model.as_str();
    let size = MODELS.iter().find(|(_, r, _)| *r == model).map(|(_, _, s)| *s).unwrap_or("");
    // F101: a mode switch can mean a different model, and one llama-server serves one model.
    // Restarting is honest — 1.5 s for the 0.8B — and two servers would not fit in RAM on the
    // machines this is for.
    if up(&format!("{}/health", cfg.chat)) && serving(cfg) != Some(model_id(model)) {
        stop_chat(cfg);
    }
    if !up(&format!("{}/health", cfg.chat)) {
        if !on_path("llama-server") {
            return Err(NEED_LLAMA.into());
        }
        std::fs::create_dir_all(&cfg.models).map_err(|e| format!("cannot create {}: {e}", cfg.models.display()))?;
        // llama.cpp fetches `-hf` models into LLAMA_CACHE itself; say so, because a silent
        // 800 MB first run looks like a hang.
        // llama.cpp names its cache dir after the repo; close enough to know whether this
        // model has ever been fetched, and a wrong guess only costs a printed line.
        let dir = format!("models--{}", model.split(':').next().unwrap_or(model).replace('/', "--"));
        if !cfg.models.join(dir).is_dir() {
            eprintln!("tny: downloading {model} {size}, once, into {}", cfg.models.display());
        }
        let threads = std::env::var("TNY_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(physical_cores);
        let mut cmd = Command::new("llama-server");
        cmd.args(["-hf", model, "--no-mmproj", "--jinja", "--host", "127.0.0.1"])
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

/// F89: a downloaded book is not a mounted book. `remount` killed the pid tny had recorded
/// and announced success, but the running kiwix-serve is often not that process — an orphan
/// from a previous session, or a server someone started by hand — and then the pid file is
/// stale or empty and the kill hits nothing. A pack downloaded 28 MB, reported the library
/// had grown, and the server carried on serving the seventeen books it was started with.
///
/// So: kill what we recorded, then kill whatever still holds our port, then verify the port
/// is down. Nothing is announced here until it is true.
fn remount(cfg: &Cfg) {
    let pidfile = cfg.cache.join("kiwix.pid");
    if let Ok(pid) = std::fs::read_to_string(&pidfile) {
        if let Ok(pid) = pid.trim().parse::<u32>() {
            let _ = Command::new("kill").arg(pid.to_string()).status();
        }
    }
    let _ = std::fs::remove_file(&pidfile);
    let url = format!("{}/", cfg.kiwix);
    for _ in 0..10 {
        if !up(&url) {
            eprintln!("tny: kiwix-serve will restart with the new corpus on the next query");
            return;
        }
        // Our port, our server: a kiwix-serve on KIWIX_PORT owned by this user is one that
        // tny started, now or in an earlier session, and leaving it up hides the new book.
        let _ = Command::new("pkill")
            .args(["-u", &whoami(), "-f", &format!("kiwix-serve --port {KIWIX_PORT}")])
            .status();
        std::thread::sleep(Duration::from_millis(300));
    }
    eprintln!("tny: something else is holding {} — new books stay invisible until it stops", cfg.kiwix);
}

fn whoami() -> String {
    std::env::var("USER").unwrap_or_else(|_| "0".into())
}


/// F89: the catalog's `_ftindex` tag cannot be trusted — it reads `no` for books that search
/// perfectly — so the only honest test of "will this book ever answer anything" is to ask it
/// something. One search per new book, after the server has remounted. A book that answers
/// nothing is not an error; it is a fact the user should hear once, at download time, rather
/// than never.
fn warn_unsearchable(cfg: &Cfg, names: &[String]) {
    if serve_kiwix(cfg).is_err() {
        return;
    }
    let dead: Vec<&String> = names
        .iter()
        .filter(|n| retrieve::search_book(&cfg.kiwix, "the file", n, 1).map_or(false, |r| r.is_empty()))
        .collect();
    if dead.is_empty() {
        return;
    }
    eprintln!(
        "tny: no full-text index, so these will never be searched: {}",
        dead.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
    );
}

/// How many books the running server actually mounted. Comparing this against the ZIM files
/// on disk is the only check that catches every way the two diverge: a pack, a manual copy, a
/// half-finished download, or a server that outlived the library it was started with.
fn mounted_books(cfg: &Cfg) -> Option<usize> {
    let body = ureq::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .get(&format!("{}/catalog/search?count=-1", cfg.kiwix))
        .call()
        .ok()?
        .into_string()
        .ok()?;
    Some(body.matches("<entry>").count())
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

/// F84: the conversation is more than one turn deep. `last.json` kept exactly one exchange,
/// so the third question in a thread could not see the first. Five is a rolling window, not a
/// transcript: the retrieval query still concatenates only the previous question (F29, which
/// was measured), while the model gets the last two exchanges for pronouns and continuity.
fn recent_turns(cfg: &Cfg, n: usize) -> Vec<(String, String)> {
    let raw = std::fs::read_to_string(cfg.cache.join("turns.json")).unwrap_or_default();
    let j: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap_or_default();
    j.iter()
        .rev()
        .take(n)
        .rev()
        .filter_map(|t| Some((t["q"].as_str()?.to_string(), t["a"].as_str()?.to_string())))
        .collect()
}

fn save_turn(cfg: &Cfg, q: &str, a: &str) {
    let mut turns = recent_turns(cfg, 4);
    turns.push((q.to_string(), a.to_string()));
    let j: Vec<serde_json::Value> =
        turns.iter().map(|(q, a)| serde_json::json!({ "q": q, "a": a })).collect();
    let _ = std::fs::write(cfg.cache.join("turns.json"), serde_json::Value::Array(j).to_string());
}

/// F85: asking the same question twice cost 40 s twice. The corpora are static files and the
/// sampling is near-deterministic, so a repeat is a lookup. Keyed by the question and the
/// turn it follows, because a follow-up's answer depends on what came before; `--fresh`
/// bypasses it, and a corpus change invalidates it through the book list in the key.
/// F103: the dials stick. Changing speed in the TUI and finding it back at medium tomorrow
/// makes the setting decorative — this is a daily tool, and the right speed is a property of
/// the machine it runs on, not of one question. Two words in a file; a flag still wins for
/// one invocation.
fn prefs_path(cfg: &Cfg) -> PathBuf {
    cfg.cache.join("prefs")
}

fn load_prefs(cache: &std::path::Path) -> (Option<Mode>, Option<Len>, Option<String>) {
    let Ok(raw) = std::fs::read_to_string(cache.join("prefs")) else { return (None, None, None) };
    let mut it = raw.split_whitespace();
    (
        it.next().and_then(Mode::parse),
        it.next().and_then(Len::parse),
        it.next().map(str::to_string).filter(|s| s.contains('/')),
    )
}

/// A named model, or anything llama-server can fetch. `4b` and `unsloth/Whatever-GGUF:Q4_K_M`
/// are both valid; only the first is one this repo has numbers for.
fn resolve_model(s: &str) -> String {
    MODELS
        .iter()
        .find(|(key, _, _)| *key == s)
        .map(|(_, repo, _)| repo.to_string())
        .unwrap_or_else(|| s.to_string())
}

fn model_name(repo: &str) -> String {
    MODELS
        .iter()
        .find(|(_, r, _)| *r == repo)
        .map(|(key, _, _)| key.to_string())
        .unwrap_or_else(|| repo.split('/').next_back().unwrap_or(repo).to_string())
}

fn save_prefs(cfg: &Cfg) {
    let _ = std::fs::write(
        prefs_path(cfg),
        format!("{} {} {}", cfg.mode.name(), cfg.len.name(), cfg.model),
    );
}

fn cache_key(cfg: &Cfg, question: &str, prev: Option<&(String, String)>) -> String {
    let books = corpus::local(&cfg.zim).join(",");
    let prev_q = prev.map(|(q, _)| q.as_str()).unwrap_or("");
    // F94: the mode is part of the question. Asking the same thing in molasses after fast is
    // a different request, and handing back the fast answer would make the flag do nothing;
    // going back down reuses the earlier answer instead of paying for it twice.
    format!(
        "{}\u{0}{}\u{0}{}\u{0}{}\u{0}{}\u{0}{}",
        question.trim().to_lowercase(),
        prev_q.to_lowercase(),
        books,
        cfg.mode.name(),
        cfg.len.name(),
        cfg.model
    )
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Thirty days — and not the main invalidation. Corpora carry their edition date in the
/// filename (`wikipedia_en_top_nopic_2026-06`), so a refresh lands as a *new* file and changes
/// the book list in the key; `--corpus update` only ever warns, it never replaces anything
/// behind the user's back. The TTL covers what the key cannot see: a file swapped by hand
/// under the same name, and unbounded growth in a cache nobody prunes.
const CACHE_TTL: u64 = 30 * 24 * 60 * 60;

/// F99: the shortlist is cached with the answer. Steering is the repair for a bad answer,
/// and the second time you ask something is exactly when you know it was bad — a cache hit
/// that dropped the other candidates left you with nothing to steer to but a retype.
fn cached(cfg: &Cfg, key: &str) -> Option<(String, Vec<Source>, Vec<Source>)> {
    let raw = std::fs::read_to_string(cfg.cache.join("answers.json")).ok()?;
    let j: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let hit = j.get(key)?;
    if now_secs().saturating_sub(hit["at"].as_u64().unwrap_or(0)) > CACHE_TTL {
        return None;
    }
    let read = |v: &serde_json::Value| -> Vec<Source> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|s| {
                        Some(Source {
                            book: s["book"].as_str()?.to_string(),
                            path: s["path"].as_str()?.to_string(),
                            title: s["title"].as_str()?.to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let sources = read(&hit["s"]);
    // Entries written before the shortlist was cached fall back to what they do have.
    let shortlist = match read(&hit["l"]) {
        l if l.is_empty() => sources.clone(),
        l => l,
    };
    Some((hit["a"].as_str()?.to_string(), sources, shortlist))
}

fn cache_put(cfg: &Cfg, key: &str, answer: &str, sources: &[Source], shortlist: &[Source]) {
    let path = cfg.cache.join("answers.json");
    let raw = std::fs::read_to_string(&path).unwrap_or_default();
    let mut j: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&raw).unwrap_or_default();
    let now = now_secs();
    // Expire on write: nothing else ever walks this file, so the sweep belongs where it is
    // already open. Then a cap, oldest first, for the pathological case of 200 questions in
    // one month.
    j.retain(|_, v| now.saturating_sub(v["at"].as_u64().unwrap_or(0)) <= CACHE_TTL);
    while j.len() >= 200 {
        let oldest = j
            .iter()
            .min_by_key(|(_, v)| v["at"].as_u64().unwrap_or(0))
            .map(|(k, _)| k.clone());
        match oldest {
            Some(k) => {
                j.remove(&k);
            }
            None => break,
        }
    }
    j.insert(
        key.to_string(),
        serde_json::json!({
            "a": answer,
            "at": now,
            "s": sources.iter().map(|s| serde_json::json!({
                "book": s.book, "path": s.path, "title": s.title
            })).collect::<Vec<_>>(),
            "l": shortlist.iter().map(|s| serde_json::json!({
                "book": s.book, "path": s.path, "title": s.title
            })).collect::<Vec<_>>(),
        }),
    );
    let _ = std::fs::write(path, serde_json::Value::Object(j).to_string());
}
